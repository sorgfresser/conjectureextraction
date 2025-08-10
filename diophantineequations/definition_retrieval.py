import chromadb
from logging import getLogger
from uuid import uuid4
from pathlib import Path
from typing import Callable, List, Union
from openai import Client
from diophantineequations.prompts import INFORMALIZATION, INFORMALIZATION_FEW_SHOT
from diophantineequations.models import LeanFile
import os
from diophantineequations.utils import IS_DISTRIBUTED, IS_MASTER
from tqdm import tqdm
import weave

logger = getLogger(__name__)
client = Client()

SKIPS = ["test", "Test", "MathlibTest", "Cache", "ImportGraphTest", "LeanSearchClientTest", "BatteriesTest",
         "AesopTest", "Archive", "Counterexamples", "docs", "scripts", "Shake", "widget"]

CHROMA_HOST = os.getenv("CHROMADB_HOST")
CHROMA_PORT = int(os.getenv("CHROMADB_PORT")) if os.getenv("CHROMADB_PORT") else None
CHROMA_TOKEN = os.getenv("CHROMADB_TOKEN")

@weave.op()
def informalize(definition: str) -> str:
    messages = [
        {
            "role": "user",
            "content": INFORMALIZATION + "\n\n" + INFORMALIZATION_FEW_SHOT + "\n\n" + "```lean4" + definition + "```"
        }
    ]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    result = response.choices[0].message.content
    logger.info("Informalized definition: %s", result)
    return result


class DefinitionVectorStore:
    def __init__(self, embedding_fn: chromadb.EmbeddingFunction, informalizer: Callable[[str], str], db_path: str = "",
                 hostname: str = "", port: int = -1, settings: Union[chromadb.Settings, None] = None):
        if hostname:
            assert port != -1
            self.client = chromadb.HttpClient(host=hostname, port=port, settings=settings)
        else:
            assert db_path
            self.client = chromadb.PersistentClient(path=db_path)
        if not IS_DISTRIBUTED or IS_MASTER:
            self.collection = self.client.get_or_create_collection("definitions",
                                                                   metadata={"hnsw:space": "cosine"},
                                                                   embedding_function=embedding_fn)
        else:
            self.collection = self.client.get_collection("definitions", embedding_function=embedding_fn)
        self.informalizer = informalizer

    @weave.op()
    def add_definition(self, informal: str, definition: str, filepath: str) -> bool:
        """
        Add a definition to the vector store.
        Will skip if the definition is already present.
        :param informal: Informal definition to add to the vector store
        :param definition: Definition to add to the vector store
        :param filepath: Filepath of the lean file where the definition was found
        :return: True if definition was added, False if definition was already present.
        """
        if self.exact_match(definition):
            logger.warning("Definition %s already present in vector store, skipping", definition)
            return False
        logger.info("Adding Definition %s to vector store", definition)
        self.collection.add(ids=[str(uuid4())], documents=[informal], metadatas={"definition": definition,
                                                                                 "filepath": filepath})
        return True

    @weave.op()
    def add_single_file(self, file: Path):
        logger.info("Trying to add single file %s to vector store", file)
        data = file.read_text()
        if self.exact_match(data):
            logger.warning("Definition already present in vector store, skipping")
            return
        logger.info("Adding single file %s to vector store", file)
        nl_definition = self.informalizer(data)
        self.add_definition(nl_definition, data, str(file.absolute()))

    def exact_match(self, definition: str) -> bool:
        """
        Check if there is an exact match for the given definition in the vector store.
        :param definition: Definition to check for
        :return: True if there is an exact match, False otherwise
        """
        potential_match = self.collection.get(
            where={
                "definition": definition
            }
        )
        logger.debug("Potential matches for definition: %s \nretrieved from vector store: %s", definition,
                     potential_match["metadatas"])
        # If there is an exact match, skip adding
        is_exact = any(
            definition.strip() == metadata["definition"].strip() for metadata in potential_match["metadatas"])
        return is_exact

    @weave.op()
    def get_definitions(self, informal: str, k: int) -> List[LeanFile]:
        """
        Get the definitions corresponding to the informal text (usually a conjecture)
        :param informal: Informal text to get the definitions for
        :return: The definitions corresponding to the informal text
        """
        logger.info("Retrieving definition for Informal %s", informal)
        result = self.collection.query(
            query_texts=[informal],
            n_results=k
        )
        assert result is not None
        files = [LeanFile(Path(metadata["filepath"]), metadata["definition"]) for metadata in
                 result["metadatas"][0]]
        logger.debug("Retrieved definitions: %s", files)
        return files

    @classmethod
    def from_directory(cls, directory_path: Path, embedding_fn: chromadb.EmbeddingFunction,
                       informalizer: Callable[[str], str]):
        if IS_DISTRIBUTED:
            store = cls(embedding_fn, informalizer, hostname=CHROMA_HOST, port=CHROMA_PORT)
        else:
            store = cls(embedding_fn, informalizer, "vectorstore")
        logger.info("Globbing in path %s", directory_path)
        for definition_file in tqdm(list(directory_path.rglob("*.lean"))):
            # .lake is allowed if it is in directory path (i.e. directory path can be inside .lake),
            # but do not go for .lake as a subdir of directory path
            relative_to = definition_file.relative_to(directory_path)
            if ".lake" in relative_to.parts:
                continue
            if any(skip in relative_to.parts for skip in SKIPS):
                continue
            store.add_single_file(definition_file)
        return store

    @classmethod
    def from_file(cls, embedding_fn: chromadb.EmbeddingFunction, informalizer: Callable[[str], str]):
        if CHROMA_HOST is not None or CHROMA_PORT is not None:
            assert CHROMA_HOST
            assert CHROMA_PORT
            settings = chromadb.Settings(
                chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
                chroma_client_auth_credentials=CHROMA_TOKEN,
                chroma_auth_token_transport_header="Authorization"
            ) if CHROMA_TOKEN else None
            logger.info("Settings %s", settings)
            store = cls(embedding_fn, informalizer, hostname=CHROMA_HOST, port=CHROMA_PORT, settings=settings)
        else:
            if not Path("vectorstore").exists():
                store = None
            else:
                store = cls(embedding_fn, informalizer, "vectorstore")
        return store


if __name__ == "__main__":
    from chromadb.utils import embedding_functions
    import logging

    logging.basicConfig(level=logging.DEBUG)
    store = DefinitionVectorStore.from_directory(Path("../FLT"), embedding_functions.DefaultEmbeddingFunction(),
                                                 informalize)
    definitions = store.get_definitions("quaternion algebra", 5)
    for definition in definitions:
        print(definition.import_string(Path("../FLT")))
