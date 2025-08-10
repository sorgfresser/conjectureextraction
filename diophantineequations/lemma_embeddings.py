import os
import chromadb
from typing import List, Dict, Union
from uuid import uuid4
from pathlib import Path
import logging

import weave

from diophantineequations.utils import get_lemma_from_file, IS_DISTRIBUTED, IS_MASTER
from diophantineequations.models import LeanFile

logger = logging.getLogger(__name__)

CHROMA_HOST = os.getenv("CHROMADB_HOST")
CHROMA_PORT = int(os.getenv("CHROMADB_PORT")) if os.getenv("CHROMADB_PORT") else None
CHROMA_TOKEN = os.getenv("CHROMADB_TOKEN")


class ReProverEmbeddingFn(chromadb.EmbeddingFunction):
    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        from diophantineequations.reprover import embed, get_model
        get_model()
        return embed(input).detach().cpu().numpy()


class LemmaVectorStore:
    def __init__(self, embedding_fn, collection_name: str = "lemmas", db_path: str = "", hostname: str = "",
                 port: int = -1, settings: Union[chromadb.Settings, None] = None):
        if hostname:
            assert port != -1
            self.client = chromadb.HttpClient(host=hostname, port=port, settings=settings)
        else:
            assert db_path
            self.client = chromadb.PersistentClient(path=db_path)
        if not IS_DISTRIBUTED or IS_MASTER:
            self.collection = self.client.get_or_create_collection(collection_name,
                                                                   metadata={"hnsw:space": "cosine"},
                                                                   embedding_function=embedding_fn)
        else:
            self.collection = self.client.get_collection(collection_name, embedding_function=embedding_fn)

    @weave.op()
    def add_lemma(self, lemma: str, filepath: Path, **metadata) -> bool:
        """
        Add a lemma to the vector store.
        Will skip if the lemma is already present.
        :param lemma: Lemma to add to the vector store
        :return: True if lemma was added, False if lemma was already present.
        """
        if self.exact_match(lemma):
            logger.warning("Lemma %s already present in vector store, skipping", lemma)
            return False
        logger.info("Adding Lemma %s to vector store", lemma)
        self.collection.add(ids=[str(uuid4())], documents=[lemma],
                            metadatas={"filepath": str(filepath.resolve()), **metadata})
        return True

    @weave.op()
    def add_single_file(self, file: Path, **metadata):
        logger.info("Adding single file %s to vector store", file)
        lemma = get_lemma_from_file(file)
        self.add_lemma(lemma, file, **metadata)

    def exact_match(self, lemma: str) -> bool:
        """
        Check if there is an exact match for the given lemma in the vector store.
        :param lemma: Lemma to check for
        :return: True if there is an exact match, False otherwise
        """
        potential_match = self.collection.get(
            where_document={
                "$contains": lemma
            }
        )
        logger.debug("Potential matches for Lemma %s retrieved from vector store: %s", lemma,
                     potential_match["documents"])
        # If there is an exact match, skip adding
        is_exact = any(lemma.strip() == document.strip() for document in potential_match["documents"])
        return is_exact

    def get_premises(self, state: str, k: int) -> List[LeanFile]:
        logger.info("Retrieving %s premises for state %s", k, state)
        result = self.collection.query(
            query_texts=[state],
            n_results=k
        )
        # we get one list for each query text, so only one list in this case
        filepaths = [metadata["filepath"] for metadata in result["metadatas"][0]]
        documents = result["documents"][0]
        logger.debug("Retrieved filepaths %s and documents %s", filepaths, documents)
        proofs = []
        for path in filepaths:
            with open(path, "r") as f:
                proofs.append(f.read())
        lean_files = [LeanFile(Path(filepath), document, full_proof=proof) for filepath, document, proof in
                      zip(filepaths, documents, proofs, strict=True)]
        return lean_files

    @classmethod
    def from_embedding_fn(cls, embedding_fn: chromadb.EmbeddingFunction, collection_name: str = "lemmas"):
        if CHROMA_HOST is not None or CHROMA_PORT is not None:
            assert CHROMA_HOST
            assert CHROMA_PORT
            settings = chromadb.Settings(
                chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
                chroma_client_auth_credentials=CHROMA_TOKEN,
                chroma_auth_token_transport_header="Authorization"
            ) if CHROMA_TOKEN else None
            logger.info("Settings %s", settings)
            store = cls(embedding_fn, collection_name, hostname=CHROMA_HOST, port=CHROMA_PORT, settings=settings)
        else:
            store = cls(embedding_fn, collection_name, "vectorstore")
        return store

    @classmethod
    def from_directory(cls, root_path: Path, embedding_fn: chromadb.EmbeddingFunction, collection_name: str = "lemmas"):
        store = cls.from_embedding_fn(embedding_fn, collection_name)
        successes = 0
        logger.info("Globbing in path %s", root_path)
        for lemma_file in root_path.rglob("*.lean"):
            lemma = get_lemma_from_file(lemma_file)
            logger.debug("Received lemma %s", lemma)
            if store.add_lemma(lemma, lemma_file):
                successes += 1
        logger.info("Added %s lemmas to vector store", successes)
        return store


if __name__ == '__main__':
    # store = LemmaVectorStore("vectorstore", ReProverEmbeddingFn())
    # store.add_lemma("""<a>theorem conjecture_minimality_implies_bound_on_the_other_root</a> (a b k : Nat) (c : Int)
    #     (h : a > 0 ∧ b > 0 ∧ k > 0 ∧ a ≥ b)
    #     (hk : a^2 - k * b * a + b^2 - k = 0)
    #     (hc : c = k * b - a) :
    #     c ≤ 0""")
    # print(store.get_premises("c ≤ 0", 5))
    store = LemmaVectorStore.from_directory(
        Path("/home/simon/PycharmProjects/diophantineequations/imo1988q6project/Imo1988q6project"),
        ReProverEmbeddingFn())
    print(store.get_premises("c ≤ 0", 5))
