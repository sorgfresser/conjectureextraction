import chromadb
import os

CHROMA_HOST = os.getenv("CHROMADB_HOST")
CHROMA_PORT = int(os.getenv("CHROMADB_PORT"))
CHROMA_TOKEN = os.getenv("CHROMADB_TOKEN")
settings = chromadb.Settings(
    chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
    chroma_client_auth_credentials=CHROMA_TOKEN,
    chroma_auth_token_transport_header="Authorization")
client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT, settings=settings)

print([c for c in client.list_collections()])


# Get all documents, delete all documents with "def"
def reset_collection(collection_name: str):
    collection = client.get_or_create_collection(name=collection_name)
    docs = collection.get()
    for doc, doc_id in zip(docs["documents"], docs["ids"]):
        print(doc, doc_id)
        # print(f"Deleting document {doc['id']} with text: {doc['text']}")
        # collection.delete(ids=[doc['id']])
    # collection.delete(ids=["157a5b07-feb6-4f7d-ba56-25f1d1ed88b9"])
    print(f"Collection {collection_name} has been reset.")

reset_collection("lemmasdeepseekkiminaformalizeputnam")
