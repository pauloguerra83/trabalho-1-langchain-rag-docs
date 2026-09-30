import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
COLLECTION_NAME = os.getenv("PG_COLLECTION_NAME", "document_collection")
EMBEDDING_MODEL = "text-embedding-3-small"

_embeddings = None
_vector_store = None


def _get_vector_store():
    global _embeddings, _vector_store
    if _vector_store is None:
        if not DATABASE_URL:
            raise RuntimeError("Defina DATABASE_URL no arquivo .env")
        _embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        _vector_store = PGVector(
            embeddings=_embeddings,
            collection_name=COLLECTION_NAME,
            connection=DATABASE_URL,
            use_jsonb=True,
        )
    return _vector_store


def search(query: str, k: int = 10):
    vector_store = _get_vector_store()
    return vector_store.similarity_search_with_score(query, k=k)
