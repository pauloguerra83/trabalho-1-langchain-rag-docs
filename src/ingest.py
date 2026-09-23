import os
import sys

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

PDF_PATH = os.path.join(os.path.dirname(__file__), "..", "document.pdf")
DATABASE_URL = os.getenv("DATABASE_URL")
COLLECTION_NAME = os.getenv("PG_COLLECTION_NAME", "document_collection")
EMBEDDING_MODEL = "text-embedding-3-small"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Erro: defina OPENAI_API_KEY no arquivo .env")
        sys.exit(1)

    if not DATABASE_URL:
        print("Erro: defina DATABASE_URL no arquivo .env")
        sys.exit(1)

    if not os.path.isfile(PDF_PATH):
        print(f"Erro: arquivo PDF não encontrado em '{PDF_PATH}'. "
              f"Coloque o arquivo 'document.pdf' na raiz do projeto.")
        sys.exit(1)

    print(f"Carregando PDF: {PDF_PATH}")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    print(f"Páginas carregadas: {len(documents)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"Chunks gerados: {len(chunks)}")

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    print("Gerando embeddings e salvando no PostgreSQL/pgVector...")
    PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
        pre_delete_collection=True,
    )

    print(f"Ingestão concluída: {len(chunks)} chunks salvos na coleção '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()
