import os
import pandas as pd
from sqlalchemy import create_engine

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector


# ============================================================
# CONFIGURATION
# ============================================================

DB_URL = os.getenv(
    "DB_URL",
    "postgresql://quant_user:quant_password@localhost:5432/quant_db"
)

COLLECTION_NAME = "sec_filings"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

engine = create_engine(DB_URL)


# ============================================================
# EMBEDDING MODEL
# ============================================================

print("=" * 60)
print("SEC FILING RAG INDEXER")
print("=" * 60)

print("\nLoading embedding model...")
print(f"Model: {EMBEDDING_MODEL}")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

print("Embedding model loaded.")


# ============================================================
# INDEX SEC FILINGS
# ============================================================

def index_filings():

    print("\n" + "=" * 60)
    print("LOADING SEC FILINGS")
    print("=" * 60)

    query = """
        SELECT
            id,
            ticker,
            published_at,
            clean_text
        FROM financial_text
        WHERE source = 'sec_8k'
          AND clean_text IS NOT NULL
          AND clean_text <> ''
        ORDER BY published_at DESC
    """

    df = pd.read_sql(query, engine)

    print(f"SEC documents loaded: {len(df)}")

    if df.empty:
        print("No SEC filings found.")
        return

    # --------------------------------------------------------
    # Text chunking
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("CHUNKING DOCUMENTS")
    print("=" * 60)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    documents = []

    for _, row in df.iterrows():

        chunks = text_splitter.split_text(row["clean_text"])

        for chunk_index, chunk in enumerate(chunks):

            documents.append(
                {
                    "text": chunk,
                    "metadata": {
                        "ticker": row["ticker"],
                        "date": str(row["published_at"]),
                        "source": "sec_8k",
                        "document_id": int(row["id"]),
                        "chunk_index": chunk_index,
                    },
                }
            )

    print(f"Total chunks created: {len(documents)}")

    if not documents:
        print("No chunks created.")
        return

    # --------------------------------------------------------
    # Create vector store
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("CREATING PGVECTOR INDEX")
    print("=" * 60)

    vector_store = PGVector(
        embeddings=embeddings,
        connection=DB_URL,
        collection_name=COLLECTION_NAME,
        embedding_length=384,
        create_extension=True,
        use_jsonb=True,
    )

    # --------------------------------------------------------
    # Insert documents
    # --------------------------------------------------------

    texts = [doc["text"] for doc in documents]
    metadatas = [doc["metadata"] for doc in documents]

    print("Generating embeddings and storing vectors...")
    print("This may take a while on CPU.")

    vector_store.add_texts(
        texts=texts,
        metadatas=metadatas,
    )

    print("\n" + "=" * 60)
    print("INDEXING COMPLETE")
    print("=" * 60)

    print(f"Documents indexed : {len(df)}")
    print(f"Chunks indexed    : {len(documents)}")
    print(f"Collection        : {COLLECTION_NAME}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    index_filings()