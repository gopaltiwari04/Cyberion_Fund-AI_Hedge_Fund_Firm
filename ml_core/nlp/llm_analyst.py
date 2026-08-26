import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain_core.prompts import PromptTemplate


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

DB_URL = os.getenv(
    "DB_URL",
    "postgresql://quant_user:quant_password@localhost:5432/quant_db"
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not configured. "
        "Add it to your .env file."
    )

COLLECTION_NAME = "sec_filings"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ============================================================
# PROMPT
# ============================================================

PROMPT_TEMPLATE = """
You are an expert Quantitative Researcher and Financial Analyst.

Answer the user's question using ONLY the retrieved SEC filing
context below.

If the context does not contain enough information to answer the
question, say:

"I cannot answer this based on the provided SEC filings."

Do not use outside knowledge.
Do not guess.
Do not hallucinate.

When making a claim, identify the relevant filing date and
specific event, metric, or disclosure from the context.

SEC Filing Context:
-------------------
{context}
-------------------

Question:
{question}

Analyst Report:
"""


# ============================================================
# RAG ANALYST
# ============================================================

def get_analyst_response(ticker: str, query: str):

    print("=" * 60)
    print(f"FINANCIAL RAG ANALYST — {ticker}")
    print("=" * 60)

    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------

    print("\nLoading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    # --------------------------------------------------------
    # Vector database
    # --------------------------------------------------------

    print("Connecting to pgvector...")

    vector_store = PGVector(
        embeddings=embeddings,
        connection=DB_URL,
        collection_name=COLLECTION_NAME,
        embedding_length=384,
        create_extension=True,
        use_jsonb=True,
    )

    # --------------------------------------------------------
    # Retrieve relevant SEC chunks
    # --------------------------------------------------------

    print(f"Searching SEC filings for {ticker}...")

    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 5,
            "filter": {
                "ticker": ticker
            },
        }
    )

    documents = retriever.invoke(query)

    if not documents:
        print("\nNo relevant SEC filing context found.")
        return

    print(f"Retrieved {len(documents)} relevant chunks.")

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context_parts = []

    for i, doc in enumerate(documents, start=1):

        metadata = doc.metadata

        context_parts.append(
            f"""
SOURCE {i}
Ticker: {metadata.get('ticker', 'Unknown')}
Date: {metadata.get('date', 'Unknown')}
Document ID: {metadata.get('document_id', 'Unknown')}
Chunk: {metadata.get('chunk_index', 'Unknown')}

Content:
{doc.page_content}
"""
        )

    context = "\n".join(context_parts)

    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    print("Sending retrieved context to OpenAI...")

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=OPENAI_API_KEY,
    )

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    chain = prompt | llm

    response = chain.invoke(
        {
            "context": context,
            "question": query,
        }
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("ANALYST REPORT")
    print("=" * 60)

    print(response.content)

    print("\n" + "=" * 60)
    print("SOURCES USED")
    print("=" * 60)

    for i, doc in enumerate(documents, start=1):

        print(
            f"{i}. "
            f"{doc.metadata.get('ticker', 'Unknown')} | "
            f"{doc.metadata.get('date', 'Unknown')} | "
            f"document={doc.metadata.get('document_id', 'Unknown')} | "
            f"chunk={doc.metadata.get('chunk_index', 'Unknown')}"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    get_analyst_response(
        "AAPL",
        "What material risks or executive changes were recently disclosed?"
    )