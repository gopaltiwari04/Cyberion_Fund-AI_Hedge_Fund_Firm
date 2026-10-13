import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Unit tests never touch real services.
os.environ.setdefault("STORAGE_BACKEND", "local")
os.environ.setdefault("LOG_FORMAT", "text")
os.environ.setdefault("RAG_EMBEDDINGS", "off")
