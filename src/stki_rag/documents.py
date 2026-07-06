"""Document loading and management."""

from pathlib import Path
from typing import Dict


def load_documents(data_dir: Path | str = "data") -> Dict[str, str]:
    """Load markdown documents from data directory.

    Args:
        data_dir: Path to directory containing markdown files.

    Returns:
        Dictionary mapping document names (without .md) to content.
    """
    data_path = Path(data_dir)
    docs = {}
    for path in data_path.glob("*.md"):
        docs[path.stem] = path.read_text(encoding="utf-8").strip()
    return docs


DOCS = load_documents()