import json
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HISTORY_FILE = PROJECT_ROOT / "outputs" / "history.json"


def _ensure_history_file() -> None:
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not HISTORY_FILE.exists():
        HISTORY_FILE.write_text("[]", encoding="utf-8")


def add_entry(entry: dict) -> None:
    """Append a generation record to outputs/history.json."""
    _ensure_history_file()
    records = get_recent(limit=1000)
    records.append(entry)
    HISTORY_FILE.write_text(json.dumps(records, indent=2), encoding="utf-8")


def get_recent(limit: int = 20) -> list[dict]:
    """Return the most recent history entries (newest first)."""
    _ensure_history_file()
    try:
        records = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        records = []
    records.reverse()
    return records[:limit]


def format_history_table(limit: int = 20) -> list[list[str]]:
    """Format history for Gradio Dataframe display."""
    rows = []
    for entry in get_recent(limit):
        rows.append([
            entry.get("time", ""),
            entry.get("type", ""),
            entry.get("prompt", "")[:80],
            entry.get("output", ""),
        ])
    return rows
