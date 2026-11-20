#!/usr/bin/env python3
"""Local web UI for labeling employers with a 2-digit NAICS sector.

Run with::

    uv run python labeling/app.py
    uv run python labeling/app.py --port 8000 --db data/union_elections.duckdb

Open http://127.0.0.1:8000/ .  Each card shows one unlabeled employer (with its
union and bargaining-unit description).  Pick a sector (or type a finer code),
optionally ask the LLM for the top 3 suggestions, then "Save & next".

Labels are written to ``naics_labels`` in the project DuckDB and propagated to
every case of the labeled employer.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Sequence

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from store import LabelStore, PROJECT_ROOT, DEFAULT_DB

LOG = logging.getLogger("naics-labeling")

STATIC_DIR = Path(__file__).resolve().parent / "static"
SECTORS_FILE = STATIC_DIR / "naics_sectors.json"

# OpenCode Go is an OpenAI-compatible chat endpoint.  ``gemini-4-argon`` is
# served over /chat/completions and works with the ``oc_sk_`` key.
OPENCODE_BASE_URL = os.environ.get(
    "OPENCODE_SUGGEST_BASE_URL", "https://opencode.ai/zen/go/v1"
)
OPENCODE_MODEL = os.environ.get("OPENCODE_SUGGEST_MODEL", "gemini-4-argon")
OPENCODE_SESSION = os.environ.get("OPENCODE_SESSION", "naics-labeling")
BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)


def load_dotenv(path: Path = PROJECT_ROOT / ".env") -> None:
    """Load simple KEY=VALUE pairs from ``.env`` without extra dependencies."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'").strip('"'))


load_dotenv()


def opencode_key() -> str | None:
    return os.environ.get("OPENCODE_GO_API") or os.environ.get("OPENCODE_API_KEY")


def load_sectors() -> dict[str, Any]:
    with SECTORS_FILE.open(encoding="utf-8") as handle:
        return json.load(handle)


store: LabelStore | None = None


def get_store() -> LabelStore:
    global store
    if store is None:
        store = LabelStore(DEFAULT_DB)
    return store


app = FastAPI(title="NAICS labeling")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class LabelItem(BaseModel):
    employer_name: str
    naics: str = Field(..., min_length=1)
    naics_title: str | None = None


class LabelBatch(BaseModel):
    labels: list[LabelItem]


class SuggestRequest(BaseModel):
    employer_name: str
    city: str | None = None
    state: str | None = None
    election_union_name: str | None = None
    voting_unit_description: str | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/sectors")
def api_sectors() -> dict[str, Any]:
    return load_sectors()


@app.get("/api/sample")
def api_sample(n: int = 10) -> dict[str, Any]:
    n = max(1, min(n, 50))
    return {"cases": get_store().sample(n)}


@app.post("/api/label")
def api_label(batch: LabelBatch) -> dict[str, int]:
    saved = get_store().save_many(
        label.model_dump() for label in batch.labels
    )
    return {"saved": saved}


@app.get("/api/progress")
def api_progress() -> dict[str, int]:
    return get_store().progress()


@app.post("/api/suggest")
def api_suggest(request: SuggestRequest) -> dict[str, Any]:
    try:
        return suggest_naics(request)
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI
        LOG.warning("suggestion failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# LLM suggestions
# ---------------------------------------------------------------------------


def suggest_naics(request: SuggestRequest) -> dict[str, Any]:
    key = opencode_key()
    if not key:
        raise RuntimeError(
            "OPENCODE_GO_API is not set (add it to .env or the environment)"
        )

    sectors = load_sectors()["sectors"]
    catalog = "\n".join(f'{s["code"]}: {s["title"]}' for s in sectors)
    location = ", ".join(
        part for part in (request.city, request.state) if part
    ) or "unknown"
    prompt = (
        "You are helping a human label a U.S. employer with a 2022 NAICS "
        "two-digit sector (an industry category).\n\n"
        f"Employer: {request.employer_name}\n"
        f"Location: {location}\n"
        f"Union: {request.election_union_name or 'unknown'}\n"
        "Bargaining unit description: "
        f"{request.voting_unit_description or 'unknown'}\n\n"
        "Choose the three most likely two-digit sectors from this list:\n"
        f"{catalog}\n\n"
        'Return ONLY a JSON array of exactly 3 objects, each with "code" '
        '(from the list above, e.g. "62" or "31-33"), "title", "probability" '
        "(a number from 0 to 1 giving the chance the employer belongs to that "
        'sector), and a one-sentence "rationale". The three probabilities '
        "must sum to 1. No markdown, no extra text."
    )

    payload = {
        "model": OPENCODE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "User-Agent": BROWSER_UA,
        "x-opencode-session": OPENCODE_SESSION,
    }
    request_http = urllib.request.Request(
        f"{OPENCODE_BASE_URL.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request_http, timeout=90) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc

    message = body["choices"][0]["message"]
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    return {
        "suggestions": parse_suggestions(content, sectors),
        "raw": content,
        "reasoning": reasoning,
    }


def _parse_probability(value: Any) -> float | None:
    """Coerce a model-supplied probability to 0..1, tolerating percentages."""
    if value is None:
        return None
    try:
        probability = float(value)
    except (TypeError, ValueError):
        return None
    if 1.0 < probability <= 100.0:
        probability = probability / 100.0
    return max(0.0, min(1.0, probability))


def parse_suggestions(
    content: str, sectors: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    titles = {str(s["code"]): str(s["title"]) for s in sectors}
    text = content.strip()

    # Strip ```json fences if the model added them.
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            parsed = []
        suggestions: list[dict[str, Any]] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            code = str(item.get("code", "")).strip()
            if code in titles:
                suggestions.append(
                    {
                        "code": code,
                        "title": titles[code],
                        "probability": _parse_probability(
                            item.get("probability")
                        ),
                        "rationale": str(item.get("rationale", "")).strip(),
                    }
                )
            if len(suggestions) == 3:
                break
        if suggestions:
            return suggestions

    # Fallback: pull any sector codes out of the raw text.
    seen: list[str] = []
    for match in re.findall(r"\b(31-33|44-45|48-49|\d{2})\b", text):
        if match in titles and match not in seen:
            seen.append(match)
        if len(seen) == 3:
            break
    return [
        {"code": c, "title": titles[c], "probability": None, "rationale": ""}
        for c in seen
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label employers with a 2-digit NAICS sector.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help="DuckDB database containing analytical_elections",
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind address")
    parser.add_argument("--port", type=int, default=8000, help="bind port")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="enable debug logging"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    global store
    store = LabelStore(args.db)
    progress = store.progress()
    LOG.info(
        "Labeled %s of %s employers", progress["labeled"], progress["total"]
    )
    if not opencode_key():
        LOG.warning(
            "OPENCODE_GO_API not set; the 'Suggest' button will be disabled."
        )

    LOG.info("Serving on http://%s:%s/", args.host, args.port)
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
