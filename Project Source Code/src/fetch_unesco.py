"""Fetch UNESCO DataHub Intangible Heritage List records."""

from __future__ import annotations

import csv
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
API_URL = "https://data.unesco.org/api/explore/v2.1/catalog/datasets/ich001/records"


def fetch_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_records(limit: int = 100, sleep_seconds: float = 0.15) -> list[dict]:
    rows: list[dict] = []
    total_count = None
    offset = 0
    while total_count is None or len(rows) < total_count:
        query = urllib.parse.urlencode({"limit": limit, "offset": offset})
        payload = fetch_json(f"{API_URL}?{query}")
        total_count = int(payload["total_count"])
        batch = payload["results"]
        rows.extend(batch)
        offset += limit
        time.sleep(sleep_seconds)
    return rows[:total_count]


def write_csv(records: list[dict], path: Path) -> None:
    fieldnames = [
        "uuid",
        "ich_public_ref",
        "inscription_year",
        "title_en",
        "description_en",
        "type_of_element_en",
        "type_acronym",
        "countries",
        "concepts_primary_names",
        "concepts_secondary_names",
        "whc",
        "http_url_en",
        "main_image_url",
        "main_image_copyright",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in fieldnames})


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    records = fetch_records()
    (RAW_DIR / "unesco_ich001.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(records, RAW_DIR / "unesco_ich001.csv")
    print(f"Fetched {len(records)} UNESCO records")


if __name__ == "__main__":
    main()
