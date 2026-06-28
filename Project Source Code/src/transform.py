"""Data transformation helpers for UNESCO intangible heritage records."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from pathlib import Path
import sys
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_io import read_csv_preserve_codes  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


TYPE_LABELS = {
    "RL": "Representative List",
    "USL": "Urgent Safeguarding List",
    "Art18": "Good Safeguarding Practices",
}


def ensure_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return [item for item in value if item not in (None, "")]
    if pd.isna(value):
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                import ast

                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, list):
                    return [item for item in parsed if item not in (None, "")]
            except (SyntaxError, ValueError):
                pass
        return [part.strip() for part in stripped.split(",") if part.strip()]
    return [value]


def normalize_elements(records: Iterable[dict]) -> pd.DataFrame:
    rows = []
    for record in records:
        element_id = str(record.get("uuid") or record.get("element_id") or record.get("ich_public_ref"))
        rows.append(
            {
                "element_id": element_id,
                "ich_public_ref": record.get("ich_public_ref"),
                "title_en": record.get("title_en"),
                "description_en": record.get("description_en"),
                "inscription_year": pd.to_numeric(record.get("inscription_year"), errors="coerce"),
                "type_acronym": record.get("type_acronym"),
                "type_label": TYPE_LABELS.get(record.get("type_acronym"), record.get("type_of_element_en")),
                "countries": as_list(record.get("countries")),
                "concepts_primary_names": as_list(record.get("concepts_primary_names")),
                "concepts_secondary_names": as_list(record.get("concepts_secondary_names")),
                "whc": str(record.get("whc")).lower() == "true",
                "http_url_en": record.get("http_url_en"),
                "main_image_url": record.get("main_image_url"),
                "main_image_copyright": record.get("main_image_copyright"),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["inscription_year"] = df["inscription_year"].astype("Int64")
        df = df.sort_values(["inscription_year", "element_id"], kind="stable").reset_index(drop=True)
    return df


def expand_element_countries(elements: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in elements.to_dict("records"):
        countries = as_list(row.get("countries"))
        for iso2 in countries:
            rows.append(
                {
                    "element_id": row["element_id"],
                    "iso2": str(iso2).upper(),
                    "title_en": row.get("title_en"),
                    "inscription_year": row.get("inscription_year"),
                    "type_acronym": row.get("type_acronym"),
                }
            )
    return pd.DataFrame(
        rows,
        columns=["element_id", "iso2", "title_en", "inscription_year", "type_acronym"],
    )


def expand_concepts(elements: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in elements.to_dict("records"):
        concepts = as_list(row.get("concepts_primary_names"))
        for index, concept in enumerate(concepts, start=1):
            rows.append(
                {
                    "element_id": row["element_id"],
                    "concept_name": str(concept),
                    "concept_rank": index,
                    "inscription_year": row.get("inscription_year"),
                    "type_acronym": row.get("type_acronym"),
                }
            )
    return pd.DataFrame(
        rows,
        columns=["element_id", "concept_name", "concept_rank", "inscription_year", "type_acronym"],
    )


def build_country_summary(element_countries: pd.DataFrame, country_lookup: pd.DataFrame) -> pd.DataFrame:
    if element_countries.empty:
        return pd.DataFrame()

    country_counts = (
        element_countries.groupby("iso2")
        .agg(
            element_count=("element_id", "nunique"),
            urgent_count=("type_acronym", lambda values: int((values == "USL").sum())),
        )
        .reset_index()
    )

    per_element_country_count = element_countries.groupby("element_id")["iso2"].nunique()
    multinational_ids = set(per_element_country_count[per_element_country_count > 1].index)
    multinational_counts = (
        element_countries[element_countries["element_id"].isin(multinational_ids)]
        .groupby("iso2")["element_id"]
        .nunique()
        .rename("multinational_count")
        .reset_index()
    )

    summary = country_counts.merge(multinational_counts, on="iso2", how="left")
    summary["multinational_count"] = summary["multinational_count"].fillna(0).astype(int)
    summary = summary.merge(country_lookup, on="iso2", how="left")
    summary["population"] = pd.to_numeric(summary.get("population"), errors="coerce")
    summary["element_count"] = summary["element_count"].astype(int)
    summary["urgent_count"] = summary["urgent_count"].astype(int)
    summary["urgent_share"] = summary["urgent_count"] / summary["element_count"]
    summary["elements_per_million"] = summary.apply(
        lambda row: row["element_count"] / (row["population"] / 1_000_000)
        if pd.notna(row["population"]) and row["population"] > 0
        else pd.NA,
        axis=1,
    )
    summary["metric_missing"] = summary["population"].isna() | (summary["population"] <= 0)
    return summary.sort_values(["element_count", "iso2"], ascending=[False, True]).reset_index(drop=True)


def build_country_edges(elements: pd.DataFrame) -> pd.DataFrame:
    edge_elements: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in elements.to_dict("records"):
        countries = sorted({str(country).upper() for country in as_list(row.get("countries"))})
        for source, target in combinations(countries, 2):
            edge_elements[(source, target)].append(row["element_id"])

    rows = [
        {
            "source_iso2": source,
            "target_iso2": target,
            "weight": len(element_ids),
            "shared_elements": ";".join(element_ids),
        }
        for (source, target), element_ids in edge_elements.items()
    ]
    return pd.DataFrame(rows, columns=["source_iso2", "target_iso2", "weight", "shared_elements"]).sort_values(
        ["weight", "source_iso2", "target_iso2"], ascending=[False, True, True], ignore_index=True
    )


def build_yearly_summary(elements: pd.DataFrame) -> pd.DataFrame:
    return (
        elements.groupby(["inscription_year", "type_acronym"], dropna=False)
        .agg(element_count=("element_id", "nunique"))
        .reset_index()
        .sort_values(["inscription_year", "type_acronym"], kind="stable")
    )


def main() -> None:
    ensure_dirs()
    records_path = RAW_DIR / "unesco_ich001.json"
    country_path = RAW_DIR / "worldbank_countries.csv"
    if not records_path.exists() or not country_path.exists():
        raise FileNotFoundError("Run fetch_unesco.py and fetch_worldbank.py before transform.py")

    import json

    records = json.loads(records_path.read_text(encoding="utf-8"))
    elements = normalize_elements(records)
    countries = read_csv_preserve_codes(country_path)
    element_countries = expand_element_countries(elements)
    concepts = expand_concepts(elements)
    country_summary = build_country_summary(element_countries, countries)
    yearly_summary = build_yearly_summary(elements)
    edges = build_country_edges(elements)

    elements.to_csv(PROCESSED_DIR / "elements.csv", index=False, encoding="utf-8-sig")
    element_countries.to_csv(PROCESSED_DIR / "element_countries.csv", index=False, encoding="utf-8-sig")
    concepts.to_csv(PROCESSED_DIR / "element_concepts.csv", index=False, encoding="utf-8-sig")
    country_summary.to_csv(PROCESSED_DIR / "country_summary.csv", index=False, encoding="utf-8-sig")
    yearly_summary.to_csv(PROCESSED_DIR / "yearly_summary.csv", index=False, encoding="utf-8-sig")
    edges.to_csv(PROCESSED_DIR / "country_edges.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
