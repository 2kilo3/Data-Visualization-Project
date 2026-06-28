"""Fetch World Bank country metadata and indicators."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
COUNTRY_URL = "https://api.worldbank.org/v2/country"
INDICATOR_URL = "https://api.worldbank.org/v2/country/all/indicator/{indicator}"

INDICATORS = {
    "population": "SP.POP.TOTL",
    "gdp_current_usd": "NY.GDP.MKTP.CD",
    "gdp_per_capita_usd": "NY.GDP.PCAP.CD",
    "urban_population_pct": "SP.URB.TOTL.IN.ZS",
}


def fetch_json(url: str) -> list:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_all_pages(base_url: str, params: dict) -> list[dict]:
    query = urllib.parse.urlencode({**params, "page": 1})
    payload = fetch_json(f"{base_url}?{query}")
    meta = payload[0]
    rows = payload[1]
    for page in range(2, int(meta["pages"]) + 1):
        query = urllib.parse.urlencode({**params, "page": page})
        rows.extend(fetch_json(f"{base_url}?{query}")[1])
    return rows


def fetch_country_metadata() -> pd.DataFrame:
    rows = fetch_all_pages(COUNTRY_URL, {"format": "json", "per_page": 400})
    records = []
    for row in rows:
        if row["region"]["value"] == "Aggregates":
            continue
        records.append(
            {
                "iso3": row["id"],
                "iso2": row["iso2Code"],
                "country_name": row["name"],
                "region": row["region"]["value"].strip(),
                "income_level": row["incomeLevel"]["value"],
                "capital_city": row["capitalCity"],
                "longitude": row["longitude"],
                "latitude": row["latitude"],
            }
        )
    return pd.DataFrame(records)


def fetch_indicator(indicator: str, column: str, year: int = 2023) -> pd.DataFrame:
    rows = fetch_all_pages(
        INDICATOR_URL.format(indicator=indicator),
        {"format": "json", "per_page": 300, "date": year},
    )
    records = []
    for row in rows:
        iso3 = row.get("countryiso3code")
        if not iso3 or len(iso3) != 3:
            continue
        records.append({"iso3": iso3, column: row.get("value")})
    return pd.DataFrame(records)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    countries = fetch_country_metadata()
    for column, indicator in INDICATORS.items():
        values = fetch_indicator(indicator, column)
        values.to_csv(RAW_DIR / f"worldbank_{column}.csv", index=False, encoding="utf-8-sig")
        countries = countries.merge(values, on="iso3", how="left")
    countries.to_csv(RAW_DIR / "worldbank_countries.csv", index=False, encoding="utf-8-sig")
    print(f"Fetched {len(countries)} World Bank country rows")


if __name__ == "__main__":
    main()
