# UNESCO Intangible Heritage Streamlit App

This folder contains the deployable Streamlit dashboard for global intangible heritage visibility and safeguarding pressure based on UNESCO ICH records and World Bank country indicators.

## Project Structure

- `app/Home.py`: Streamlit app entrypoint.
- `src/`: reusable data-loading, transformation, and visualization code.
- `data/processed/`: CSV files required by the deployed app.
- `.streamlit/config.toml`: Streamlit theme configuration.
- `requirements.txt`: Python packages needed by Streamlit Cloud.

Generated figures, report drafts, audit notes, local virtual environments, tests, and raw data caches are intentionally excluded from this deployment folder.

## Run Locally

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app\Home.py
```

## Deploy On Streamlit Cloud

Use `app/Home.py` as the app entrypoint.

The app reads its bundled CSV inputs from `data/processed/`, so no data download step is required during deployment.

If the processed data must be rebuilt later, restore or regenerate `data/raw/`, then run the fetch and transform scripts in `src/`.
