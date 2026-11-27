# Trust Horizon website

React + TypeScript + Vite. All data is precomputed and bundled; the page makes no network requests.

```
python site/scripts/export_site_data.py   # from the repo root: runs/ -> site/src/data, site/src/assets/frames
cd site && npm install && npm run dev     # local preview
npm run build                             # static site in site/dist
```

`export_site_data.py` needs `runs/forecast_tests/error_tau0.5/report.json` (from `python forecast_tests.py`).
Pushing to `main` deploys to GitHub Pages through `.github/workflows/pages.yml`
(repo Settings, Pages, Source: GitHub Actions).
