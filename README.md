# Car Advisor

A Python-based used car buying advisor that scrapes and aggregates data from multiple automotive sources, performs mileage-aware analysis, and generates detailed pre-purchase reports.

## Features

- **Multi-source data collection** from NHTSA, CarComplaints.com, and more
- **Mileage-aware risk analysis** — reports shift meaningfully based on the vehicle's current mileage
- **Pre-purchase inspection checklists** prioritized by known failure points
- **Future issue forecasting** with cost estimates for the next 20k/40k/60k miles
- **Negotiation ammunition** backed by real complaint data
- **Polished web UI** with Flask + Tailwind CSS

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the web UI
python main.py serve

# Or run a CLI report
python main.py --make Toyota --model Camry --year 2018 --mileage 75000
```

## Project Structure

```
├── scrapers/          # Data collection modules (one per source)
├── analysis/          # Data normalization, mileage model, scoring
├── reports/           # Report generation
├── ui/                # Flask web application
├── database/          # SQLAlchemy models and caching
├── config/            # Settings and configuration
├── tests/             # Test suite
└── main.py            # CLI and server entry point
```

## Adding a New Data Source

1. Create a new scraper in `scrapers/` that extends `BaseScraper`
2. Implement the `scrape(make, model, year)` method returning the standardized schema
3. Register it in the scraper registry — no changes to analysis or report logic needed
