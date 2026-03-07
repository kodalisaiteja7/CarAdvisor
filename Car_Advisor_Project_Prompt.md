# Second-Hand Car Buying Advisor — Full Project Prompt

## PROJECT OVERVIEW

Build a Python-based application where a user inputs:
- **Make** (e.g., Toyota, Honda, BMW)
- **Model** (e.g., Camry, Civic, 3 Series)
- **Year** (e.g., 2018)
- **Mileage** (e.g., 75,000 miles)
- **Optional conditions**: transmission type, drivetrain, engine variant, trim level, region/climate

The system then:
1. Scrapes and aggregates data from multiple automotive sources
2. Performs thorough data analysis
3. Returns a detailed, mileage-aware buying report covering known issues, what to inspect, predicted future failures, and real owner experiences

---

## PHASE 1: DATA COLLECTION & WEB SCRAPING

Build robust, polite scrapers (with rate limiting, retries, User-Agent rotation, and caching) for the following sources. Each scraper should be its own module under `scrapers/`:

### Primary Sources (structured complaint/recall data):

1. **NHTSA (nhtsa.gov)** — Use their public API (https://api.nhtsa.gov/). Pull:
   - Recalls by make/model/year
   - Complaints (TSBs, investigations, defect reports)
   - Crash ratings

2. **CarComplaints.com** — Scrape:
   - Common problems by make/model/year
   - "Worst model year" badges
   - Problem severity ratings, typical mileage at failure, repair costs
   - "CarComplaints Seal of Awesome" or warnings

3. **RepairPal.com** — Scrape:
   - Reliability ratings
   - Common problems with frequency and severity
   - Estimated repair costs
   - Maintenance schedules

### Community & Forum Sources (real owner experiences):

4. **Reddit** — Use Reddit API (or PRAW library). Search subreddits:
   - r/MechanicAdvice, r/UsedCars, r/Cartalk, r/AskMechanics, r/[make-specific subs like r/Toyota, r/BMW]
   - Search queries: "{year} {make} {model} problems", "{make} {model} reliability", "{make} {model} things to check"
   - Extract and categorize recurring complaints, mileage-specific warnings, and buying tips

5. **JustAnswer.com** — Scrape mechanic Q&A threads related to the make/model/year for expert-level diagnostic insights

### Vehicle History & Cost Sources:

6. **Carfax / AutoCheck** — Note: these are paywalled. Build a module that:
   - Scrapes any publicly available summary data
   - Flags common Carfax-reported issues for the model (flood damage prevalence, salvage title rates, etc.)
   - Documents what the user should look for in a Carfax report

7. **Edmunds.com** — Scrape:
   - Long-term road test reports
   - Consumer reviews with problem mentions
   - True Cost to Own (TCO) data
   - Common problems listed in editorial reviews

8. **Consumer Reports** — Note: paywalled. Scrape any publicly available reliability verdicts, or document what to look for

### Technical & Diagnostic Sources:

9. **OBD-Codes.com** — Scrape:
   - Common OBD-II codes for the specific engine/model
   - What each code means, severity, and likely root causes

10. **AutoMD.com** — Scrape:
    - Common repairs and symptoms
    - Diagnostic guidance

11. **TrueDelta.com** — Scrape:
    - Reliability statistics (repair frequency by system)
    - Comparison with competitors
    - Repair cost data

### Consumer Review Sources:

12. **CarSurvey.org** — Scrape:
    - Owner satisfaction ratings
    - Reported faults by category

13. **ConsumerAffairs.com** — Scrape:
    - Consumer reviews and complaints
    - Common issue themes

### Scraper Requirements:
- Use `requests` + `BeautifulSoup` for static pages
- Use `playwright` or `selenium` for JavaScript-rendered pages
- Implement `scrapy` pipelines where appropriate for large-scale crawls
- All scraped data must be cached locally in a SQLite database to avoid redundant requests
- Each scraper must output a standardized JSON schema:

```json
{
  "source": "carcomplaints.com",
  "make": "Toyota",
  "model": "Camry",
  "year": 2018,
  "problems": [
    {
      "category": "Engine",
      "description": "Excessive oil consumption",
      "typical_mileage_range": [60000, 100000],
      "severity": "high",
      "frequency": "common",
      "estimated_repair_cost": "$1500-$3000",
      "complaint_count": 342,
      "user_reports": ["At 72k miles my engine started burning..."]
    }
  ],
  "recalls": [],
  "ratings": {}
}
```

---

## PHASE 2: DATA PROCESSING & ANALYSIS

Build an analysis engine under `analysis/` that:

### 2.1 Data Normalization
- Merge data from all sources into a unified schema
- Deduplicate similar complaints across sources
- Normalize severity ratings to a common scale (1-10)
- Categorize all issues into systems: Engine, Transmission, Electrical, Suspension, Brakes, Body/Paint, Interior, HVAC, Steering, Fuel System, Exhaust, Cooling

### 2.2 Mileage-Based Analysis (CRITICAL)

The output MUST be heavily influenced by the user's input mileage. Build a mileage-aware model:
- Create mileage brackets: 0-30k, 30k-60k, 60k-90k, 90k-120k, 120k-150k, 150k+
- For each known issue, determine the typical mileage range at which it manifests
- When the user inputs mileage, highlight:
  - Issues that commonly appear AT their current mileage (imminent concerns)
  - Issues they've likely already passed (lower risk but verify)
  - Issues approaching in the next 20k-40k miles (upcoming risks)
- If the user changes only the mileage (same make/model/year), the report must meaningfully change to reflect the new mileage context
- Calculate a "risk score" per system that shifts based on mileage

### 2.3 Severity & Frequency Scoring
- Weight issues by: number of complaints, severity of failure, safety impact, repair cost
- Aggregate a composite "Reliability Risk Score" (0-100) for the specific make/model/year/mileage combination
- Rank the top 10 issues by likelihood at the given mileage

### 2.4 Pattern Recognition
- Use NLP (spaCy or similar) on scraped complaint text to:
  - Extract common failure symptoms
  - Identify early warning signs / indicators
  - Cluster similar complaints
  - Extract mileage mentions from free-text complaints

---

## PHASE 3: REPORT GENERATION

Generate a detailed, structured report with these sections:

### 3.1 Vehicle Summary
- Make/Model/Year overview
- Overall reliability rating (aggregated from all sources)
- How this model year compares to adjacent years (is this a "good year" or "bad year"?)
- Known recalls (with completion rates if available)

### 3.2 Pre-Purchase Inspection Checklist
Based on the specific make/model/year/mileage, generate a prioritized checklist:
- **MUST CHECK** items (known failure points at this mileage)
- **RECOMMENDED** checks (emerging issues for this mileage range)
- **STANDARD** checks (general used car inspection items)
- For each item: what to look for, what's normal vs. concerning, estimated cost if bad

### 3.3 Current Mileage Risk Assessment
- Issues that typically manifest at or around the input mileage
- Probability indicator (High / Medium / Low) based on complaint frequency
- What symptoms to look for during a test drive
- Specific diagnostic tests to request (e.g., "ask for a compression test", "check OBD codes for P0420")

### 3.4 Future Issues Forecast (Next 20k, 40k, 60k miles)
- Timeline of predicted issues based on complaint data patterns
- Estimated costs for each predicted repair
- Total estimated maintenance/repair cost for the next 2-3 years
- Comparison: "At 75k miles, you can expect $X in repairs over the next 3 years"

### 3.5 Owner Experience Summary
- Synthesized real owner feedback from Reddit, forums, reviews
- Common praise points (what owners love)
- Common pain points (what owners regret)
- "If I had to buy this car again" sentiment analysis

### 3.6 Red Flags & Deal Breakers
- Known catastrophic failures for this model (e.g., engine failure, transmission failure)
- Signs the car may have been affected by a recall that wasn't completed
- VIN-check recommendations
- Price red flags (if it's priced too low for its condition, why)

### 3.7 Negotiation Ammunition
- Known issues to bring up during negotiation
- Fair price adjustments based on upcoming maintenance needs
- "This car will need $X in maintenance in the next 30k miles" summary

---

## PHASE 4: USER INTERFACE

Build a clean web interface using **Streamlit** or **Flask + a modern frontend**:
- Input form: Make (dropdown), Model (dropdown, filtered by make), Year (dropdown), Mileage (slider + input), Optional conditions (checkboxes/dropdowns)
- Loading state with progress indicator showing which sources are being scraped
- Report displayed in expandable sections with severity color-coding
- Allow re-running with different mileage to compare risk profiles side by side
- Export report as PDF

---

## TECHNICAL REQUIREMENTS

- **Language**: Python 3.11+
- **Database**: SQLite for scraped data cache, with option to upgrade to PostgreSQL

### Project Structure:

```
car-advisor/
├── scrapers/
│   ├── base.py               # Base scraper class
│   ├── nhtsa.py
│   ├── carcomplaints.py
│   ├── repairpal.py
│   ├── reddit.py
│   ├── edmunds.py
│   ├── obd_codes.py
│   ├── truedelta.py
│   ├── justanswer.py
│   ├── carsurvey.py
│   ├── consumeraffairs.py
│   └── automd.py
├── analysis/
│   ├── normalizer.py         # Data normalization
│   ├── mileage_model.py      # Mileage-based risk analysis
│   ├── scorer.py             # Severity & frequency scoring
│   ├── nlp_engine.py         # NLP on complaint text
│   └── aggregator.py         # Cross-source aggregation
├── reports/
│   ├── generator.py          # Report builder
│   └── templates/            # Report templates
├── ui/
│   ├── app.py                # Streamlit/Flask app
│   └── components/           # UI components
├── database/
│   ├── models.py             # SQLAlchemy models
│   ├── cache.py              # Caching layer
│   └── migrations/
├── config/
│   ├── settings.py           # Configuration
│   └── sources.yaml          # Source URLs and selectors
├── tests/
├── requirements.txt
├── README.md
└── main.py
```

### Key Libraries:
- `requests` — HTTP requests
- `beautifulsoup4` — HTML parsing
- `playwright` or `selenium` — JavaScript-rendered pages
- `scrapy` — Large-scale crawling pipelines
- `praw` — Reddit API
- `spacy` — NLP on complaint text
- `pandas` — Data manipulation
- `sqlalchemy` — Database ORM
- `streamlit` or `flask` — Web UI
- `pdfkit` — PDF export
- `tenacity` — Retry logic
- `fake-useragent` — User-Agent rotation

### Error Handling:
- Graceful degradation — if one source fails, still generate report from remaining sources
- Log all failures with source name, URL, and error message

### Rate Limiting:
- Respectful scraping — minimum 2-second delay between requests per domain
- Honor robots.txt where applicable

---

## IMPLEMENTATION ORDER

Build in this sequence to ensure a solid foundation before adding complexity:

1. Database schema and models
2. Base scraper class with caching, rate limiting, retries
3. NHTSA scraper (public API — easiest starting point)
4. CarComplaints scraper (richest complaint data)
5. Data normalizer and aggregator
6. Mileage-based analysis engine
7. Report generator (CLI output first, UI later)
8. Remaining scrapers one by one
9. NLP engine for complaint text analysis
10. Web UI
11. PDF export
12. Testing and refinement

---

## KEY DESIGN PRINCIPLES

- **Mileage is the primary filter**: The same make/model/year with different mileage should produce meaningfully different reports — different risk assessments, different inspection priorities, different cost forecasts.
- **Source diversity**: No single source is authoritative. Weight and cross-reference across all available sources.
- **Extensibility**: Adding a new data source should only require creating a new scraper module that extends the base class — no changes to core analysis logic.
- **Transparency**: Always show which sources contributed to each finding so the user understands where the data came from.
- **Graceful degradation**: Paywalled or blocked sources should not break the system. The report should still generate with whatever data is available.
