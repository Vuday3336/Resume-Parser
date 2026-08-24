# Resume Intelligence Platform

A resume parsing and candidate-matching system that combines deterministic rule-based extraction
with GenAI, backed by vector search and SQL analytics — with an offline evaluation harness that
actually measures whether the pipeline works, instead of just demoing that it looks like it does.

## Why this isn't just another "resume parser" project

Most portfolio resume parsers are a single LLM call that dumps JSON. This one makes and documents
real engineering tradeoffs:

- **Hybrid extraction**: regex/spaCy handles fields where determinism beats an LLM (emails, phone
  numbers, a curated skills taxonomy — free, instant, exact). An LLM structured-extraction call
  handles the fields that need real language understanding (education, experience narratives,
  skills outside the static taxonomy). Every extraction run logs latency and token cost per
  method, so the tradeoff is measured, not asserted.
- **Two-stage matching**: cheap embedding similarity (pgvector cosine search) shortlists
  candidates against a job description; the expensive LLM rerank — which produces the fit score,
  matched/missing skills, and explanation — only runs on that shortlist. This is what keeps the
  system viable past a handful of resumes.
- **An actual evaluation harness** (`app/evaluation/`): a hand-labeled set of resume/JD pairs with
  gold skill labels and human relevance judgments. It reports extraction precision/recall/F1 and
  the Spearman correlation between the pipeline's ranking and human judgment — for both the
  rule-based-only and hybrid pipelines, so you can show *why* the hybrid approach is better, with
  numbers, in an interview.
- **Real SQL analytics** (`app/analytics/queries.py`): window functions and aggregations, not a
  pandas dump — skill demand, experience-by-skill, score distributions, pipeline funnel.

## Architecture

```
Resume file (PDF/DOCX)
        │
        ▼
  file_extract.py ──► raw text
        │
        ├──► rule_based.py  (regex + spaCy: contact info, taxonomy skills)   ─┐
        │                                                                     ├─► ingestion.py (merge)
        └──► llm_extract.py (OpenAI structured output: education, experience,│
                              narrative skills)                              ─┘
                                                                                 │
                                                                                 ▼
                                                              Supabase Postgres (+ pgvector)
                                                                                 │
                        JD text ──► embed ──► pgvector cosine shortlist (stage 1)
                                                        │
                                                        ▼
                                          llm_score.py — LLM rerank on shortlist (stage 2)
                                                        │
                                                        ▼
                                            match_results table ──► Streamlit dashboard
                                                                     (Upload · Explorer · Matching ·
                                                                      Analytics · Evaluation Report)
```

## Tech stack

Python · OpenAI API (GPT-4o-mini + text-embedding-3-small) · Supabase Postgres + pgvector ·
Streamlit · psycopg2 · pydantic · spaCy · pytest

## Project layout

```
app/
  parsing/        file extraction, rule-based extraction, LLM structured extraction, schemas
  embeddings/      OpenAI embedding wrapper
  matching/        pgvector shortlist (stage 1) + LLM rerank (stage 2)
  analytics/       raw SQL analytical queries
  evaluation/      labeled_set.csv + evaluate.py (precision/recall/F1, Spearman correlation)
  ingestion.py     orchestrates extraction merge + storage
  db.py, config.py
dashboard/
  streamlit_app.py
scripts/
  seed_sample_data.py
supabase/
  schema.sql
tests/
  test_rule_based.py, test_matching.py   (offline, no API key needed)
```

---

## Setup (VS Code, Windows)

### 1. Install prerequisites
- **Python 3.11+** from [python.org/downloads](https://www.python.org/downloads/) (not the
  Microsoft Store version — check "Add python.exe to PATH" during install).
- The **Python extension** for VS Code (Ms-python.python), if not already installed.
- A free **Supabase** account: [supabase.com](https://supabase.com).
- A funded **OpenAI API key**: [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
  (add a few dollars of credit — gpt-4o-mini and text-embedding-3-small are both very cheap;
  the whole seed dataset costs well under $0.10 to run).

### 2. Open the project
Open the `Resume Parser` folder in VS Code (`File > Open Folder`).

### 3. Create a virtual environment
Open a terminal in VS Code (`` Ctrl+` ``) and run:

```bash
python -m venv .venv
.venv\Scripts\activate
```

VS Code should prompt "Select Interpreter" — pick the `.venv` one if it doesn't auto-select.

### 4. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 5. Create your Supabase project
1. In the Supabase dashboard, create a new project (pick a region close to you, set a DB password).
2. Go to **SQL Editor**, paste the contents of `supabase/schema.sql`, and run it. This creates all
   tables plus the `pgvector` extension and the ANN index.
3. Go to **Project Settings → Database → Connection string → URI**, copy the "Session pooler"
   connection string.

### 6. Configure environment variables

```bash
copy .env.example .env
```

Edit `.env` and fill in `OPENAI_API_KEY` and `DATABASE_URL` (the Supabase connection string from
step 5 — replace `[YOUR-PASSWORD]` with your actual DB password).

### 7. Run the offline tests
These don't touch the database or OpenAI — good smoke test that the environment is set up right:

```bash
pytest
```

### 8. Seed sample data
Populates the DB with 5 synthetic candidates and 2 job descriptions so the dashboard isn't empty:

```bash
python scripts/seed_sample_data.py
```

(Add `--no-llm` to skip LLM calls and use rule-based extraction only, e.g. to test without
spending API credit.)

### 9. Run the dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

Opens at `http://localhost:8501`. Try all five tabs — Upload & Parse (upload your own resume PDF),
Candidate Explorer, JD Matching, Analytics, and Evaluation Report (click "Run Evaluation").

---

## Push to GitHub

You're creating the repo yourself, so from the VS Code terminal:

```bash
git init
git add .
git commit -m "Initial commit: resume intelligence platform"
```

Then on GitHub.com: **New repository** → name it (e.g. `resume-intelligence-platform`) → don't
initialize with a README (you already have one) → **Create repository**. Copy the commands GitHub
shows you, which will look like:

```bash
git remote add origin https://github.com/<your-username>/resume-intelligence-platform.git
git branch -M main
git push -u origin main
```

`.env` and `.streamlit/secrets.toml` are already in `.gitignore` — double-check `git status` before
your first commit that neither shows up as staged.

---

## Deploy (free)

**Database** — already live on Supabase from setup step 5, nothing further to do.

**Dashboard — Streamlit Community Cloud:**
1. Push the repo to GitHub (above).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, **New app**.
3. Pick your repo, branch `main`, main file path `dashboard/streamlit_app.py`.
4. Under **Advanced settings → Secrets**, paste the contents of `.streamlit/secrets.toml.example`
   with your real `OPENAI_API_KEY` and `DATABASE_URL` filled in.
5. Deploy. You'll get a public URL like `https://your-app.streamlit.app` — that's what goes on
   your resume/LinkedIn.

---

## Design decisions worth mentioning in an interview

- **Why hybrid extraction instead of LLM-only?** Regex is free, instant, and never hallucinates an
  email address. The evaluation report quantifies the accuracy/cost tradeoff directly.
- **Why two-stage matching instead of LLM-scoring every candidate?** Embedding similarity is
  orders of magnitude cheaper than a chat completion; restricting the LLM call to a shortlist is
  what makes the cost scale sub-linearly with candidate pool size.
- **Why evaluate at all?** Without gold labels and a ranking-quality metric (Spearman correlation
  against human judgment), there's no way to know if the "AI" part of an AI project actually works
  better than the naive baseline — this project measures that explicitly instead of assuming it.

## Suggested resume bullets

- Built a hybrid resume-parsing pipeline (regex/NER + GPT-4o-mini structured extraction) with a
  two-stage vector-search + LLM-rerank matching engine, evaluated against a hand-labeled set
  (skill-extraction F1 + Spearman ranking correlation) to quantify the hybrid approach's
  improvement over a rule-based baseline.
- Designed a Postgres/pgvector schema and SQL analytics layer (window functions, ANN cosine
  search) powering a Streamlit dashboard for candidate exploration and JD matching, deployed on
  Streamlit Community Cloud with Supabase as the managed database.
