"""Populates the database with synthetic sample resumes so the dashboard isn't empty on first
run. Safe to re-run — candidates are upserted on email.

Usage: python scripts/seed_sample_data.py [--no-llm]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ingestion import ingest_resume_file

SAMPLE_RESUMES = [
    (
        "priya_backend.txt",
        """Priya Nair
priya.nair.dev@example.com | +1 555-201-3344 | linkedin.com/in/priyanairdev | github.com/priyanair

Backend engineer with 4 years of experience building production services.

Experience:
Backend Engineer, Meridian Health — Jan 2022 to Present
- Built REST APIs in Python using FastAPI, deployed on AWS with Docker containers.
- Designed PostgreSQL schemas for a multi-tenant SaaS product serving 50k+ users.
- Set up CI/CD with GitHub Actions.

Software Engineer, DataForge — Jun 2020 to Dec 2021
- Developed internal tools with Flask and PostgreSQL.

Education:
B.S. Computer Science, University of Washington, 2016-2020
""",
    ),
    (
        "daniel_datascience.txt",
        """Daniel Osei
daniel.osei.ds@example.com | +1 555-887-2211

Data scientist with 3 years of experience.

Experience:
Data Scientist, Northwind Analytics — Mar 2022 to Present
- Built churn-prediction models in Python using Pandas and scikit-learn.
- Wrote complex SQL queries for feature extraction across a 10M-row warehouse.
- Ran and analyzed A/B tests for pricing experiments.

Junior Analyst, RetailCo — Aug 2020 to Feb 2022
- Built Tableau dashboards for merchandising teams.

Education:
M.S. Statistics, Ohio State University, 2018-2020
""",
    ),
    (
        "mei_genai.txt",
        """Mei Zhang
mei.zhang.ai@example.com | +1 555-440-9981 | github.com/meizhang

AI engineer with 2 years building GenAI features in production.

Experience:
AI Engineer, Solace Labs — Jul 2023 to Present
- Built a RAG pipeline with LangChain and the OpenAI API over internal documentation.
- Wrote prompt-engineering guidelines and an LLM evaluation harness in Python.
- Migrated a legacy support-ticket classifier to a fine-tuned model.

Software Engineer, Solace Labs — Jun 2022 to Jun 2023
- Python backend services, REST API design.

Education:
B.S. Computer Science, UC San Diego, 2018-2022
""",
    ),
    (
        "james_frontend.txt",
        """James Carter
jcarter.dev@example.com | +1 555-772-6630

Frontend developer with 3 years of experience.

Experience:
Frontend Engineer, Bright Retail — Jan 2022 to Present
- Built React and TypeScript interfaces for an e-commerce platform.
- Integrated third-party AI APIs into customer-facing chat widgets.

Education:
B.A. Information Systems, Arizona State University, 2017-2021
""",
    ),
    (
        "sara_senior_backend.txt",
        """Sara Kim
sara.kim.eng@example.com | +1 555-330-4477 | linkedin.com/in/sarakim

Senior backend engineer with 6 years of experience.

Experience:
Senior Backend Engineer, Vertex Systems — 2020 to Present
- Python and Go microservices running on AWS ECS and Lambda.
- PostgreSQL and Redis at scale, full CI/CD with Docker and GitHub Actions.
- Mentored 3 junior engineers.

Backend Engineer, Vertex Systems — 2018 to 2020
- Built the initial REST API layer in Python.

Education:
B.S. Computer Engineering, Georgia Tech, 2014-2018
""",
    ),
]

SAMPLE_JDS = [
    (
        "Backend Engineer",
        "Backend Engineer needed. Must have Python, PostgreSQL, REST API design, Docker, and AWS "
        "experience. 3+ years building production services.",
        ["Python", "PostgreSQL", "REST API", "Docker", "AWS"],
    ),
    (
        "GenAI/LLM Engineer",
        "GenAI Engineer needed. Must know Python, LLM, RAG, LangChain, OpenAI API, and Prompt "
        "Engineering. Building production AI features.",
        ["Python", "LLM", "RAG", "LangChain", "OpenAI API", "Prompt Engineering"],
    ),
]


def main(use_llm: bool = True) -> None:
    for filename, text in SAMPLE_RESUMES:
        resume_id = ingest_resume_file(text, source_filename=filename, use_llm=use_llm)
        print(f"Ingested {filename} -> resume #{resume_id}")

    from app.matching.pipeline import run_matching, store_job_description

    for title, text, skills in SAMPLE_JDS:
        jd_id = store_job_description(title, None, text, skills, None)
        results = run_matching(jd_id, text)
        print(f"Matched {len(results)} candidates against '{title}' (jd #{jd_id})")


if __name__ == "__main__":
    main(use_llm="--no-llm" not in sys.argv)
