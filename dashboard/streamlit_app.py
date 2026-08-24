"""Resume Intelligence Platform — Streamlit dashboard.

Five tabs: Upload & Parse, Candidate Explorer, JD Matching, Analytics, Evaluation Report.
Run with: streamlit run dashboard/streamlit_app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `import app.*` when run directly

import pandas as pd
import plotly.express as px
import streamlit as st

from app.analytics import queries as analytics
from app.db import fetch_all
from app.evaluation.evaluate import run_full_evaluation
from app.ingestion import ingest_resume_file
from app.matching.pipeline import run_matching, store_job_description
from app.parsing.file_extract import extract_text_from_bytes

st.set_page_config(page_title="Resume Intelligence Platform", layout="wide")
st.title("Resume Intelligence Platform")
st.caption("Hybrid rule-based + GenAI resume parsing, vector-based JD matching, and SQL analytics.")

tab_upload, tab_explorer, tab_matching, tab_analytics, tab_eval = st.tabs(
    ["Upload & Parse", "Candidate Explorer", "JD Matching", "Analytics", "Evaluation Report"]
)

# ---------------------------------------------------------------- Upload & Parse
with tab_upload:
    st.subheader("Upload a resume")
    use_llm = st.checkbox("Use hybrid extraction (rule-based + LLM)", value=True,
                           help="Uncheck to run rule-based extraction only — free, no API key needed.")
    uploaded = st.file_uploader("PDF, DOCX, or TXT", type=["pdf", "docx", "txt"])

    if uploaded is not None and st.button("Parse & Store"):
        with st.spinner("Extracting..."):
            raw_text = extract_text_from_bytes(uploaded.getvalue(), uploaded.name)
            resume_id = ingest_resume_file(raw_text, source_filename=uploaded.name, use_llm=use_llm)
        st.success(f"Stored as resume #{resume_id}")

        row = fetch_all(
            """SELECT r.summary, r.total_years_experience, c.full_name, c.email
               FROM resumes r JOIN candidates c ON c.id = r.candidate_id WHERE r.id = %s""",
            (resume_id,),
        )[0]
        skills = fetch_all("SELECT skill_name, source FROM resume_skills WHERE resume_id = %s", (resume_id,))

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Name:** {row['full_name']}")
            st.markdown(f"**Email:** {row['email']}")
            st.markdown(f"**Years experience:** {row['total_years_experience']}")
            st.markdown(f"**Summary:** {row['summary'] or '—'}")
        with col2:
            st.markdown("**Extracted skills** (tagged by extraction source):")
            st.dataframe(pd.DataFrame(skills), hide_index=True, use_container_width=True)

# ---------------------------------------------------------------- Candidate Explorer
with tab_explorer:
    st.subheader("Candidates in the database")
    candidates = fetch_all(
        """SELECT c.id, c.full_name, c.email, r.total_years_experience,
                  string_agg(DISTINCT rs.skill_name, ', ' ORDER BY rs.skill_name) AS skills
           FROM candidates c
           JOIN resumes r ON r.candidate_id = c.id
           LEFT JOIN resume_skills rs ON rs.resume_id = r.id
           GROUP BY c.id, c.full_name, c.email, r.total_years_experience
           ORDER BY c.id DESC"""
    )
    if candidates:
        df = pd.DataFrame(candidates)
        skill_filter = st.text_input("Filter by skill (e.g. 'Python')")
        if skill_filter:
            df = df[df["skills"].str.contains(skill_filter, case=False, na=False)]
        st.dataframe(df, hide_index=True, use_container_width=True)
    else:
        st.info("No candidates yet — upload a resume, or run `python scripts/seed_sample_data.py`.")

# ---------------------------------------------------------------- JD Matching
with tab_matching:
    st.subheader("Match candidates against a job description")
    jd_title = st.text_input("Job title", value="Backend Engineer")
    jd_text = st.text_area("Paste the job description", height=150)
    required_skills_input = st.text_input("Required skills (comma-separated)", value="Python, PostgreSQL, AWS")

    if st.button("Run Matching") and jd_text.strip():
        with st.spinner("Embedding, shortlisting, and scoring with the LLM..."):
            required_skills = [s.strip() for s in required_skills_input.split(",") if s.strip()]
            jd_id = store_job_description(jd_title, None, jd_text, required_skills, None)
            results = run_matching(jd_id, jd_text)

        if not results:
            st.warning("No candidates in the database yet to match against.")
        for rank, match in enumerate(results, start=1):
            with st.expander(f"#{rank} — Resume #{match.resume_id} — fit score {match.llm_fit_score:.0f}/100"):
                st.markdown(f"**Embedding similarity:** {match.embedding_similarity:.3f}")
                st.markdown(f"**Matched skills:** {', '.join(match.matched_skills) or '—'}")
                st.markdown(f"**Missing skills:** {', '.join(match.missing_skills) or '—'}")
                st.markdown(f"**Explanation:** {match.explanation}")
                if match.risk_flags:
                    st.markdown(f"**Risk flags:** {', '.join(match.risk_flags)}")

# ---------------------------------------------------------------- Analytics
with tab_analytics:
    st.subheader("Pipeline & skill analytics (raw SQL)")

    funnel = analytics.pipeline_funnel()
    c1, c2, c3 = st.columns(3)
    c1.metric("Resumes parsed", funnel["parsed"])
    c2.metric("Embedded", funnel["embedded"])
    c3.metric("Scored against a JD", funnel["scored"])

    demand = analytics.skill_demand()
    if demand:
        st.plotly_chart(
            px.bar(pd.DataFrame(demand), x="skill_name", y="candidate_count", title="Skill demand across candidates"),
            use_container_width=True,
        )

    exp_by_skill = analytics.avg_experience_by_skill()
    if exp_by_skill:
        st.plotly_chart(
            px.bar(pd.DataFrame(exp_by_skill), x="skill_name", y="avg_years_experience",
                   title="Average years of experience by skill"),
            use_container_width=True,
        )

    cost = analytics.extraction_cost_summary()
    if cost:
        st.markdown("**Extraction cost/latency by method** (why the pipeline is hybrid, not LLM-only):")
        st.dataframe(pd.DataFrame(cost), hide_index=True, use_container_width=True)

# ---------------------------------------------------------------- Evaluation Report
with tab_eval:
    st.subheader("Offline evaluation: rule-based vs. hybrid pipeline")
    st.caption(
        "Runs the pipeline against a hand-labeled set of resume/JD pairs (app/evaluation/labeled_set.csv) "
        "and measures skill-extraction precision/recall/F1 plus ranking quality (Spearman correlation "
        "against human relevance judgments)."
    )
    include_hybrid = st.checkbox("Include hybrid (LLM) evaluation — uses API credits", value=False)

    if st.button("Run Evaluation"):
        with st.spinner("Evaluating..."):
            reports = run_full_evaluation(include_hybrid=include_hybrid)
        st.dataframe(
            pd.DataFrame([r.__dict__ for r in reports]).drop(columns=["per_jd_correlation"]),
            hide_index=True, use_container_width=True,
        )
        for r in reports:
            st.markdown(f"**{r.method} — per-JD Spearman correlation:**")
            st.json(r.per_jd_correlation)
