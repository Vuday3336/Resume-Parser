"""A curated skills taxonomy used by the rule-based extractor.

This is intentionally a plain, editable list rather than a black box — in a real system this
would be backed by a database table (see `skills` in supabase/schema.sql) so it can grow without
a code change. Grouped by category purely for readability; matching is case-insensitive and
uses word boundaries so "R" doesn't match inside "Order", etc.
"""

SKILLS_TAXONOMY: dict[str, list[str]] = {
    "languages": [
        "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Go", "Rust", "SQL", "R",
        "Scala", "Kotlin", "Swift", "PHP", "Ruby", "MATLAB", "Bash",
    ],
    "ml_ai": [
        "Machine Learning", "Deep Learning", "PyTorch", "TensorFlow", "Keras", "scikit-learn",
        "NLP", "Computer Vision", "LLM", "GenAI", "Prompt Engineering", "RAG",
        "Hugging Face", "XGBoost", "LangChain", "OpenAI API", "Reinforcement Learning",
    ],
    "data": [
        "Pandas", "NumPy", "Spark", "Hadoop", "Airflow", "dbt", "Kafka", "ETL",
        "Data Warehousing", "Snowflake", "BigQuery", "Redshift", "Tableau", "Power BI",
    ],
    "databases": [
        "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch", "Supabase",
        "DynamoDB", "Cassandra", "SQLite", "pgvector",
    ],
    "web_backend": [
        "React", "Node.js", "Express", "Django", "Flask", "FastAPI", "REST API",
        "GraphQL", "Next.js", "Streamlit",
    ],
    "cloud_devops": [
        "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform", "CI/CD",
        "GitHub Actions", "Render", "Vercel", "Jenkins",
    ],
    "practices": [
        "Agile", "Scrum", "TDD", "Microservices", "System Design", "A/B Testing",
        "Data Structures", "Algorithms",
    ],
}

ALL_SKILLS: list[str] = [skill for group in SKILLS_TAXONOMY.values() for skill in group]
