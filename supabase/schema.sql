-- Resume Intelligence Platform — Supabase/Postgres schema.
-- Run this once against a fresh Supabase project (SQL Editor, or `psql $DATABASE_URL -f schema.sql`).

create extension if not exists vector;

create table if not exists candidates (
    id serial primary key,
    full_name text,
    email text unique,
    phone text,
    linkedin_url text,
    github_url text,
    created_at timestamptz not null default now()
);

create table if not exists resumes (
    id serial primary key,
    candidate_id integer not null references candidates(id) on delete cascade,
    source_filename text,
    summary text,
    raw_text text not null,
    total_years_experience numeric,
    extraction_method text not null,
    created_at timestamptz not null default now()
);
create index if not exists idx_resumes_candidate_id on resumes(candidate_id);

create table if not exists education (
    id serial primary key,
    resume_id integer not null references resumes(id) on delete cascade,
    institution text not null,
    degree text,
    field_of_study text,
    start_year integer,
    end_year integer
);
create index if not exists idx_education_resume_id on education(resume_id);

create table if not exists experience (
    id serial primary key,
    resume_id integer not null references resumes(id) on delete cascade,
    company text not null,
    title text not null,
    start_date text,
    end_date text,
    is_current boolean not null default false,
    bullets text[] not null default '{}'
);
create index if not exists idx_experience_resume_id on experience(resume_id);

create table if not exists resume_skills (
    id serial primary key,
    resume_id integer not null references resumes(id) on delete cascade,
    skill_name text not null,
    source text not null check (source in ('rule_based', 'llm', 'merged')),
    unique (resume_id, skill_name)
);
create index if not exists idx_resume_skills_resume_id on resume_skills(resume_id);
create index if not exists idx_resume_skills_skill_name on resume_skills(skill_name);

create table if not exists resume_embeddings (
    resume_id integer primary key references resumes(id) on delete cascade,
    embedding vector(1536) not null
);
-- IVFFlat index for approximate nearest-neighbor cosine search. `lists` is tuned low here since
-- portfolio-scale data is small (hundreds, not millions, of rows) — bump it up as data grows.
create index if not exists idx_resume_embeddings_ann
    on resume_embeddings using ivfflat (embedding vector_cosine_ops) with (lists = 10);

create table if not exists job_descriptions (
    id serial primary key,
    title text not null,
    company text,
    raw_text text not null,
    min_years_experience numeric,
    created_at timestamptz not null default now()
);

create table if not exists jd_required_skills (
    id serial primary key,
    jd_id integer not null references job_descriptions(id) on delete cascade,
    skill_name text not null
);
create index if not exists idx_jd_required_skills_jd_id on jd_required_skills(jd_id);

create table if not exists jd_embeddings (
    jd_id integer primary key references job_descriptions(id) on delete cascade,
    embedding vector(1536) not null
);

create table if not exists match_results (
    id serial primary key,
    resume_id integer not null references resumes(id) on delete cascade,
    jd_id integer not null references job_descriptions(id) on delete cascade,
    embedding_similarity numeric not null,
    llm_fit_score numeric,
    matched_skills text[] not null default '{}',
    missing_skills text[] not null default '{}',
    explanation text,
    risk_flags text[] not null default '{}',
    created_at timestamptz not null default now(),
    unique (resume_id, jd_id)
);
create index if not exists idx_match_results_jd_id on match_results(jd_id);

create table if not exists extraction_runs (
    id serial primary key,
    resume_id integer not null references resumes(id) on delete cascade,
    extraction_method text not null,
    rule_based_latency_ms numeric,
    llm_latency_ms numeric,
    llm_tokens_used integer,
    created_at timestamptz not null default now()
);
