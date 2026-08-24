"""Offline evaluation harness comparing the rule-based-only pipeline against the hybrid
(rule-based + LLM) pipeline on a small hand-labeled set (labeled_set.csv).

This is the piece most portfolio "resume parser" projects skip entirely: an actual measurement
of whether the pipeline works, not just a demo that looks like it works. Two things are measured:

1. Skill-extraction quality: precision/recall/F1 of extracted skills against gold labels.
2. Ranking quality: Spearman correlation between the pipeline's predicted relevance score and a
   human relevance judgment (0-3), computed per job description and averaged.

Rule-based mode requires no API key and runs instantly — useful in CI. Hybrid mode calls OpenAI
for extraction + embeddings and costs a small amount of API credit.
"""
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from scipy.stats import spearmanr

from app.embeddings.embedder import cosine_similarity, embed_batch
from app.parsing.llm_extract import run_llm_extraction
from app.parsing.rule_based import extract_skills

LABELED_SET_PATH = Path(__file__).parent / "labeled_set.csv"

REQUIRED_SKILLS_BY_JD = {
    "jd_backend": ["Python", "PostgreSQL", "REST API", "Docker", "AWS"],
    "jd_data_scientist": ["Python", "Pandas", "Machine Learning", "SQL", "scikit-learn"],
    "jd_genai_engineer": ["Python", "LLM", "RAG", "LangChain", "OpenAI API", "Prompt Engineering"],
}


@dataclass
class EvalRow:
    jd_id: str
    jd_text: str
    resume_text: str
    gold_skills: set[str]
    human_relevance: int


@dataclass
class EvalReport:
    method: str
    extraction_precision: float
    extraction_recall: float
    extraction_f1: float
    mean_spearman_correlation: float
    per_jd_correlation: dict[str, float] = field(default_factory=dict)


def load_labeled_set() -> list[EvalRow]:
    rows = []
    with open(LABELED_SET_PATH, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(
                EvalRow(
                    jd_id=r["jd_id"],
                    jd_text=r["jd_text"],
                    resume_text=r["resume_text"],
                    gold_skills={s.strip().lower() for s in r["gold_skills"].split(";")},
                    human_relevance=int(r["human_relevance"]),
                )
            )
    return rows


def _predicted_skills_rule_based(text: str) -> set[str]:
    return {s.name.lower() for s in extract_skills(text)}


def _predicted_skills_hybrid(text: str) -> set[str]:
    rule_skills = {s.name.lower() for s in extract_skills(text)}
    llm_result = run_llm_extraction(text)
    llm_skills = {s.name.lower() for s in llm_result["additional_skills"]}
    return rule_skills | llm_skills


def _extraction_metrics(rows: list[EvalRow], predictor) -> tuple[float, float, float]:
    tp = fp = fn = 0
    for row in rows:
        predicted = predictor(row.resume_text)
        tp += len(predicted & row.gold_skills)
        fp += len(predicted - row.gold_skills)
        fn += len(row.gold_skills - predicted)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def _ranking_score_rule_based(row: EvalRow) -> float:
    predicted = _predicted_skills_rule_based(row.resume_text)
    required = {s.lower() for s in REQUIRED_SKILLS_BY_JD[row.jd_id]}
    return len(predicted & required)


def _ranking_scores_embedding(rows: list[EvalRow]) -> list[float]:
    """Batch-embeds every resume + JD text once, then scores via cosine similarity —
    far cheaper than one embedding call per row."""
    resume_vectors = embed_batch([r.resume_text for r in rows])
    jd_text_by_id = {r.jd_id: r.jd_text for r in rows}
    jd_ids = list(jd_text_by_id.keys())
    jd_vectors = dict(zip(jd_ids, embed_batch([jd_text_by_id[j] for j in jd_ids])))
    return [cosine_similarity(vec, jd_vectors[row.jd_id]) for vec, row in zip(resume_vectors, rows)]


def _mean_per_jd_spearman(rows: list[EvalRow], scores: list[float]) -> tuple[float, dict[str, float]]:
    by_jd: dict[str, list[tuple[float, int]]] = {}
    for row, score in zip(rows, scores):
        by_jd.setdefault(row.jd_id, []).append((score, row.human_relevance))

    per_jd_corr = {}
    for jd_id, pairs in by_jd.items():
        predicted = [p[0] for p in pairs]
        human = [p[1] for p in pairs]
        if len(set(predicted)) < 2 or len(set(human)) < 2:
            continue  # spearman is undefined with no variance
        corr, _ = spearmanr(predicted, human)
        per_jd_corr[jd_id] = float(corr)

    mean_corr = sum(per_jd_corr.values()) / len(per_jd_corr) if per_jd_corr else 0.0
    return mean_corr, per_jd_corr


def evaluate_rule_based(rows: list[EvalRow]) -> EvalReport:
    precision, recall, f1 = _extraction_metrics(rows, _predicted_skills_rule_based)
    scores = [_ranking_score_rule_based(r) for r in rows]
    mean_corr, per_jd = _mean_per_jd_spearman(rows, scores)
    return EvalReport("rule_based", precision, recall, f1, mean_corr, per_jd)


def evaluate_hybrid(rows: list[EvalRow]) -> EvalReport:
    """Requires OPENAI_API_KEY. Skips gracefully — callers should catch RuntimeError."""
    precision, recall, f1 = _extraction_metrics(rows, _predicted_skills_hybrid)
    scores = _ranking_scores_embedding(rows)
    mean_corr, per_jd = _mean_per_jd_spearman(rows, scores)
    return EvalReport("hybrid (rule-based + LLM)", precision, recall, f1, mean_corr, per_jd)


def run_full_evaluation(include_hybrid: bool = True) -> list[EvalReport]:
    rows = load_labeled_set()
    reports = [evaluate_rule_based(rows)]
    if include_hybrid:
        try:
            reports.append(evaluate_hybrid(rows))
        except RuntimeError as e:
            print(f"Skipping hybrid evaluation: {e}")
    return reports


def reports_to_markdown(reports: list[EvalReport]) -> str:
    lines = ["# Evaluation Report\n", "| Method | Precision | Recall | F1 | Mean Spearman r |",
             "|---|---|---|---|---|"]
    for r in reports:
        lines.append(
            f"| {r.method} | {r.extraction_precision:.2f} | {r.extraction_recall:.2f} | "
            f"{r.extraction_f1:.2f} | {r.mean_spearman_correlation:.2f} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    reports = run_full_evaluation()
    print(reports_to_markdown(reports))
    out_path = Path(__file__).parent / "evaluation_report.json"
    out_path.write_text(json.dumps([r.__dict__ for r in reports], indent=2))
    print(f"\nSaved {out_path}")
