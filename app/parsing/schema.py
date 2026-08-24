"""Pydantic schemas shared across the parsing, matching, and evaluation pipeline."""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ExtractionSource(str, Enum):
    RULE_BASED = "rule_based"
    LLM = "llm"
    MERGED = "merged"


class ContactInfo(BaseModel):
    full_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin_url: Optional[str] = None
    github_url: Optional[str] = None


class Education(BaseModel):
    institution: str
    degree: Optional[str] = None
    field_of_study: Optional[str] = None
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    gpa: Optional[float] = None


class Experience(BaseModel):
    company: str
    title: str
    start_date: Optional[str] = None  # kept as free text ("Jan 2021") — normalized later
    end_date: Optional[str] = None
    is_current: bool = False
    bullets: list[str] = Field(default_factory=list)


class SkillMention(BaseModel):
    name: str
    source: ExtractionSource
    years_experience: Optional[float] = None


class ParsedResume(BaseModel):
    contact: ContactInfo
    summary: Optional[str] = None
    education: list[Education] = Field(default_factory=list)
    experience: list[Experience] = Field(default_factory=list)
    skills: list[SkillMention] = Field(default_factory=list)
    total_years_experience: Optional[float] = None
    raw_text: str
    extraction_method: ExtractionSource
    rule_based_latency_ms: Optional[float] = None
    llm_latency_ms: Optional[float] = None
    llm_tokens_used: Optional[int] = None


class JobDescription(BaseModel):
    title: str
    company: Optional[str] = None
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    min_years_experience: Optional[float] = None
    raw_text: str


class MatchResult(BaseModel):
    resume_id: int
    jd_id: int
    embedding_similarity: float
    llm_fit_score: Optional[float] = None
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    explanation: Optional[str] = None
    risk_flags: list[str] = Field(default_factory=list)
