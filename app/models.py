"""Request bodies and the response shapes shared with the frontend (keys are a contract: keep them exact)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app import settings

JobStatus = Literal["queued", "frames", "track", "name", "done", "failed"]
StepStatus = Literal["done", "running", "queued", "failed", "planned"]


class AskRequest(BaseModel):
    question: str = Field(max_length=settings.MAX_QUESTION)

    @field_validator("question")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be empty")
        return v.strip()


class Step(BaseModel):
    id: str
    label: str
    engine: str
    status: StepStatus


class SuggestedName(BaseModel):
    playId: str
    title: str
    confidence: Literal["high", "medium", "low"]
    why: str
    needsReview: Literal[True] = True


class JobView(BaseModel):
    jobId: str
    status: JobStatus
    finishedAt: str | None
    steps: list[Step]
    suggestedName: SuggestedName | None
    playDna: dict | None
    error: str | None


class Citation(BaseModel):
    kind: Literal["play", "clip", "stat"]
    label: str
    href: str


class Answer(BaseModel):
    body: list[str]
    citations: list[Citation]
    confidence: Literal["high", "medium", "low"] | None
    credits: int


class AskResponse(BaseModel):
    question: str
    answer: Answer
    noSource: bool
