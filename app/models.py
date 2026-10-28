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


class OverlayPlayer(BaseModel):
    team: Literal["offense", "defense"]
    # One entry per clip.mp4 frame: [x1, y1, x2, y2] in clip.mp4 pixels, None where SAM missed the player.
    boxes: list[list[int] | None]


class Overlay(BaseModel):
    fps: int
    frames: int
    imageSize: list[int]
    players: dict[str, OverlayPlayer]
    # One entry per clip.mp4 frame: the ball box in clip.mp4 pixels, None where SAM did not see the ball.
    # Missing (None) on jobs from before 0.3.0.
    ball: list[list[int] | None] | None = None


class CalibrationView(BaseModel):
    source: Literal["estimated", "coach"]
    frame: int
    points: list[list[float]]
    landmarks: Literal["lane", "half"] = "half"


class CameraMotionView(BaseModel):
    framesTracked: int
    framesFallback: int


class JobView(BaseModel):
    jobId: str
    status: JobStatus
    finishedAt: str | None
    steps: list[Step]
    suggestedName: SuggestedName | None
    playDna: dict | None
    error: str | None
    clipUrl: str | None
    overlay: Overlay | None
    calibration: CalibrationView | None = None
    cameraMotion: CameraMotionView | None = None


class JobSummary(BaseModel):
    jobId: str
    finishedAt: str | None
    title: str | None
    playersFound: int


class JobList(BaseModel):
    jobs: list[JobSummary]


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
