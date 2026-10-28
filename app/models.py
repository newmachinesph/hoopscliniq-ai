"""Request bodies and the response shapes shared with the frontend (keys are a contract: keep them exact)."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator

from app import settings

JobStatus = Literal["queued", "frames", "track", "name", "done", "failed"]
StepStatus = Literal["done", "running", "queued", "failed", "planned"]

COURT_X = (-20.0, 520.0)
COURT_Y = (-20.0, 440.0)
MAX_FRAMES = 12
MAX_ACTIONS = 8
MAX_NAME = 80
MAX_NOTE = 200


class AskRequest(BaseModel):
    question: str = Field(max_length=settings.MAX_QUESTION)

    @field_validator("question")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be empty")
        return v.strip()


class HideRequest(BaseModel):
    playerIds: list[str] = Field(max_length=settings.MAX_HIDDEN_PLAYERS)


def _in_court(p: tuple[float, float]) -> tuple[float, float]:
    # Tenths of a foot, half court, with a little room outside the lines. NaN fails both comparisons.
    if not (COURT_X[0] <= p[0] <= COURT_X[1] and COURT_Y[0] <= p[1] <= COURT_Y[1]):
        raise ValueError("point is outside the court")
    return p


Point = Annotated[tuple[float, float], AfterValidator(_in_court)]


class PlayAction(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    kind: str = Field(max_length=20)
    from_: Point = Field(alias="from")
    to: Point
    via: Point | None = None


class PlayFrame(BaseModel):
    offense: list[Point] = Field(min_length=5, max_length=5)
    defense: list[Point] | None = Field(default=None, max_length=5)
    ball: int = Field(ge=0, le=4)
    actions: list[PlayAction] = Field(default_factory=list, max_length=MAX_ACTIONS)
    note: str | None = Field(default=None, max_length=MAX_NOTE)


class QuizPlay(BaseModel):
    id: str = Field(min_length=1, max_length=MAX_NAME)
    title: str = Field(min_length=1, max_length=MAX_NAME)
    frames: list[PlayFrame] = Field(min_length=1, max_length=MAX_FRAMES)

    @field_validator("id", "title")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be empty")
        return v.strip()


class QuizRequest(BaseModel):
    play: QuizPlay
    spot: int = Field(default=2, ge=1, le=5)
    count: int = Field(default=3, ge=2, le=5)


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


class HiddenView(BaseModel):
    playerIds: list[str]
    framesMissed: dict[str, int]
    frames: int
    at: str


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
    hidden: HiddenView | None = None


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
