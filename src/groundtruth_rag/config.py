"""Configuration contracts for later evaluation runs."""

from pydantic import BaseModel, Field


class RetrievalConfig(BaseModel):
    name: str = Field(min_length=1)
    k: int = Field(default=10, ge=1)
    parameters: dict[str, str | int | float | bool] = Field(default_factory=dict)
