"""Shared type definitions for benchmark."""

from typing import Any, TypedDict


class GenerateResult(TypedDict):
    """Result from model generation."""

    response: str
    raw_api_response: dict[str, Any]


class JudgeResult(TypedDict):
    """Result from judge evaluation."""

    score: int
    reasoning: str
    raw_api_response: dict[str, Any]


class CIMJudgeResult(TypedDict):
    """Result from CIM judge evaluation."""

    revealed_attributes: list[str]
    used_attributes: list[str]
    violation_rate: float
    completeness_rate: float
    raw_api_response: dict[str, Any]


class GenerationEntry(TypedDict, total=False):
    """Stored generation record shared across evaluation modes."""

    generation_index: int
    error: str | None
    memory_response: str | None
    memory_raw_api_response: dict[str, Any]
    judge: JudgeResult | None
