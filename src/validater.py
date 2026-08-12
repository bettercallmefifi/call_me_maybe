"""Models for the input and output files, functionally refactored."""

from __future__ import annotations

from typing import Any, Literal, Mapping, Sequence
from pydantic import BaseModel, ConfigDict

ValidJsonTypes = Literal["string", "number", "integer", "boolean"]


class ParameterSpec(BaseModel):
    type: ValidJsonTypes


class FunctionDefinition(BaseModel):
    """Blueprint for available LLM actions."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: Mapping[str, ParameterSpec]
    returns: ParameterSpec

    def ordered_parameters(self) -> Sequence[tuple[str, ValidJsonTypes]]:
        """Transforms parameters mapping into a strict sequence."""
        def _extract(item: tuple[str, ParameterSpec]) -> tuple[str, ValidJsonTypes]:
            return item[0], item[1].type

        return tuple(map(_extract, self.parameters.items()))


class FunctionCall(BaseModel):
    """The decoded output payload."""

    prompt: str
    name: str
    parameters: Mapping[str, Any]
