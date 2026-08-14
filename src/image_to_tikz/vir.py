from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Point:
    x: float
    y: float


@dataclass
class BoundingBox:
    x: float
    y: float
    width: float
    height: float

    @property
    def center(self) -> Point:
        return Point(self.x + self.width / 2, self.y + self.height / 2)


@dataclass
class TextBlock:
    id: str
    text: str
    bbox: BoundingBox
    confidence: float | None = None
    language: str | None = None


@dataclass
class VisualElement:
    id: str
    kind: str
    bbox: BoundingBox
    center: Point
    confidence: float
    geometry: dict[str, Any] = field(default_factory=dict)
    style: dict[str, Any] = field(default_factory=dict)
    labels: list[str] = field(default_factory=list)
    text_refs: list[str] = field(default_factory=list)


@dataclass
class Relation:
    source: str
    relation: str
    target: str
    confidence: float
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class VisualScene:
    schema: str
    version: str
    image: dict[str, Any]
    coordinate_system: dict[str, Any]
    elements: list[VisualElement] = field(default_factory=list)
    texts: list[TextBlock] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    semantic_summary: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
