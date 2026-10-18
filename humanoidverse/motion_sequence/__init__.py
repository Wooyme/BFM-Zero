"""Stable-ended motion segments, indexed in their original source frame coordinates."""

from .database import Sequence, SequenceDatabase, extract_spans

__all__ = ["Sequence", "SequenceDatabase", "extract_spans"]
