"""Candidate generation strategies."""

from .base import CandidateReason, finalize_candidates
from .exact import ExactBlocker
from .tokens import RareTokenBlocker
from .tfidf import TfidfTopKBlocker
from .union import BLOCKING_VERSION, CandidateGenerator

__all__ = ["CandidateReason", "ExactBlocker", "RareTokenBlocker", "TfidfTopKBlocker", "CandidateGenerator", "finalize_candidates", "BLOCKING_VERSION"]

