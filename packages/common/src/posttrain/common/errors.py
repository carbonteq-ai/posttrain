"""Errors shared by reusable post-training operations."""

from __future__ import annotations


class PostTrainError(Exception):
    """Base class for errors with stable platform semantics."""


class ContractError(PostTrainError, ValueError):
    """Raised when a public request or identity violates its contract."""


class OperationCancelled(PostTrainError):
    """Raised when cooperative cancellation is observed."""
