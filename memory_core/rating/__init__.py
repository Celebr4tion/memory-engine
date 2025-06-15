"""
Rating system package for evaluating knowledge node quality.

This package provides functionality for assessing and updating
knowledge node ratings such as truthfulness, richness, and stability.
"""

from memory_core.rating.rating_system import update_rating, RatingUpdater

__all__ = ["update_rating", "RatingUpdater"]
