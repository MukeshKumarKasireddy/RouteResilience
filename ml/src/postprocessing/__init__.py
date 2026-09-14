"""Mask post-processing utilities (Part 1: cleanup foundation only).

Full occlusion/gap recovery is scoped for Part 2 and owned by Suryakant.
"""

from .mask_cleanup import clean_mask

__all__ = ["clean_mask"]
