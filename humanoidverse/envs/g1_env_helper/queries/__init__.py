"""G1 per-frame motion metrics: METRICS maps a column name to its Metric.

Frames are selected with SQL over these columns, see humanoidverse/query/motion_query.py.
"""

from .metrics import METRICS, Metric

__all__ = ["METRICS", "Metric"]
