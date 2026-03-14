from __future__ import annotations

from pydantic import BaseModel


class ExperimentFeedback(BaseModel):
    catalyst_id: str
    measured_metric: float
    metric_name: str
    unit: str
    notes: str = ""
