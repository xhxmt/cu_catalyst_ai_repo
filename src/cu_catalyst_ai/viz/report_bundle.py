from __future__ import annotations

import pandas as pd

from cu_catalyst_ai.utils.io import ensure_parent


def write_report_bundle(
    model_name: str, metrics_path: str, explanation_path: str, output_path: str
) -> None:
    metrics = pd.read_csv(metrics_path)
    explanation = pd.read_csv(explanation_path)
    top_features = ", ".join(explanation.head(5)["feature"].tolist())
    text = f"""# Run summary

Model: {model_name}

## Metrics

{metrics.to_markdown(index=False)}

## Top features

{top_features}
"""
    ensure_parent(output_path).write_text(text, encoding="utf-8")
