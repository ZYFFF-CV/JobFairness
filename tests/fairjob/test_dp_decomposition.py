import json
import math

import pandas as pd

from fuxictr_ext.fairjob.dp_decomposition import decompose_dp


def test_dp_decomposition_separates_within_and_composition_terms():
    frame = pd.DataFrame(
        {
            "protected_attribute": [0, 0, 0, 1, 1, 1],
            "context": ["a", "a", "b", "a", "b", "b"],
            "y_pred": [0.1, 0.1, 0.5, 0.2, 0.6, 0.6],
        }
    )
    result = decompose_dp(frame, ["context"], reference="pooled")
    assert math.isclose(result["within_context_dp_signed"], 0.1)
    assert math.isclose(
        result["aggregate_dp_signed"],
        result["within_context_dp_signed"] + result["composition_residual_signed"],
    )
    assert result["common_contexts"] == 2
    assert sum(item["reference_weight"] for item in result["contexts"]) == 1.0
    json.dumps(result)
