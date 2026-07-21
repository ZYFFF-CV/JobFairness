import numpy as np
import pandas as pd
import pytest

from fuxictr_ext.fairjob.selection_protocols import position_weights, select_rows


def make_frame():
    return pd.DataFrame(
        {
            "displayrandom": [0, 1, 1, 0],
            "senior": [0, 0, 1, 1],
            "rank": [1, 2, 1, 2],
        }
    )


def test_selection_scopes_are_explicit():
    frame = make_frame()
    assert len(select_rows(frame, "all_logged")) == 4
    assert len(select_rows(frame, "random_display")) == 2
    assert len(select_rows(frame, "context_conditioned", {"senior": 1})) == 2


def test_position_correction_requires_external_propensity():
    with pytest.raises(ValueError, match="logging_propensity"):
        position_weights(make_frame())

    frame = make_frame().assign(logging_propensity=[0.5, 0.25, 0.5, 0.25])
    weights = position_weights(frame)
    assert np.isclose(weights.mean(), 1.0)
    assert weights[1] > weights[0]
