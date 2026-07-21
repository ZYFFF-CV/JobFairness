import numpy as np

from fuxictr_ext.fairjob.interaction_ablation import (
    mask_columns,
    standardized_mean_difference,
)


def test_standardized_mean_difference_ranks_group_separating_unit_first():
    X = np.array(
        [
            [0.0, 1.0, 2.0],
            [0.2, 2.0, 1.0],
            [3.0, 1.5, 2.0],
            [3.2, 1.5, 1.0],
        ],
        dtype=np.float32,
    )
    y = np.array([0, 0, 1, 1], dtype=np.int8)
    scores = standardized_mean_difference(X, y)
    assert int(np.argmax(scores)) == 0
    assert np.isfinite(scores).all()


def test_mask_columns_uses_train_replacement_without_mutating_input():
    X = np.arange(12, dtype=np.float32).reshape(3, 4)
    replacement = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    masked = mask_columns(X, np.array([1, 3]), replacement)
    np.testing.assert_array_equal(masked[:, 1], np.full(3, 20.0))
    np.testing.assert_array_equal(masked[:, 3], np.full(3, 40.0))
    np.testing.assert_array_equal(X, np.arange(12, dtype=np.float32).reshape(3, 4))
