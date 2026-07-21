import numpy as np
import pandas as pd

from fuxictr_ext.fairjob.bootstrap import cluster_bootstrap


def test_cluster_bootstrap_is_deterministic_and_keeps_cluster_blocks():
    frame = pd.DataFrame(
        {
            "impression_id": [10, 10, 20, 20, 30, 30],
            "value": [1.0, 1.0, 2.0, 2.0, 4.0, 4.0],
        }
    )

    def statistic(sample):
        cluster_means = sample.groupby("impression_id")["value"].mean()
        return {"mean": float(cluster_means.mean())}

    first = cluster_bootstrap(frame, "impression_id", statistic, repeats=20, seed=7)
    second = cluster_bootstrap(frame, "impression_id", statistic, repeats=20, seed=7)
    assert first == second
    assert first["clusters"] == 3
    assert np.isfinite(first["metrics"]["mean"]["ci_low"])
