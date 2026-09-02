"""Temporal-label tests for the predictive-increment analysis."""

import numpy as np
import pandas as pd

from analysis.predictive_increment import add_forward_event_target


def test_forward_target_uses_only_future_steps_within_each_run() -> None:
    frame = pd.DataFrame(
        {
            "run_id": ["a"] * 5 + ["b"] * 5,
            "event": [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        }
    )
    result = add_forward_event_target(frame, event_column="event", horizon=2)
    np.testing.assert_array_equal(result.loc[:2, "future_event"], [1.0, 1.0, 0.0])
    np.testing.assert_array_equal(result.loc[5:7, "future_event"], [1.0, 0.0, 0.0])
    assert result.loc[[3, 4, 8, 9], "future_event"].isna().all()
