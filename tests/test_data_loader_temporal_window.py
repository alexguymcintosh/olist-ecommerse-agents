from pathlib import Path
import sys

import pandas as pd
from pandas.testing import assert_frame_equal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from utils.data_loader import temporal_window


def _build_df(months: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "month": months,
            "value": list(range(1, len(months) + 1)),
        }
    )


def test_temporal_window_inclusive_bounds() -> None:
    df = _build_df(["2016-09", "2016-10", "2016-11", "2016-12"])

    filtered_df, month_count = temporal_window(df, "2016-10", "2016-11")

    assert filtered_df["month"].tolist() == ["2016-10", "2016-11"]
    assert month_count == 2


def test_temporal_window_reports_missing_month_count() -> None:
    df = _build_df(["2016-10", "2016-12"])

    filtered_df, month_count = temporal_window(df, "2016-10", "2016-12")

    assert filtered_df["month"].tolist() == ["2016-10", "2016-12"]
    assert month_count == 2


def test_temporal_window_returns_dataframe_and_int() -> None:
    df = _build_df(["2016-09", "2016-10"])

    filtered_df, month_count = temporal_window(df, "2016-09", "2016-10")

    assert isinstance(filtered_df, pd.DataFrame)
    assert isinstance(month_count, int)


def test_temporal_window_does_not_mutate_input() -> None:
    df = _build_df(["2016-09", "2016-10", "2016-11"])
    original = df.copy(deep=True)

    _ = temporal_window(df, "2016-10", "2016-11")

    assert_frame_equal(df, original)
