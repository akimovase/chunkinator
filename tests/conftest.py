import pandas as pd
import pytest

from helpers import make_frame


@pytest.fixture
def example_df() -> pd.DataFrame:
    """The series from the task description: 1, 1, 2, 2, 2, 3."""
    return make_frame([1, 1, 2, 2, 2, 3])
