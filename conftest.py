from unittest.mock import patch

import pytest

from utils.data_loader import load_all


@pytest.fixture(scope="session")
def sample_data():
    """All eight Olist CSVs from data/ (same keys as utils.data_loader.load_all)."""
    return load_all()


@pytest.fixture
def mock_query_llm():
    with patch("utils.openrouter_client.query_llm") as mock:
        mock.return_value = "{}"
        yield mock
