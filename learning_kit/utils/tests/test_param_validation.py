import os

import pytest
from pydantic import Field

from learning_kit.utils.param_validation import validate_params


@validate_params({"a": [Field(gt=0, lt=10)]})
def mock_function(a: int):
    pass


def test_validate_params():
    mock_function(a=1)
    mock_function(a=0)


if __name__ == "__main__":
    pytest.main(["-s", "-v", f"{os.path.abspath(__file__)}"])
