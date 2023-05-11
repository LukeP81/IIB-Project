"""Test initialise_home"""

from unittest.mock import patch

import pytest

from initialising.initialise_home import Initialise


@pytest.mark.parametrize(
    "value",
    (None, "file")
)
def test_returned_file(value):
    """Test that the return value of the file elicits the correct action"""

    calling_func = "initialising.initialise_home.details_display"
    value_func = "initialising.initialise_home.FileSelector.display"

    with (patch(calling_func) as mock_calling,
          patch(value_func) as mock_value, ):

        mock_value.return_value = value
        Initialise.display()

        if value is None:
            mock_calling.assert_not_called()
        else:
            mock_calling.assert_called_once()
