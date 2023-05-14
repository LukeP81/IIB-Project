"""Test initialise_home"""

from unittest.mock import call, patch

import pytest

from initialising.initialise_home import Initialise
from other.custom_mocks import MockUploadedFile

file1 = MockUploadedFile("file1")


# pylint: disable = too-many-arguments, too-many-locals
@pytest.mark.parametrize(
    """save_value, select_value,
    expected_save, expected_select,
    expected_example, expected_details,
    expected_cache_clear""",
    [(None, None, 1, 1, 0, 0, 1),
     (None, "string", 2, 1, 1, 0, 1),
     (None, file1, 2, 1, 0, 1, 1),
     ("string", None, 1, 0, 1, 0, 0),
     ("string", "string", 1, 0, 1, 0, 0),
     ("string", file1, 1, 0, 1, 0, 0),
     (file1, None, 1, 0, 0, 1, 0),
     (file1, "string", 1, 0, 0, 1, 0),
     (file1, file1, 1, 0, 0, 1, 0),
     ]
)
def test_expected_calls(save_value, select_value,
                        expected_save, expected_select,
                        expected_example, expected_details, expected_cache_clear):
    """Test that the correct number of calls are made"""

    save_func = "initialising.initialise_home.get_saved_file"
    select_func = "initialising.initialise_home.FileSelector.display"
    example_func = "initialising.initialise_home.load_example"
    details_func = "initialising.initialise_home.UploadedDetails.display"
    cache_func = "streamlit.cache_data.clear"
    data_func = "initialising.initialise_home.extract_data"

    with (patch(save_func) as mock_save_func,
          patch(select_func) as mock_select_func,
          patch(example_func) as mock_example_func,
          patch(details_func) as mock_details_func,
          patch(cache_func) as mock_cache_func,
          patch(data_func) as mock_data_func
          ):
        mock_save_func.return_value = save_value
        mock_select_func.return_value = select_value
        data_return = {"file": "file"}
        mock_data_func.return_value = data_return

        Initialise.display()

        expected_save_calls = [call()] * expected_save
        expected_select_calls = [call()] * expected_select
        expected_example_calls = [call(filename="string")] * expected_example
        expected_details_calls = [call(file_data=data_return,
                                       filename="file1")] * expected_details
        expected_cache_clear_calls = [call()] * expected_cache_clear

        mock_save_func.assert_has_calls(expected_save_calls)
        mock_select_func.assert_has_calls(expected_select_calls)
        mock_example_func.assert_has_calls(expected_example_calls)
        mock_details_func.assert_has_calls(expected_details_calls)
        mock_cache_func.assert_has_calls(expected_cache_clear_calls)
