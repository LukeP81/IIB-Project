"""Test file_select"""

import pytest
import streamlit_mock

from initialising.file_selection import FileSelector
from other.custom_mocks import MockUploadedFile

file1 = MockUploadedFile("file1")
file2 = MockUploadedFile("file2")
multiple_files = [file1, file2]


def test_no_return():
    """Check that nothing is returned on no input"""

    assert FileSelector.display() is None


@pytest.mark.parametrize(
    "files",
    (file1,
     multiple_files),
)
def test_no_return_when_uploaded(files):
    """Check that nothing is returned immediately after a file upload"""

    st_mock = streamlit_mock.StreamlitMock()
    session_state = st_mock.get_session_state()
    session_state["initial_file_upload"] = files
    assert FileSelector.display() is None


@pytest.mark.parametrize(
    "key,selected,expected",
    [("upload", None, file1),
     ("upload", file2.name, file2),
     ("example", None, "cw1a"),
     ("example", "pima", "pima")]
)
def test_returned_upload(key, selected, expected):
    """Test that the expected file is returned"""

    st_mock = streamlit_mock.StreamlitMock()
    session_state = st_mock.get_session_state()
    session_state["initial_file_upload"] = multiple_files

    if selected is not None:
        session_state[f"initial_file_select_{key}"] = selected

    session_state[f"using_selected_file_{key}"] = True

    assert FileSelector.display() == expected
