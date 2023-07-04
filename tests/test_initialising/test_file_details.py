"""Test file details"""

import pytest

from initialising.file_details import load_example, FileFormatError, \
    UploadedDetails


@pytest.mark.parametrize(
    "file_data",
    ({"key1": [1, 2, 3]
      },
     {"key1": [1, 2, 3],
      "key2": [1, 2, 3],
      "key3": [1, 2, 3]
      },
     {"key1": [[1, 2], [3, 4]],
      "key2": [[1, 2], [3, 4]]
      }
     )
)
def test_file_data_errors(file_data):
    """Checks that file errors are flagged"""
    with pytest.raises(FileFormatError):
        UploadedDetails.display(file_data=file_data, filename="")


@pytest.mark.parametrize(
    "filename, file_data, expected_output",
    [
        ("concrete",
         {"x": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
          "y": [10, 20, 30]
          },
         {"x": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
          "y": [10, 20, 30],
          "f": ['Cement', 'Slag', 'Fly Ash', 'Water',
                'Plasticizer', 'Coarse', 'Fine', 'Age'],
          "v": 'Strength'
          }
         ),
        ("cw1e",
         {"x": [[1, 2], [3, 4], [5, 6]],
          "y": [7, 8, 9]
          },
         {"x": [[1, 2], [3, 4], [5, 6]],
          "y": [7, 8, 9],
          "f": ['Dimension 1', 'Dimension 2'],
          "v": 'Y value'
          }
         )
    ]
)
def test_load_example_concrete(filename, file_data, expected_output):
    """Checks example files work as expected"""
    assert load_example(file_data, filename) == expected_output
