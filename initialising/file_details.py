"""Module for setting data and feature names of files"""
from typing import List, Optional

import mat4py
import pandas as pd
import streamlit as st


class FileFormatError(Exception):
    """Error for an incorrect format of file"""


class UploadedDetails:  # pylint: disable = too-few-public-methods
    """Class containing methods for getting uploaded
    file data and feature names"""

    @classmethod
    def _get_file_keys(cls, file_data: dict) -> List[str]:
        """Gets the file keys of the data and checks that the
        file format is usable
        :param file_data:
        :return: the keys of the data in the correct order
        """

        file_keys = list(file_data.keys())

        # check that file data is correct format
        if len(file_keys) != 2:
            raise FileFormatError
        if len(file_data[file_keys[1]][0]) != 1:
            if len(file_data[file_keys[0]][0]) != 1:
                raise FileFormatError
            file_keys.reverse()

        return file_keys

    @classmethod
    def _feature_naming(cls, num_features: int) -> List[str]:
        """Creates text inputs for feature naming
        :param num_features: the number of features to name
        :return: the feature names
        """

        st.subheader("Feature Names")

        # generate flexible number of input boxes
        feature_names = [""] * num_features
        for i in range(num_features):
            feature_names[i] = st.text_input(
                label=f"Feature {i + 1}",
                key=f"file_feature_name_{i + 1}",
                help="Enter the name of the feature here",
                placeholder=f"Feature {i + 1}",
                label_visibility="collapsed"
            )

        # auto-name empty inputs
        for i, feature in enumerate(feature_names):
            if not feature:
                feature_names[i] = f"Feature {i + 1}"

        return feature_names

    @classmethod
    def _value_naming(cls) -> str:
        """Creates text input for value naming
        :return: the value name
        """

        st.subheader("Value Name")
        value_name = st.text_input(
            label="Output Value",
            key="file_value_name",
            help="Enter the name of the output value here",
            placeholder="Output Value",
            label_visibility="collapsed"
        )

        return value_name

    @classmethod
    def display(cls, file_data: dict, filename: str) -> Optional[dict]:
        """Method for getting file data and displaying the UI for feature naming
        :param file_data: the data from the file
        :param filename: the name of the file
        :return: None
        """
        st.title("File Details")
        file_keys = cls._get_file_keys(file_data=file_data)
        num_features = len(file_data[file_keys[0]][0])

        df_col, feature_col, value_col = st.columns([3, 1, 1])

        with df_col:
            st.subheader(f"Filename: {filename}")
            file_dataframe = pd.DataFrame(data=file_data)
            file_dataframe.index += 1
            st.dataframe(data=file_dataframe)

        with feature_col:
            feature_names = cls._feature_naming(num_features=num_features)

        with value_col:
            value_name = cls._value_naming()

        _, back_col, next_col, _ = st.columns([4, 1, 1, 4])

        with back_col:
            st.button(label="back",
                      key="file_details_back_button",
                      help="Press this to return to file selection",
                      on_click=st.cache_data.clear)

        with next_col:
            if st.button(label="next",
                         key="file_details_next_button",
                         help="Press this to advance to optimisation"):
                return {"x": file_data[file_keys[0]],
                        "y": file_data[file_keys[1]],
                        "f": feature_names,
                        "v": value_name,
                        }
        return None


def load_example(file_data: dict, filename: str):
    """Loads the data and feature names from an example file into cache"""

    # fine to be hardcoded as definite examples
    feature_names = {
        "concrete": [['Cement', 'Slag', 'Fly Ash', 'Water',
                      'Plasticizer', 'Coarse', 'Fine', 'Age'],
                     'Strength'],
        "cw1e": [['Dimension 1', 'Dimension 2'],
                 'Y value']
    }

    return {"x": list(file_data.values())[0],
            "y": list(file_data.values())[1],
            "f": feature_names[filename][0],
            "v": feature_names[filename][1],
            }
