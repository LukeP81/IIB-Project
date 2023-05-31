"""Module for caching file information"""

import streamlit as st

from cacheable_api.api_exceptions import NotCachedError


class FileAPI:

    @staticmethod
    @st.cache_data
    def get_saved_file(_file=None):
        """Caches the chosen file"""
        if _file is None:
            raise NotCachedError
        return _file

    @staticmethod
    def set_saved_file(file):
        _ = FileAPI.get_saved_file(file)

    @staticmethod
    @st.cache_data
    def get_file_data(_file_x_data=None, _file_y_data=None):
        """Caches the data from the chosen file"""
        if _file_x_data is None or _file_y_data is None:
            raise NotCachedError
        return _file_x_data, _file_y_data

    @staticmethod
    def set_file_data(file_x_data, file_y_data):
        _ = FileAPI.get_file_data(file_x_data, file_y_data)

    @staticmethod
    @st.cache_data
    def get_feature_names(_file_feature_names=None, _file_value_name=None):
        """Caches the feature names"""
        if _file_feature_names is None or _file_value_name is None:
            raise NotCachedError
        return _file_feature_names, _file_value_name

    @staticmethod
    def set_feature_names(data_feature_names, data_value_name):
        _ = FileAPI.get_feature_names(data_feature_names, data_value_name)

    @staticmethod
    def get_covariate_names():
        """Caches the feature names"""
        return FileAPI.get_feature_names()[0]

    @staticmethod
    def get_value_name():
        """Caches the feature names"""
        return FileAPI.get_feature_names()[1]
