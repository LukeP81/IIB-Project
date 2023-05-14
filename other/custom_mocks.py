"""Module containing mocks for testing"""


class MockUploadedFile:  # pylint: disable = too-few-public-methods
    """Class for mocking UploadedFile object"""

    def __init__(self, name):
        self.name = name
