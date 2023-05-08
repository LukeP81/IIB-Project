"""Module for the main page of the UI"""

from frontend.file_utilities import FileUtilities
from frontend.model_utilities import ModelUtilities
from frontend.pre_optimisation import FileDetails
from frontend.post_optimisation.main_post import MainPost


class MainPage:
    """Namespace for the main methods of the UI"""

    @classmethod
    def running(cls):
        """Module for running main page of the UI"""

        file = FileUtilities.file_selector()
        if file is None:
            FileUtilities.no_file_selected()
            return
        opt = ModelUtilities.check_optimised()
        if not opt:
            FileDetails.details(file)
            return
        model = ModelUtilities.load_model()
        MainPost.main_post(model)
# todo neaten
