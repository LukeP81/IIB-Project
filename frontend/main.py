from frontend.file_utilities import FileUtilities
from frontend.model_utilities import ModelUtilities
from frontend.pre_optimisation import FileDetails
from frontend.post_optimisation.main_post import MainPost


class MainPage:

    @classmethod
    def running(cls):
        file = FileUtilities.file_selector()
        if file is None:
            FileUtilities.not_selected()
            return
        opt = ModelUtilities.check_optimised()
        if not opt:
            FileDetails.details(file)
            return
        else:
            model = ModelUtilities.load_model()
            MainPost.main_post(model)
