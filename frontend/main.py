from frontend.file_utilities import FileUtilities
from frontend.check_optimised import ModelSomething
from frontend.pre_optimisation import FileDetails
from frontend.post_optimisation.main_post import MainPost


class MainPage:

    @classmethod
    def running(cls):
        file = FileUtilities.file_selector()
        if file is None:
            FileUtilities.not_selected()
            return
        opt = ModelSomething.check_optimised()
        if not opt:
            FileDetails.details(file)
            return
        else:
            model = ModelSomething.load_model()
            MainPost.main_post(model)
