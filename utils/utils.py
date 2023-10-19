import pickle


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Util:
    models = {}

    def load_model_file(self, file_path, default=None):
        try:
            if file_path not in self.models.keys():
                with open(file_path, "rb") as pickle_file:
                    model = pickle.load(pickle_file)
                    self.models.update({file_path: model})
                    return model
            else:
                return self.models[file_path]
        except FileNotFoundError as e:
            if default:
                return default
            else:
                raise e

    def save_model_file(self, obj, file_path):
        with open(file_path, "wb") as pickle_file:
            pickle.dump(obj, pickle_file)
