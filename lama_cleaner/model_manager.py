from lama_cleaner.model.lama import LaMa
from lama_cleaner.schema import Config

models = {
    'lama': LaMa,
}


class ModelManager:
    def __init__(self, name: str, device):
        self.name = name
        self.device = device
        self.model = self.init_model(name, device)

    def init_model(self, name: str, device):
        if name in models:
            model = models[name](device)
        else:
            raise NotImplementedError(f"Not supported model: {name}")
        return model

    def is_downloaded(self, name: str) -> bool:
        if name in models:
            return models[name].is_downloaded()
        else:
            raise NotImplementedError(f"Not supported model: {name}")

    def __call__(self, image, mask, config: Config):
        return self.model(image, mask, config)
