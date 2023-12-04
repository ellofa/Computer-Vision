"""Python file to instantite the model and the transform that goes with it."""
from model import Net, GoogleNetModel
from data import data_transforms
from DiffAugment import modified_DiffAugment

class ModelFactory:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = self.init_model()
        self.transform = self.init_transform()

    def init_model(self):
        if self.model_name == "basic_cnn":
            return Net()
        if self.model_name == "GoogleNet":
            return GoogleNetModel()
        else:
            raise NotImplementedError("Model not implemented")

    def init_transform(self):
        if self.model_name == "basic_cnn":
            return data_transforms
        if self.model_name =="GoogleNet" :
            return modified_DiffAugment

        else:
            raise NotImplementedError("Transform not implemented")

    def get_model(self):
        return self.model

    def get_transform(self):
        return self.transform

    def get_all(self):
        return self.model, self.transform

