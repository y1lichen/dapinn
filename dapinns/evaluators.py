class BaseEvaluator:
    def __init__(self, model):
        self.model = model

    def evaluate(self, data_loader):
        raise NotImplementedError("Subclasses should implement this method.")