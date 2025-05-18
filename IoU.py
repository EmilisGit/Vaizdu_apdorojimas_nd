import numpy as np

class IoU:
    def __init__(self, ground_truth, prediction):
        assert ground_truth.shape == prediction.shape, "Mask shapes have to match."
        self.ground_truth = ground_truth > 0
        self.prediction = prediction > 0

    def evaluate(self):
        intersection = np.logical_and(self.prediction, self.ground_truth).sum()
        union = np.logical_or(self.prediction, self.ground_truth).sum()

        if union == 0:
            return 1.0
        
        return intersection / union
