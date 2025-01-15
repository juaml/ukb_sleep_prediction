import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from .logging import logger


def heuristic_C(data=None):
    """
    Calculate the heuristic C for linearSVR (Joachims 2002).
    """

    if data is None:
        logger.error("No data was provided.")

    C = 1 / np.mean(np.sqrt((data**2).sum(axis=1)))

    # Formular Kaustubh: C = 1/mean(sqrt(rowSums(data^2)))

    return C


class LinearSVCHeuristicC(LinearSVC):
    """Inherit LinearSVC but overwrite fit function to set heuristically
    calculated C value in CV consistent manner without data leakage.
    """

    # inherit constructor completely from SVC

    # Overwrite fit method to use heuristic C as HP
    def fit(self, X, y, sample_weight=None):
        # calculate heuristic C
        C = heuristic_C(X)
        logger.info(f"Using heuristic C = {C} for LinearSVC")

        # set C value
        self.C = C

        # call super fit method
        super().fit(X, y, sample_weight=sample_weight)
        return self  # convention in scikitlearn


class LogisticRegressionHeuristicC(LogisticRegression):
    """Inherit LogisticRegression but overwrite fit function to set
    heuristically calculated C value in CV consistent manner without data
    leakage.
    """

    # inherit constructor completely from SVC

    # Overwrite fit method to use heuristic C as HP
    def fit(self, X, y, sample_weight=None):
        # calculate heuristic C
        C = heuristic_C(X)
        logger.info(f"Using heuristic C = {C} for LogisticRegression")

        # set C value
        self.C = C

        # call super fit method
        super().fit(X, y, sample_weight=sample_weight)
        return self  # convention in scikitlearn
