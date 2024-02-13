"""
This script provides custom implementations of Linear Support Vector
Regression (LinearSVR) and Linear Support Vector Classification (LinearSVC)
models with heuristic calculation of the C parameter, while ensuring
cross-validation consistency and avoiding data leakage.

Ref: https://search.r-project.org/CRAN/refmans/LiblineaR/html/heuristicC.html

Written by: Kaustubh Patil,
            Vera Kumeyer
Contact: no-reply@zoom.us
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.svm import LinearSVC

class LinearSVRHeuristicC_zscore(LinearSVR):
    """Inherit LinearSVR but overwrite fit function to set heuristically
    calculated C value in CV consistent manner without data leakage.
    """

    # inherit constructor entirely from LinearSVR

    # Overwrite fit method to use heuristic C as HP
    def fit(self, X, y, sample_weight=None):

        # z-score the features
        self.scaler_ = StandardScaler().fit(X)
        
        # calculate heuristic C
        # for this we first need to zscore the X
        X = self.scaler_.transform(X)
        C = 1/np.mean(np.sqrt((X**2).sum(axis=1)))

        # set C value
        self.C = C

        # call super fit method
        super().fit(X, y, sample_weight=sample_weight)
        return self  # convention in scikitlearn

    def predict(self, X=None):
        X = self.scaler_.transform(X)
        return super().predict(X)

    def score(self, X, y):
        X = self.scaler_.transform(X)
        return super().score(X, y)

class LinearSVCHeuristicC_zscore(LinearSVC):
    """Inherit LinearSVR but overwrite fit function to set heuristically
    calculated C value in CV consistent manner without data leakage.
    """

    # inherit constructor entirely from LinearSVR

    # Overwrite fit method to use heuristic C as HP
    def fit(self, X, y, sample_weight=None):

        # z-score the features
        self.scaler_ = StandardScaler().fit(X)
        
        # calculate heuristic C
        # for this we first need to zscore the X
        X = self.scaler_.transform(X)
        C = 1/np.mean(np.sqrt((X**2).sum(axis=1)))

        # set C value
        self.C = C

        # call super fit method
        super().fit(X, y, sample_weight=sample_weight)
        return self  # convention in scikitlearn

    def predict(self, X=None):
        X = self.scaler_.transform(X)
        return super().predict(X)

    def score(self, X, y):
        X = self.scaler_.transform(X)
        return super().score(X, y)