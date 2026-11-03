import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from skimage.feature import hog

def extract_hog(X):
    images = X[..., 0] if X.ndim == 4 else X
    return np.asarray([hog(im, orientations=9, pixels_per_cell=(4,4), cells_per_block=(2,2), block_norm="L2-Hys") for im in images])

def get_classical_models():
    return {
        "hog_svm": make_pipeline(StandardScaler(), SVC(C=3.0, kernel="rbf", class_weight="balanced")),
        "hog_logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "hog_random_forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced_subsample"),
    }
