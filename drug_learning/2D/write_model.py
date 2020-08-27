import os
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.svm import SVC

PATH_DATA = "../datasets/SARS2/"
PATH_DATA_GHDDI = "../datasets/SAR1/GHDDI"
threshold_activity = 20 # uM

data = pd.read_csv(os.path.join(PATH_DATA, "dataset_cleaned_SARS1_SARS2.csv"))
features = np.load(os.path.join("features", "SARS1_SARS2", "morgan.npy"))
active = (data["activity_merged"] < threshold_activity).values.astype(int)
C = 4.17531
gamm = 0.02807
svm = SVC(C=C, kernel="rbf", gamma=gamm, class_weight="balanced")
svm.fit(features, active)
dump(svm, os.path.join("models", "svm_SARS1_SARS2_20uM.joblib"))
