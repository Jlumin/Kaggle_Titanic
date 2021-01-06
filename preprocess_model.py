import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection,metrics
from xgboost import XGBClassifier
from xgboost import XGBRegressor

#gender= pd.read_csv('gender_submission.csv')
# def read_and_concat_dataset(training_path, test_path):
#     train = pd.read_csv(training_path)
#     train['train'] = 1
#     test = pd.read_csv(test_path)
#     test['train'] = 0
#     data = train.append(test, ignore_index=True)
#     return train, test, data
# train_df, test_df, combine = read_and_concat_dataset('train.csv', 'test.csv')


# Load in the train and test datasets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')