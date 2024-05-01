from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np # arrays multidimentional calculations
import pandas as pd # manipulate data, load data, manip data
import matplotlib.pyplot as plt # visualazation
from IPython.display import clear_output
from six.moves import urllib
from tensorflow import feature_column as fc # algorithme
from tensorflow_estimator import estimator 


dftrain= pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')#training
dfeval=pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')#testing

CATEGORICAL_COLUMNS=['sex', 'n_siblings_spouses', 'parch','class','deck','embark_town','alone']
NUMERICAL_COLUMNS=['age','fare']