from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np # arrays multidimentional calculations
import pandas as pd # manipulate data, load data, manip data
import matplotlib.pyplot as plt # visualazation
from IPython.display import clear_output
from six.moves import urllib
from tensorflow import feature_column as fc # algorithme
from tensorflow_estimator import estimator 
# load datasets

dftrain= pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')#training
dfeval=pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')#testing

y_train=dftrain.pop('survived')
y_eval=dfeval.pop('survived')
print(dftrain["age"].loc[23])
print(dftrain.describe())
print(dftrain.shape)

# Catégorical columns to transform to numerical cols
# you needs the features
CATEGORICAL_COLUMNS=['sex', 'n_siblings_spouses', 'parch','class','deck','embark_town','alone']
NUMERICAL_COLUMNS=['age','fare']
feature_columns=[]
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary=dftrain[feature_name].unique() # gets unique values from given colums
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name,vocabulary))

for feature_name in NUMERICAL_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name,dtype=tf.float32))

print(feature_columns)

# training process
# we need the input for the model: the model needs to see tf.dataset

def make_input_fn(data_df, label_df, num_epochs=10, batch_size=32, shuffle=True):
    """ 
    data_df is the pandas datafram
    label_df is the data we  want to predict the means the survived label
    num_epochs is the number of times the model will see the same dataset
    batc_size is set of data we feed the model 
       """
    def input_function():# the input we want to return
        ds=tf.data.Dataset.from_tensor_slices((dict(data_df),label_df)) # create dataset
        if shuffle:
            ds=ds.shuffle(1000)
        ds=ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_function

train_input_fn=make_input_fn(dftrain, y_train)
eval_input_fn=make_input_fn(dfeval,y_eval, num_epochs=1, shuffle=False)

# estimation
linear_est=tf.estimator.LinearClassifier(feature_columns=feature_columns)
linear_est.train(train_input_fn)# train the model
result=linear_est.evaluate(eval_input_fn) # get the stats

clear_output()
#print(result['accuracy'])

get_probality_of_first=list(linear_est.predict(eval_input_fn))

#the first probability of survival
print(dfeval.loc[2])
print(get_probality_of_first[2]['probabilities'][1])

