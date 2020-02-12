# Lab 02: Feature engineering with Amazon SageMaker

Create a Scikit-learn script to train with
https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/scikit_learn_inference_pipeline/Inference%20Pipeline%20with%20Scikit-learn%20and%20Linear%20Learner.ipynb

Preprocess input data before making predictions using Amazon SageMaker inference pipelines and Scikit-learn
https://aws.amazon.com/blogs/machine-learning/preprocess-input-data-before-making-predictions-using-amazon-sagemaker-inference-pipelines-and-scikit-learn/


Let's first create our Sagemaker session and role, and create a S3 prefix to use for the notebook example.
```
import sagemaker
from sagemaker import get_execution_role

sagemaker_session = sagemaker.Session()

# S3 prefix
bucket = '< ENTER BUCKET NAME HERE >'
prefix = 'Scikit-LinearLearner-pipeline-abalone-example'

# Get a SageMaker-compatible role used by this Notebook Instance.
# you created this role in the previous step
role = get_execution_role()
```

Download the dataset from our publically available S3 bucket and save it to `abalone_data` folder on your SageMaker instance
```
!wget --directory-prefix=./abalone_data https://s3-us-west-2.amazonaws.com/sparkml-mleap/data/abalone/abalone.csv
```

Now, upload the data to your own S3 bucket inside a folder respresented by the `prefix` variable
```
WORK_DIRECTORY = 'abalone_data'

train_input = sagemaker_session.upload_data(
    path='{}/{}'.format(WORK_DIRECTORY, 'abalone.csv'), 
    bucket=bucket,
    key_prefix='{}/{}'.format(prefix, 'train'))
```

Import standard python libraries including `pandas` and `np` to your notebook because you'll need them for feature engineering
```
from __future__ import print_function

import time
import sys
from io import StringIO
import os
import shutil

import argparse
import csv
import json
import numpy as np
import pandas as pd
```

Import Scikit learn libraries for pre-processing
```
from sklearn.compose import ColumnTransformer
from sklearn.externals import joblib
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer, StandardScaler, OneHotEncoder
```

and finally, SageMaker provided utilities like `encoders`, and `worker`
```
from sagemaker_containers.beta.framework import (content_types, encoders, env, modules, transformer, worker)
```

Specify columns for the data because we have imported a headerless CSV file and call it `feature_columns_names` then specify data types for those columns
```
feature_columns_names = [
    'sex', # M, F, and I (infant)
    'length', # Longest shell measurement
    'diameter', # perpendicular to length
    'height', # with meat in shell
    'whole_weight', # whole abalone
    'shucked_weight', # weight of meat
    'viscera_weight', # gut weight (after bleeding)
    'shell_weight'] # after being dried

label_column = 'rings'

feature_columns_dtype = {
    'sex': str,
    'length': np.float64,
    'diameter': np.float64,
    'height': np.float64,
    'whole_weight': np.float64,
    'shucked_weight': np.float64,
    'viscera_weight': np.float64,
    'shell_weight': np.float64}

label_column_dtype = {'rings': np.float64} # +1.5 gives the age in years
```



[< Prev: Lab 01](./01-lab.md) | [Home](./readme.md) | [Next: Lab 03 >](./03-lab.md)