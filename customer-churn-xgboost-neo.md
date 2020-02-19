# Lab: Feature engineering and customer Churn Prediction with XGBoost

## Overview

### Problem statement

Losing customers is costly for any business. Identifying unhappy customers early
on gives you a chance to offer them incentives to stay. This notebook describes
using machine learning (ML) for the automated identification of unhappy
customers, also known as customer churn prediction. ML models rarely give
perfect predictions though, so this notebook is also about how to incorporate
the relative costs of prediction mistakes when determining the financial outcome
of using ML.

We use an example of churn that is familiar to all of us–leaving a mobile phone
operator. Seems like I can always find fault with my provider du jour! And if my
provider knows that I’m thinking of leaving, it can offer timely incentives–I
can always use a phone upgrade or perhaps have a new feature activated–and I
might just stick around. Incentives are often much more cost effective than
losing and reacquiring a customer.

### About XGBoost

XGBoost (extreme gradient boosting) is a popular and efficient open-source
implementation of the gradient-boosted trees algorithm. Gradient boosting is a
machine learning algorithm that attempts to accurately predict target variables
by combining the estimates of a set of simpler, weaker models. By applying
gradient boosting to decision tree models in a highly scalable manner, XGBoost
does remarkably well in machine learning competitions. It also robustly handles
a variety of data types, relationships, and distributions. It provides a large
number of hyperparameters—variables that can be tuned to improve model
performance. This flexibility makes XGBoost a solid choice for various machine
learning problems.

## Setup

In this section, we setup the following:

- The S3 bucket and prefix that you want to use for training and model data.
  This should be within the same region as the Notebook Instance, training, and
  hosting.
- The IAM role arn used to give training and hosting access to your data. See
  the documentation for how to create these. Note, if more than one role is
  required for notebook instances, training, and/or hosting, please replace the
  boto regexp with a the appropriate full IAM role arn string(s).

```python
bucket = '<your_s3_bucket_name_here>'
prefix = 'sagemaker/DEMO-xgboost-churn'

# Define IAM role
import boto3
import re
from sagemaker import get_execution_role

role = get_execution_role()
```

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import sys
import time
import json
from IPython.display import display
from time import strftime, gmtime
import sagemaker
from sagemaker.predictor import csv_serializer
```

## Data exploration feature engineering

Mobile operators have historical records on which customers ultimately ended up
churning and which continued using the service. We can use this historical
information to construct an ML model of one mobile operator’s churn using a
process called training. After training the model, we can pass the profile
information of an arbitrary customer (the same profile information that we used
to train the model) to the model, and have the model predict whether this
customer is going to churn. Of course, we expect the model to make
mistakes–after all, predicting the future is tricky business! But I’ll also show
how to deal with prediction errors.

```
!wget http://dataminingconsultant.com/DKD2e_data_sets.zip
!unzip -o DKD2e_data_sets.zip
```

By modern standards, it’s a relatively small dataset, with only 3,333 records,
where each record uses 21 attributes to describe the profile of a customer of an
unknown US mobile operator. The attributes are:

- State: the US state in which the customer resides, indicated by a two-letter
  abbreviation; for example, OH or NJ
- Account Length: the number of days that this account has been active
- Area Code: the three-digit area code of the corresponding customer’s phone
  number
- Phone: the remaining seven-digit phone number
- Int’l Plan: whether the customer has an international calling plan: yes/no
- VMail Plan: whether the customer has a voice mail feature: yes/no
- VMail Message: presumably the average number of voice mail messages per month
- Day Mins: the total number of calling minutes used during the day
- Day Calls: the total number of calls placed during the day
- Day Charge: the billed cost of daytime calls
- Eve Mins, Eve Calls, Eve Charge: the billed cost for calls placed during the
  evening
- Night Mins, Night Calls, Night Charge: the billed cost for calls placed during
  nighttime
- Intl Mins, Intl Calls, Intl Charge: the billed cost for international calls
- CustServ Calls: the number of calls placed to Customer Service
- Churn?: whether the customer left the service: true/false

The last attribute, Churn?, is known as the target attribute–the attribute that
we want the ML model to predict. Because the target attribute is binary, our
model will be performing binary prediction, also known as binary classification.

```python
churn = pd.read_csv('./Data sets/churn.txt')
pd.set_option('display.max_columns', 500)
churn
```

```python
# Frequency tables for each categorical feature
for column in churn.select_dtypes(include=['object']).columns:
    display(pd.crosstab(index=churn[column], columns='% observations', normalize='columns'))

# Histograms for each numeric features
display(churn.describe())
%matplotlib inline
hist = churn.hist(bins=30, sharey=True, figsize=(10, 10))
```

We can see immediately that:

- State appears to be quite evenly distributed
- Phone takes on too many unique values to be of any practical use. It's
  possible parsing out the prefix could have some value, but without more
  context on how these are allocated, we should avoid using it.
- Only 14% of customers churned, so there is some class imabalance, but nothing
  extreme.
- Most of the numeric features are surprisingly nicely distributed, with many
  showing bell-like gaussianity. VMail Message being a notable exception (and
  Area Code showing up as a feature we should convert to non-numeric).

```python
churn = churn.drop('Phone', axis=1)
churn['Area Code'] = churn['Area Code'].astype(object)
```

Next let's look at the relationship between each of the features and our target
variable.

```python
for column in churn.select_dtypes(include=['object']).columns:
    if column != 'Churn?':
        display(pd.crosstab(index=churn[column], columns=churn['Churn?'], normalize='columns'))

for column in churn.select_dtypes(exclude=['object']).columns:
    print(column)
    hist = churn[[column, 'Churn?']].hist(by='Churn?', bins=30)
    plt.show()
```

Interestingly we see that churners appear:

- Fairly evenly distributed geographically
- More likely to have an international plan
- Less likely to have a voicemail plan
- To exhibit some bimodality in daily minutes (either higher or lower than the
  average for non-churners)
- To have a larger number of customer service calls (which makes sense as we'd
  expect customers who experience lots of problems may be more likely to churn)

In addition, we see that churners take on very similar distributions for
features like Day Mins and Day Charge. That's not surprising as we'd expect
minutes spent talking to correlate with charges. Let's dig deeper into the
relationships between our features.

```python
display(churn.corr())
pd.plotting.scatter_matrix(churn, figsize=(12, 12))
plt.show()
```

We see several features that essentially have 100% correlation with one another.
Including these feature pairs in some machine learning algorithms can create
catastrophic problems, while in others it will only introduce minor redundancy
and bias. Let's remove one feature from each of the highly correlated pairs: Day
Charge from the pair with Day Mins, Night Charge from the pair with Night Mins,
Intl Charge from the pair with Intl Mins:

```python
churn = churn.drop(['Day Charge', 'Eve Charge', 'Night Charge', 'Intl Charge'], axis=1)
```

Now that we've cleaned up our dataset, let's determine which algorithm to use.
As mentioned above, there appear to be some variables where both high and low
(but not intermediate) values are predictive of churn. In order to accommodate
this in an algorithm like linear regression, we'd need to generate polynomial
(or bucketed) terms. Instead, let's attempt to model this problem using gradient
boosted trees. Amazon SageMaker provides an XGBoost container that we can use to
train in a managed, distributed setting, and then host as a real-time prediction
endpoint. XGBoost uses gradient boosted trees which naturally account for
non-linear relationships between features and the target variable, as well as
accommodating complex interactions between features.

Amazon SageMaker XGBoost can train on data in either a CSV or LibSVM format. For
this example, we'll stick with CSV. It should:

- Have the predictor variable in the first column
- Not have a header row

But first, let's convert our categorical features into numeric features.

```python
model_data = pd.get_dummies(churn)
model_data = pd.concat([model_data['Churn?_True.'], model_data.drop(['Churn?_False.', 'Churn?_True.'], axis=1)], axis=1)
```

And now let's split the data into training, validation, and test sets. This will
help prevent us from overfitting the model, and allow us to test the models
accuracy on data it hasn't already seen.

```python
train_data, validation_data, test_data = np.split(model_data.sample(frac=1, random_state=1729), [int(0.7 * len(model_data)), int(0.9 * len(model_data))])
train_data.to_csv('train.csv', header=False, index=False)
validation_data.to_csv('validation.csv', header=False, index=False)
```

Now we'll upload these files to S3.

```python
boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train/train.csv')).upload_file('train.csv')
boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'validation/validation.csv')).upload_file('validation.csv')
```

## Training

Moving onto training, first we'll need to specify the locations of the XGBoost
algorithm containers.

```python
from sagemaker.amazon.amazon_estimator import get_image_uri
container = get_image_uri(boto3.Session().region_name, 'xgboost')
```

Then, because we're training with the CSV file format, we'll create s3_inputs
that our training function can use as a pointer to the files in S3.

```python
s3_input_train = sagemaker.s3_input(s3_data='s3://{}/{}/train'.format(bucket, prefix), content_type='csv')
s3_input_validation = sagemaker.s3_input(s3_data='s3://{}/{}/validation/'.format(bucket, prefix), content_type='csv')
```

and then, start the training job

```python
sess = sagemaker.Session()

xgb = sagemaker.estimator.Estimator(container,
                                    role,
                                    train_instance_count=1,
                                    train_instance_type='ml.m4.xlarge',
                                    output_path='s3://{}/{}/output'.format(bucket, prefix),
                                    sagemaker_session=sess)

xgb.fit({'train': s3_input_train, 'validation': s3_input_validation})
```

[< Home](./readme.md)
