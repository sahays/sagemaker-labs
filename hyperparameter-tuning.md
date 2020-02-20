# Lab: Tune your models: Hyperparameter tuning with Amazon SageMaker

## Overview

Direct marketing, either through mail, email, phone, etc., is a common tactic to
acquire customers. Because resources and a customer's attention is limited, the
goal is to only target the subset of prospects who are likely to engage with a
specific offer. Predicting those potential customers based on readily available
information like demographics, past interactions, and environmental factors is a
common machine learning problem.

## Preparation

Let's setup the Amazon S3 bucket and a prefix, and the IAM role

```python
import sagemaker
import boto3
from sagemaker.tuner import IntegerParameter, CategoricalParameter, ContinuousParameter, HyperparameterTuner

# For matrix operations and numerical processing
import numpy as np
# For munging tabular data
import pandas as pd
import os

region = boto3.Session().region_name
smclient = boto3.Session().client('sagemaker')

role = sagemaker.get_execution_role()

bucket = sagemaker.Session().default_bucket()
prefix = 'sagemaker/DEMO-hpo-xgboost-dm'
```

## Download data

```console
!wget -N https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip
!unzip -o bank-additional.zip
```

Read them into pandas frames

```python
data = pd.read_csv('./bank-additional/bank-additional-full.csv', sep=';')
pd.set_option('display.max_columns', 500)     # Make sure we can see all of the columns
pd.set_option('display.max_rows', 50)         # Keep the output on one page
data
```

## Understand the data

Let's talk about the data. At a high level, we can see:

- We have a little over 40K customer records, and 20 features for each customer
- The features are mixed; some numeric, some categorical
- The data appears to be sorted, at least by time and contact, maybe more

Specifics on each of the features:

- Demographics:

  - age: Customer's age (numeric)
  - job: Type of job (categorical: 'admin.', 'services', ...)
  - marital: Marital status (categorical: 'married', 'single', ...)
  - education: Level of education (categorical: 'basic.4y', 'high.school', ...)

- Past customer events:

  - default: Has credit in default? (categorical: 'no', 'unknown', ...)
  - housing: Has housing loan? (categorical: 'no', 'yes', ...)
  - loan: Has personal loan? (categorical: 'no', 'yes', ...)

- Past direct marketing contacts:

  - contact: Contact communication type (categorical: 'cellular', 'telephone',
    ...)
  - month: Last contact month of year (categorical: 'may', 'nov', ...)
  - day_of_week: Last contact day of the week (categorical: 'mon', 'fri', ...)
  - duration: Last contact duration, in seconds (numeric). Important note: If
    duration = 0 then y = 'no'.

- Campaign information:

  - campaign: Number of contacts performed during this campaign and for this
    client (numeric, includes last contact)
  - pdays: Number of days that passed by after the client was last contacted
    from a previous campaign (numeric)
  - previous: Number of contacts performed before this campaign and for this
    client (numeric)
  - poutcome: Outcome of the previous marketing campaign (categorical:
    'nonexistent','success', ...)

- External environment factors:

  - emp.var.rate: Employment variation rate - quarterly indicator (numeric)
  - cons.price.idx: Consumer price index - monthly indicator (numeric)
  - cons.conf.idx: Consumer confidence index - monthly indicator (numeric)
  - euribor3m: Euribor 3 month rate - daily indicator (numeric)
  - nr.employed: Number of employees - quarterly indicator (numeric)

- Target variable:
  - y: Has the client subscribed a term deposit? (binary: 'yes','no')

## Data transformation

Cleaning up data is part of nearly every machine learning project. It arguably
presents the biggest risk if done incorrectly and is one of the more subjective
aspects in the process.

### Convert categorical to numeric

The most common method is one hot encoding, which for each feature maps every
distinct value of that column to its own feature which takes a value of 1 when
the categorical feature is equal to that value, and 0 otherwise.

First of all, Many records have the value of "999" for pdays, number of days
that passed by after a client was last contacted. It is very likely to be a
magic number to represent that no contact was made before. Considering that, we
create a new column called "no_previous_contact", then grant it value of "1"
when pdays is 999 and "0" otherwise.

In the "job" column, there are categories that mean the customer is not working,
e.g., "student", "retire", and "unemployed". Since it is very likely whether or
not a customer is working will affect his/her decision to enroll in the term
deposit, we generate a new column to show whether the customer is working based
on "job" column.

Let's convert categorical to numeric.

```python
# Indicator variable to capture when pdays takes a value of 999
data['no_previous_contact'] = np.where(data['pdays'] == 999, 1, 0)
# Indicator for individuals not actively employed
data['not_working'] = np.where(np.in1d(data['job'], ['student', 'retired', 'unemployed']), 1, 0)
# Convert categorical variables to sets of indicators
model_data = pd.get_dummies(data)
model_data
```

### Drop fields

Another question to ask yourself before building a model is whether certain
features will add value in your final use case. For example, if your goal is to
deliver the best prediction, then will you have access to that data at the
moment of prediction? Knowing it's raining is highly predictive for umbrella
sales, but forecasting weather far enough out to plan inventory on umbrellas is
probably just as difficult as forecasting umbrella sales without knowledge of
the weather. So, including this in your model may give you a false sense of
precision.

Following this logic, let's remove the economic features and duration from our
data as they would need to be forecasted with high precision to use as inputs in
future predictions.

Even if we were to use values of the economic indicators from the previous
quarter, this value is likely not as relevant for prospects contacted early in
the next quarter as those contacted later on.

```python
model_data = model_data.drop(['duration', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed'], axis=1)
```

### Split data

We'll then split the dataset into training (70%), validation (20%), and test
(10%) datasets and convert the datasets to the right format the algorithm
expects. We will use training and validation datasets during training. Test
dataset will be used to evaluate model performance after it is deployed to an
endpoint.

Amazon SageMaker's XGBoost algorithm expects data in the libSVM or CSV data
format. For this example, we'll stick to CSV. Note that the first column must be
the target variable and the CSV should not include headers. Also, notice that
although repetitive it's easiest to do this after the train|validation|test
split rather than before. This avoids any misalignment issues due to random
reordering.

```python
train_data, validation_data, test_data = np.split(model_data.sample(frac=1, random_state=1729), [int(0.7 * len(model_data)), int(0.9*len(model_data))])

pd.concat([train_data['y_yes'], train_data.drop(['y_no', 'y_yes'], axis=1)], axis=1).to_csv('train.csv', index=False, header=False)
pd.concat([validation_data['y_yes'], validation_data.drop(['y_no', 'y_yes'], axis=1)], axis=1).to_csv('validation.csv', index=False, header=False)
pd.concat([test_data['y_yes'], test_data.drop(['y_no', 'y_yes'], axis=1)], axis=1).to_csv('test.csv', index=False, header=False)
```

### Copy files to Amazon S3

Now we'll copy the file to S3 for Amazon SageMaker training to pickup.

```python
boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train/train.csv')).upload_file('train.csv')
boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'validation/validation.csv')).upload_file('validation.csv')
```

## Setup Hyperparameter tuning

> Note, with the default setting below, the hyperparameter tuning job can take
> about 30 minutes to complete.

Now that we have prepared the dataset, we are ready to train models. Before we
do that, one thing to note is there are algorithm settings which are called
"hyperparameters" that can dramtically affect the performance of the trained
models. For example, XGBoost algorithm has dozens of hyperparameters and we need
to pick the right values for those hyperparameters in order to achieve the
desired model training results. Since which hyperparameter setting can lead to
the best result depends on the dataset as well, it is almost impossible to pick
the best hyperparameter setting without searching for it, and a good search
algorithm can search for the best hyperparameter setting in an automated and
effective way.

We will use SageMaker hyperparameter tuning to automate the searching process
effectively. Specifically, we specify a range, or a list of possible values in
the case of categorical hyperparameters, for each of the hyperparameter that we
plan to tune. SageMaker hyperparameter tuning will automatically launch multiple
training jobs with different hyperparameter settings, evaluate results of those
training jobs based on a predefined "objective metric", and select the
hyperparameter settings for future attempts based on previous results. For each
hyperparameter tuning job, we will give it a budget (max number of training
jobs) and it will complete once that many training jobs have been executed.

### Configure hyperparameters

In this example, we are using SageMaker Python SDK to set up and manage the
hyperparameter tuning job. We first configure the training jobs the
hyperparameter tuning job will launch by initiating an estimator, which
includes:

- The container image for the algorithm (XGBoost)
- Configuration for the output of the training jobs
- The values of static algorithm hyperparameters, those that are not specified
  will be given default values
- The type and number of instances to use for the training jobs

```python
from sagemaker.amazon.amazon_estimator import get_image_uri

sess = sagemaker.Session()

container = get_image_uri(region, 'xgboost', repo_version='latest')

xgb = sagemaker.estimator.Estimator(container,
                                    role,
                                    train_instance_count=1,
                                    train_instance_type='ml.m4.xlarge',
                                    output_path='s3://{}/{}/output'.format(bucket, prefix),
                                    sagemaker_session=sess)

xgb.set_hyperparameters(eval_metric='auc',
                        objective='binary:logistic',
                        num_round=100,
                        rate_drop=0.3,
                        tweedie_variance_power=1.4)
```

### Specify hyperparameters

We will tune four hyperparameters in this examples:

- eta: Step size shrinkage used in updates to prevent overfitting. After each
  boosting step, you can directly get the weights of new features. The eta
  parameter actually shrinks the feature weights to make the boosting process
  more conservative.
- alpha: L1 regularization term on weights. Increasing this value makes models
  more conservative.
- min_child_weight: Minimum sum of instance weight (hessian) needed in a child.
  If the tree partition step results in a leaf node with the sum of instance
  weight less than min_child_weight, the building process gives up further
  partitioning. In linear regression models, this simply corresponds to a
  minimum number of instances needed in each node. The larger the algorithm, the
  more conservative it is.
- max_depth: Maximum depth of a tree. Increasing this value makes the model more
  complex and likely to be overfitted.

```python
hyperparameter_ranges = {'eta': ContinuousParameter(0, 1),
                        'min_child_weight': ContinuousParameter(1, 10),
                        'alpha': ContinuousParameter(0, 2),
                        'max_depth': IntegerParameter(1, 10)}
```

### Specify objective metric

Next we'll specify the objective metric that we'd like to tune and its
definition, which includes the regular expression (Regex) needed to extract that
metric from the CloudWatch logs of the training job. Since we are using built-in
XGBoost algorithm here, it emits two predefined metrics: validation:auc and
train:auc, and we elected to monitor validation:auc as you can see below. In
this case, we only need to specify the metric name and do not need to provide
regex. If you bring your own algorithm, your algorithm emits metrics by itself.
In that case, you'll need to add a MetricDefinition object here to define the
format of those metrics through regex, so that SageMaker knows how to extract
those metrics from your CloudWatch logs.

```python
objective_metric_name = 'validation:auc'
```

### Create HyperparameterTuner object

Now, we'll create a HyperparameterTuner object, to which we pass:

- The XGBoost estimator we created above
- Our hyperparameter ranges
- Objective metric name and definition
- Tuning resource configurations such as Number of training jobs to run in total
  and how many training jobs can be run in parallel.

```python
tuner = HyperparameterTuner(xgb,
                            objective_metric_name,
                            hyperparameter_ranges,
                            max_jobs=20,
                            max_parallel_jobs=3)
```

## Launch Hyperparameter tuning

Now we can launch a hyperparameter tuning job by calling fit() function. After
the hyperparameter tuning job is created, we can go to SageMaker console to
track the progress of the hyperparameter tuning job until it is completed.

```python
s3_input_train = sagemaker.s3_input(s3_data='s3://{}/{}/train'.format(bucket, prefix), content_type='csv')
s3_input_validation = sagemaker.s3_input(s3_data='s3://{}/{}/validation/'.format(bucket, prefix), content_type='csv')

tuner.fit({'train': s3_input_train, 'validation': s3_input_validation}, include_cls_metadata=False)
```

Let's just run a quick check of the hyperparameter tuning jobs status to make
sure it started successfully.

```python
boto3.client('sagemaker').describe_hyper_parameter_tuning_job(
    HyperParameterTuningJobName=tuner.latest_tuning_job.job_name)['HyperParameterTuningJobStatus']
```

## Analyze tuning job results

Once the tuning jobs have completed, we can compare the distribution of the
hyperparameter configurations chosen in the two cases.

```python
tuner_log = HyperparameterTuner(
    xgb,
    objective_metric_name,
    hyperparameter_ranges,
    max_jobs=20,
    max_parallel_jobs=10,
    strategy='Random'
)

tuner_log.fit({'train': s3_input_train, 'validation': s3_input_validation}, include_cls_metadata=False)
```

```
boto3.client('sagemaker').describe_hyper_parameter_tuning_job(
    HyperParameterTuningJobName=tuner_log.latest_tuning_job.job_name)['HyperParameterTuningJobStatus']
```

```python
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# check jobs have finished
status_log = boto3.client('sagemaker').describe_hyper_parameter_tuning_job(
    HyperParameterTuningJobName=tuner_log.latest_tuning_job.job_name)['HyperParameterTuningJobStatus']
status_linear = boto3.client('sagemaker').describe_hyper_parameter_tuning_job(
    HyperParameterTuningJobName=tuner_linear.latest_tuning_job.job_name)['HyperParameterTuningJobStatus']

assert status_log == 'Completed', "First must be completed, was {}".format(status_log)
assert status_linear == 'Completed', "Second must be completed, was {}".format(status_linear)

df_log = sagemaker.HyperparameterTuningJobAnalytics(tuner_log.latest_tuning_job.job_name).dataframe()
df_linear = sagemaker.HyperparameterTuningJobAnalytics(tuner_linear.latest_tuning_job.job_name).dataframe()
df_log['scaling'] = 'log'
df_linear['scaling'] = 'linear'
df = pd.concat([df_log, df_linear], ignore_index=True)
```

```python
g = sns.FacetGrid(df, col="scaling", palette='viridis')
g = g.map(plt.scatter, "alpha", "lambda", alpha=0.6)
```

[< Home](./readme.md)
