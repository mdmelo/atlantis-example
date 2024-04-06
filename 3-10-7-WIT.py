#!/usr/bin/env python
# coding: utf-8

# # Introduction to Responsible AI using Google's What-If Tool (WIT)
#
# This notebook will walk you through exploring a real-life dataset, and how to identify sources
# of unfairness in the data across different social groups. In this case, we will train a DNN on the UCI census problem
# predicting whether a person earns more than $50K from their census information, and visualize its results using Google's
# What-If Tool.
#
# see https://archive.ics.uci.edu/ml/datasets/census+income) and https://pair-code.github.io/what-if-tool
#
# This notebook has been adapted from Google's WIT model comparison notebook.
# see https://colab.research.google.com/github/pair-code/what-if-tool/blob/master/WIT_Model_Comparison.ipynb.
#
# This notebook has 3 parts:
#     Part 1 is dedicated to setting up and training the UCI dataset, then starting the What-If Tool within your browser.
#     Part 2 contains a series of instructions for you to follow along in your instance of the What-If Tool (instantiated in Part 1).
#     Part 3 has a short knowledge check for self-evaluation.

# # Part 1: Run the What-If Tool within Collab

# ## Section A: Setting up the UCI census dataset and training the DNN classifier

# ### Install the What-If Tool widget  (python -m pip install witwidget)



# ### helper functions

import pandas as pd
import numpy as np
import tensorflow as tf
import functools

# Creates a tf feature spec from the dataframe and columns specified.
def create_feature_spec(df, columns=None):
    feature_spec = {}
    if columns == None:
        columns = df.columns.values.tolist()
    for f in columns:
        if df[f].dtype is np.dtype(np.int64):
            feature_spec[f] = tf.io.FixedLenFeature(shape=(), dtype=tf.int64)
        elif df[f].dtype is np.dtype(np.float64):
            feature_spec[f] = tf.io.FixedLenFeature(shape=(), dtype=tf.float32)
        else:
            feature_spec[f] = tf.io.FixedLenFeature(shape=(), dtype=tf.string)
    return feature_spec

# Creates simple numeric and categorical feature columns from a feature spec and a
# list of columns from that spec to use.
#
# NOTE: Models might perform better with some feature engineering such as bucketed
# numeric columns and hash-bucket/embedding columns for categorical features.
def create_feature_columns(columns, feature_spec):
    ret = []
    for col in columns:
        if feature_spec[col].dtype is tf.int64 or feature_spec[col].dtype is tf.float32:
            ret.append(tf.feature_column.numeric_column(col))
        else:
            ret.append(tf.feature_column.indicator_column(
                tf.feature_column.categorical_column_with_vocabulary_list(col, list(df[col].unique()))))
    return ret

# An input function for providing input to a model from tf.Examples
def tfexamples_input_fn(examples, feature_spec, label, mode=tf.estimator.ModeKeys.EVAL,
                        num_epochs=None,
                       batch_size=64):
    def ex_generator():
        for i in range(len(examples)):
            yield examples[i].SerializeToString()
    dataset = tf.data.Dataset.from_generator(
        ex_generator, tf.dtypes.string, tf.TensorShape([]))
    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.shuffle(buffer_size=2 * batch_size + 1)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda tf_example: parse_tf_example(tf_example, label, feature_spec))
    dataset = dataset.repeat(num_epochs)
    return dataset

# Parses Tf.Example protos into features for the input function.
def parse_tf_example(example_proto, label, feature_spec):
    parsed_features = tf.io.parse_example(serialized=example_proto, features=feature_spec)
    target = parsed_features.pop(label)
    return parsed_features, target

# Converts a dataframe into a list of tf.Example protos.
def df_to_examples(df, columns=None):
    examples = []
    if columns == None:
        columns = df.columns.values.tolist()
    for index, row in df.iterrows():
        example = tf.train.Example()
        for col in columns:
            if df[col].dtype is np.dtype(np.int64):
                example.features.feature[col].int64_list.value.append(int(row[col]))
            elif df[col].dtype is np.dtype(np.float64):
                example.features.feature[col].float_list.value.append(row[col])
            elif row[col] == row[col]:
                example.features.feature[col].bytes_list.value.append(row[col].encode('utf-8'))
        examples.append(example)
    return examples

# Converts a dataframe column into a column of 0's and 1's based on the provided test.
# Used to force label columns to be numeric for binary classification using a TF estimator.
def make_label_column_numeric(df, label_column, test):
    df[label_column] = np.where(test(df[label_column]), 1, 0)


# ### Set up and pre-process the dataset for training

#Read training dataset from CSV

import pandas as pd

# Set the path to the CSV containing the dataset to train on.
# csv_path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
csv_path = "/files/pico/ML/atlantis-example//adult.data"

# Set the column names for the columns in the CSV.
csv_columns = [
    "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital-Status",
  "Occupation", "Relationship", "Race", "Sex", "Capital-Gain", "Capital-Loss",
  "Hours-per-week", "Country", "Over-50K"]

# Read the dataset from the provided CSV and print out information about it.
df = pd.read_csv(csv_path, names=csv_columns, skipinitialspace=True)
print(df.head())



### Specify input columns and columns to predict

import numpy as np

# Set the column in the dataset you wish for the model to predict
label_column = 'Over-50K'

# Make the label column numeric (0 and 1), for use in our model.
# In this case, examples with a target value of '>50K' are considered to be in
# the '1' (positive) class and all other examples are considered to be in the
# '0' (negative) class.
make_label_column_numeric(df, label_column, lambda val: val == '>50K')

# Set list of all columns from the dataset we will use for model input.
input_features = [
    'Age', 'Workclass', 'Education', 'Marital-Status', 'Occupation',
  'Relationship', 'Race', 'Sex', 'Capital-Gain', 'Capital-Loss',
  'Hours-per-week', 'Country']

# Create a list containing all input features and the label column
features_and_labels = input_features + [label_column]

# Convert dataset to tf.Example protos
examples = df_to_examples(df)




# ### Create and train the DNN classifier

num_steps = 2000  #@param {type: "number"}


# Create a feature spec for the classifier
feature_spec = create_feature_spec(df, features_and_labels)

# Define and train classifier
train_inpf = functools.partial(tfexamples_input_fn, examples, feature_spec, label_column)
classifier = tf.estimator.DNNClassifier(feature_columns=create_feature_columns(input_features, feature_spec), hidden_units=[128, 64, 32])
classifier.train(train_inpf, steps=num_steps)


# ## Launch the What-If Tool

num_datapoints = 1000  #@param {type: "number"}
tool_height_in_px = 1000  #@param {type: "number"}

from witwidget.notebook.visualization import WitConfigBuilder
from witwidget.notebook.visualization import WitWidget

# Load up the test dataset
# test_csv_path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'
test_csv_path = "/files/pico/ML/atlantis-example//adult.test"
test_df = pd.read_csv(test_csv_path, names=csv_columns, skipinitialspace=True, skiprows=1)
make_label_column_numeric(test_df, label_column, lambda val: val == '>50K.')
test_examples = df_to_examples(test_df[0:num_datapoints])

# Setup the tool with the test examples and the trained classifier
config_builder = WitConfigBuilder(test_examples[0:num_datapoints]).set_estimator_and_feature_spec(
    classifier, feature_spec).set_label_vocab(['Under 50K', 'Over 50K'])
a = WitWidget(config_builder, height=tool_height_in_px)


# # Part 2: Exploring the UCI census dataset using the What-If Tool

# ### Now that we have the What-If Tool (WIT) running, let's use it to find potential sources of bias and unfairness!
#
# Short forms used throughout: Number (#), True Positive (TP), True Negative (TN), False Positive (FP), False Negative (FN).
#
# 1. Click on the “Performance & Fairness” (PF) tab. You will see that the positive classification threshold is set as 0.5 by default for both models.
#
# ![pf_click](https://static.vpal.harvard.edu/cdn/courses/TinyML_02/3-10-7_images/screenshot_20201202_003512.png)
#
# 2. Click on the "Features" tab. For key features such as sex and race, you will notice the distribution of the data is skewed heavily.
#
# ![feature_click](https://static.vpal.harvard.edu/cdn/courses/TinyML_02/3-10-7_images/screenshot_20201202_002557.png)
#
# 3. Now go back to the PF tab. Set Ground Truth Feature to "Over-50K" and slice by sex. You should now see how the model performs
# for men and women separately. We observe that the model is more accurate for women than men. The percentage of FPs and FNs for men
# is much higher as well.
#
# ![pf_slice](https://static.vpal.harvard.edu/cdn/courses/TinyML_02/3-10-7_images/screenshot_20201202_003026.png)
#
# 4. Let's dig deeper into why this is the case. Open up the Male and Female tabs and take a look at their confusion matrices.
#
# ![sex_confusion_matrices](https://static.vpal.harvard.edu/cdn/courses/TinyML_02/3-10-7_images/screenshot_20201202_004610.png)
#
# 5. We see that men are far better represented than women in the dataset (~700 men vs ~300 women). We also see that even out of those
# 300 women, only 12.5% make over $50K. For men, this rises to 29%. Our model is probably overfitting to the data, and we have a lack of
# female representation in the dataset to boot.
#
# 6. So how do we try to make our model more fair? Let's say we are trying to approve loans based on the income prediction. One way is to try
# different threshold optimization strategies for the two groups.
#
# 7. As a first attempt, let's optimize our thresholds to enforce equal accuracy for men and women. Click on the "**Equal Accuracy**" radio button
# under the Fairness window.
#
# ![equal_accuracy](https://static.vpal.harvard.edu/cdn/courses/TinyML_02/3-10-7_images/screenshot_20201203_235347.png)
#
# 8. We see that in an attempt to preserve equal accuracy across men and women, the strategy lowered the threshold for women from 0.5 to 0.2, and
# increased the threshold for men from 0.5 to 0.85. Accuracy for men got slightly better by skewing their results negative; the # of TNs went up,
# increasing the accuracy, but the # of TPs went down slightly. Skewing the results positive for women resulted in many more deserving women
# being offered loans (TPs went up) but consequently ballooned the FP rate. This strategy turns out to be unfair to men.
#
# 9.  Let's try something different; click on the "Demographic Parity" radio button. This strategy makes sure that if 30% of the applicants
# are women, then 30% of the approved loans should be women as well.
#
# ![demographic_parity](https://static.vpal.harvard.edu/cdn/courses/TinyML_02/3-10-7_images/screenshot_20201204_002639.png)
#
# 8. However this turns out to be unfair to men as well because more non loan-worthy women may turn out to be approved, in order to preserve equal
# proportions of men and women. This is reflected by the higher # of FPs for women, and the increase in FNs for men.
#
# 9. So far, trying to enforce equal outcome (either by enforcing equal accuracy or equal proportion) has been resulting in deserving candidates
# from the more advantaged group to be denied loans (because these strategies skew their results negative). Perhaps we should try to optimize for
# equal opportunity instead, where the same percentage of deserving men and women are given loans, i.e. the percentage of TPs are the same for men and women.
# Now click on the "Equal Opportunity" radio button.
#
# ![equal_opportunity](https://static.vpal.harvard.edu/cdn/courses/TinyML_02/3-10-7_images/screenshot_20201204_012958.png)
#
# 10. Here, we observe that the TP rate for men and women is equal (56/202 ~ 11/38). Using this strategy, deserving individuals,
# regardless of gender, have an equal chance of being approved for a loan. This strategy is also the fairest out of all the ones we tried,
# for this particular dataset. (caveat: strategies for fairness can vary wildly depending on what your dataset looks like and what kind of
# outcome do you want to achieve!)

# # Part 3: Learning check
#
# ### Use the questions below to self-evaluate your learnings!
#
# 1. Which one of the following is the fairest threshold optimization strategy for this dataset?
# a. Single threshold
# b. Demographic parity
# c. Equal opportunity
# d. Equal accuracy
#
#
# 2. Which of the following are ways of mitigating bias in a dataset/model?
# a. Increasing representation for specific social groups
# b. Increasing overall accuracy
# c. Adaptive thresholding
# d. Avoiding overfitting

# ### Expand this subsection to view the answers

# 1. c
# 2. a, c, d
