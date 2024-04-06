#!/usr/bin/env python
# coding: utf-8

# using TF2.x with Keras 2.x see https://keras.io/getting_started/ and https://github.com/tensorflow/tensorflow/issues/63849
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"



# # Autoencoders for Anomaly Detection

# In this example, you will train an autoencoder to detect anomalies on the ECG5000 dataset
# (http://www.timeseriesclassification.com/description.php?Dataset=ECG5000).
# This dataset contains 5,000 Electrocardiograms, each with 140 data points. We will use a
# simplified version of the dataset, where each example has been labeled either `0` (corresponding
# to an abnormal rhythm), or `1` (corresponding to a normal rhythm). We are interested in identifying
# the abnormal rhythms.
#
# Note: This is a labeled dataset, so you could phrase this as a supervised learning problem. The goal of
# this example is to illustrate anomaly detection concepts you can apply to larger datasets, where you do
# not have labels available (for example, if you had many thousands of normal rhythms, and only a small
# number of abnormal rhythms).
#
# How will you detect anomalies using an autoencoder? Recall that an autoencoder is trained to minimize
# reconstruction error. You will train an autoencoder on the normal rhythms only, then use it to reconstruct
# all the data. Our hypothesis is that the abnormal rhythms will have higher reconstruction error. You will
# then classify a rhythm as an anomaly if the reconstruction error surpasses a fixed threshold.
#
# Note: For additional info see this example: https://www.tensorflow.org/tutorials/generative/autoencoder




# ### Import TensorFlow and other libraries

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model


# ### Load ECG data

# The dataset is based on one from timeseriesclassification.com - http://www.timeseriesclassification.com/description.php?Dataset=ECG5000.

# Download the dataset
# dataframe = pd.read_csv('http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv', header=None)
dataframe = pd.read_csv('/files/pico/ML/atlantis-example/ecg.csv', header=None)
raw_data = dataframe.values
print(dataframe.head())




# The last element contains the labels (1.0, 0.0)
labels = raw_data[:, -1]

# The other data points are the electrocadriogram data
data = raw_data[:, 0:-1]

train_data, test_data, train_labels, test_labels = train_test_split(
    data, labels, test_size=0.2, random_state=21
)


# Normalize the data to `[0,1]` to improve training accuracy.

min_val = tf.reduce_min(train_data)
max_val = tf.reduce_max(train_data)

train_data = (train_data - min_val) / (max_val - min_val)
test_data = (test_data - min_val) / (max_val - min_val)

train_data = tf.cast(train_data, tf.float32)
test_data = tf.cast(test_data, tf.float32)


# now train the autoencoder using only the normal rhythms, which are labeled in
# this dataset as `1`. Separate the normal rhythms from the abnormal rhythms.

train_labels = train_labels.astype(bool)
test_labels = test_labels.astype(bool)

normal_train_data = train_data[train_labels]
normal_test_data = test_data[test_labels]

anomalous_train_data = train_data[~train_labels]
anomalous_test_data = test_data[~test_labels]


# Plot a normal ECG.

plt.grid()
plt.plot(np.arange(140), normal_train_data[0])
plt.title("A Normal ECG")
plt.show()


# Plot an anomalous ECG.

plt.grid()
plt.plot(np.arange(140), anomalous_train_data[0])
plt.title("An Anomalous ECG")
plt.show()


# ### Build the model
#
# After training and evaluating the example model, try modifying the size and number
# of layers to build an understanding for autoencoder architectures.
#
# Note: Changing the size of the embedding (the smallest layer) can produce interesting
# results.

class AnomalyDetector(Model):
  def __init__(self):
    super(AnomalyDetector, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Dense(32, activation="relu"),
      layers.Dense(16, activation="relu"),
      layers.Dense(8, activation="relu")]) # Smallest Layer Defined Here

    self.decoder = tf.keras.Sequential([
      layers.Dense(16, activation="relu"),
      layers.Dense(32, activation="relu"),
      layers.Dense(140, activation="sigmoid")])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = AnomalyDetector()

autoencoder.compile(optimizer='adam', loss='mae')



# ### Train the model
#
# Notice that the autoencoder is trained using only the normal ECGs, but is evaluated
# using the full test set.

history = autoencoder.fit(normal_train_data, normal_train_data,
          epochs=20,
          batch_size=512,
          validation_data=(test_data, test_data),
          shuffle=True)

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.title("trained model")
plt.show()




# ### Evaluate Training
#
# We will classify an ECG as anomalous if the reconstruction error is greater than one
# standard deviation from the normal training examples. First, we'll plot a normal ECG from the
# training set, the reconstruction after it's encoded and decoded by the autoencoder, and the
# reconstruction error.


encoded_imgs = autoencoder.encoder(normal_test_data).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

plt.plot(normal_test_data[0],'b')
plt.plot(decoded_imgs[0],'r')
plt.fill_between(np.arange(140), decoded_imgs[0], normal_test_data[0], color='lightcoral' )
plt.legend(labels=["Input", "Reconstruction", "Error"])
plt.title("normal ECG")
plt.show()


# Create a similar plot, this time for an anomalous example.

encoded_imgs = autoencoder.encoder(anomalous_test_data).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

plt.plot(anomalous_test_data[0],'b')
plt.plot(decoded_imgs[0],'r')
plt.fill_between(np.arange(140), decoded_imgs[0], anomalous_test_data[0], color='lightcoral' )
plt.legend(labels=["Input", "Reconstruction", "Error"])
plt.title("abnormal ECG")
plt.show()



# ### Detect anomalies

# Detect anomalies by calculating whether the reconstruction loss is greater than a fixed threshold.
# We will calculate the mean average error for normal examples from the training set, then classify future
# examples as anomalous if the reconstruction error is higher than one standard deviation from the training set.

# Plot the reconstruction error on normal ECGs from the training set

reconstructions = autoencoder.predict(normal_train_data)
train_loss = tf.keras.losses.mae(reconstructions, normal_train_data)

plt.hist(train_loss, bins=50)
plt.xlabel("Train loss")
plt.ylabel("No of examples")
plt.title("reconstruction error for normal ECG")
plt.show()


# Choose a threshold value that is one standard deviations above the mean.

threshold = np.mean(train_loss) + np.std(train_loss)
print("Reconstruction Error Threshold (eg 1 standard deviation): ", threshold)


# Note: There are other strategies you could use to select a threshold value above which test examples
# should be classified as anomalous, the correct approach will depend on your dataset.

# If you examine the recontruction error for the anomalous examples in the test set, you'll notice most
# have greater reconstruction error than the threshold. By varing the threshold, you can adjust the precision
# and recall of your classifier. See https://developers.google.com/machine-learning/glossary#precision and
# https://developers.google.com/machine-learning/glossary#recall.

reconstructions = autoencoder.predict(anomalous_test_data)
test_loss = tf.keras.losses.mae(reconstructions, anomalous_test_data)

plt.hist(test_loss, bins=50)
plt.xlabel("Test loss")
plt.ylabel("No of examples")
plt.title("abnormal ECG vs reconstruction error")
plt.show()




# Classify an ECG as an anomaly if the reconstruction error is greater than the threshold.

def predict(model, data, threshold):
  reconstructions = model(data)
  loss = tf.keras.losses.mae(reconstructions, data)
  return tf.math.less(loss, threshold), loss

def print_stats(predictions, labels):
  print("Accuracy = {}".format(accuracy_score(labels, predictions)))
  print("Precision = {}".format(precision_score(labels, predictions)))
  print("Recall = {}".format(recall_score(labels, predictions)))


preds, scores = predict(autoencoder, test_data, threshold)
print_stats(preds, test_labels)


# ### ROC and AUC Metrics
#
# We've created a fairly accurate model for anomaly detection but our accuracy is highly dependant on the threshold we select.
#
# What if we wanted to evaluate how different thresholds impact our true positive and false positive rates?
#
# Enter Receiver Operating Characteristic (ROC) plots. This metric allows us to visualize the tradeoff between predicting
# anomalies as normal (false positives) and predicting normal data as an anomaly (false negative). Remember that normal rhythms
# are labeled as `1` in this dataset.


fpr = []
tpr = []

#the test labels are flipped to match how the roc_curve function expects them.
flipped_labels = 1 - test_labels

fpr, tpr, _ = roc_curve(flipped_labels, scores)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve ')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# Since our model does a great job in diferentiating normal rythms from abnormal ones it seems
# easy to pick the threshold that would give us the high true positive rate (TPR) and low false
# positive rate (FPR) that is at the 'knee' of the curve.
#
# However, in some cases there may be an application constraint that requires a specific TPR or FPR,
# in which case we would have to move off of the 'knee' and sacrifice overall accuracy. In this case we
# might rather have false alarms than miss a potentially dangerous rythm.



# Now that we understand how to visualize the impact of the selected threshold, what if we wanted to
# compare the performance of models without factoring in the threshold? Simply comparing the accuracy
# won't work since that depends on the threshold you pick and that won't have the same impact across models.
#
# Instead we can measure the area under the curve (AUC) in the ROC plot. One way to interpret the AUC metric
# is as the probability that the model ranks a random positive example more highly than a random negative example.
#
# In general the AUC is a useful metic for comparison as it is threshold invariant *and* scale invariant



roc_auc = auc(fpr, tpr)
print("ROC area under the curve: ", roc_auc)


# ## Links to Continue Learning
#
# If you would like to learn more about anomaly detection with autoencoders, check out this excellent
# interactive example built with TensorFlow.js by Victor Dibia: https://anomagram.fastforwardlabs.com/#/

# For a real-world use case, you can learn how Airbus Detects Anomalies in ISS Telemetry Data using TensorFlow:
# https://blog.tensorflow.org/2020/04/how-airbus-detects-anomalies-iss-telemetry-data-tfx.html
#
# To learn more about the basics of autoencoders, try the basis for this colab, TensorFlow's Intro to Autoencoders:
# https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/generative/autoencoder.ipynb#scrollTo=xfNT-mlFwxVM
#
# For more info, consider reading this blog post by François Chollet, and check out chapter 14 from Deep Learning
# by Ian Goodfellow, Yoshua Bengio, and Aaron Courville.
#
# https://blog.keras.io/building-autoencoders-in-keras.html
# https://www.deeplearningbook.org/
