#!/usr/bin/env python
# coding: utf-8


# using TF2.x with Keras 2.x see https://keras.io/getting_started/ and https://github.com/tensorflow/tensorflow/issues/63849
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"


# # Mask-Detection using Transfer Learning
#
# In this assignment you will use transfer learning to detect if the person is wearing a mask or not.
# We will use a pre-trained model (MobileNet-V1) trained on the ImageNet dataset and use a modified version of
# [kaggle mask dataset](https://www.kaggle.com/prasoonkottarathil/face-mask-lite-dataset) to preform transfer learning.
#
# Remember, a pre-trained model is a saved network that was previously trained on a large dataset, typically on a large-scale image-classification task. You either use the pretrained model as is or use transfer learning to customize this model to a given task. The intuition behind transfer learning for image classification is that if a model is trained on a large and general enough dataset, this model will effectively serve as a generic model of the visual world. You can then take advantage of these learned feature maps without having to start from scratch by training a large model on a large dataset.
#
#  In this assignment, you will customize a pretrained model using Feature Extraction:
#
#  Use the representations learned by a previous network to extract meaningful features from new samples.
#    You simply add a new classifier, which will be trained from scratch, on top of the pretrained model so that you can repurpose
#    the feature maps learned previously for the dataset.
#
#  You do not need to (re)train the entire model. The base convolutional network already contains features that are generically useful for classifying pictures.
#  However, the final, classification part of the pretrained model is specific to the original classification task, and subsequently specific to the set of
#  classes on which the model was trained.
#
#
# You will follow the general machine learning workflow.
#
# 1. Examine and understand the data
# 2. Build an input pipeline, in this case using Keras ImageDataGenerator
# 3. Compose the model
#    * Load in the pretrained base model (and pretrained weights)
#    * Stack the classification layers on top
# 4. Train the model
# 5. Evaluate model



import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from tensorflow.keras.preprocessing import image_dataset_from_directory


# ## Build the dataset

# ### Download the dataset
#
# You will use a dataset containing several hundreds images of person with and without mask.
# First you will download and extract a zip file containing the images.
# Then  you will create a `tf.data.Dataset` for training and validation using the `tf.keras.preprocessing.image_dataset_from_directory` utility.
# You can learn more about loading images in this [tutorial](https://www.tensorflow.org/tutorials/load_data/images).

# get_ipython().system('pip install --upgrade --no-cache-dir gdown')
# get_ipython().system('gdown --id 1lYOgCLLJU8TCIeTxJHkjsxBq_GPzQYb9')
# get_ipython().system('unzip edx_transfer_learningv3.zip')


import os
os.chdir("/content/edx_transfer_learningv3/edx_transfer_learning/train/without_mask")


path_to_zip = "/content/"
PATH = os.path.join(os.path.dirname(path_to_zip), 'edx_transfer_learningv3/edx_transfer_learning/')
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

BATCH_SIZE = 32
IMG_SIZE = (96, 96)
train_dataset = image_dataset_from_directory(train_dir,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE)


validation_dataset = image_dataset_from_directory(validation_dir,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  image_size=IMG_SIZE)


# Now that we have built the dataset lets view the first nine images and labels from the training set:

class_names = train_dataset.class_names

plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")


# ### Split the dataset
#
# As the original dataset doesn't contains a test set, you need to create one. To do so,
# determine how many batches of data are available in the validation set using ```tf.data.experimental.cardinality```,
# then move 20% of them to a test set.



val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)

print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))
print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))


# ### Configure the dataset for performance

# Use buffered prefetching to load images from disk without having I/O become blocking.
# To learn more about this method see the [data performance](https://www.tensorflow.org/guide/data_performance) guide.

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)


# ### Use data augmentation
# As we have explored earlier in this specialization, if you don't have a large image dataset,
# it's a good practice to artificially introduce sample diversity by applying random, yet realistic,
# transformations to the training images to expand the dataset. For example, you could apply rotations or
# flip the data horizontally. This helps expose the model to different aspects of the training data and
# reduce [overfitting](https://www.tensorflow.org/tutorials/keras/overfit_and_underfit).
#
# You can learn more about data augmentation in this [tutorial](https://www.tensorflow.org/tutorials/images/data_augmentation).


data_augmentation = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
                                         tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),])



# Note: These layers are active only during training (when you call `model.fit`). They are inactive when the model is used in
# inference mode (`model.evaluate` or `model.predict`).
#
# Let's repeatedly apply these layers to the same image and see the result to better understand why this augmentation can help the dataset generalize its learning.

for image, _ in train_dataset.take(1):
    plt.figure(figsize=(10, 10))
    first_image = image[0]
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
        plt.imshow(augmented_image[0] / 255)
        plt.axis('off')


# ### Rescale pixel values
#
# In a moment, you will download `tf.keras.applications.MobileNetV2` for use as your base model. This model expects
# pixel vaues in `[-1,1]`, but at this point, the pixel values in your images are in `[0-255]`. To rescale them,
# use the preprocessing method included with the model.

preprocess_input = tf.keras.applications.mobilenet.preprocess_input


# Note: Alternatively, you could rescale pixel values from `[0,255]` to `[-1, 1]` using a [Rescaling]
# (https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing/Rescaling) layer.

rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1)



# Note: If you go on to use other `tf.keras.applications`, be sure to check the API doc to determine if they
# expect pixels in `[-1,1]` or `[0,1]`, or use the included `preprocess_input` function.




# ## Build the Model

# ### Create the base model
# You will create the base model from the **MobileNet V1** model developed at Google. This is pre-trained on the ImageNet dataset,
# a large dataset consisting of 1.4M images and 1000 classes. ImageNet is a research training dataset with a wide variety of
# categories including objects such as `jackfruit` and `syringe`. This base of knowledge will help us classify if a person
# is wearing a mask or not from our specific dataset.
#
# First, you need to pick which layer of MobileNet V1 you will leverage as the high level features you wish to re-use. Since we want
# to adapt the classifications coming out of the model to a new task, we want to leverage the features coming out of the
# *last* layer BEFORE the classification layers. In many image models this is the output of the final convolution BEFORE the
# flatten layer. You may see this layer referred to as the "bottleneck layer" in some texts. Since many machine
# learning models are defined as the inputs occuring at the bottom and the outputs occuring at the top we would like to
# ignore the top few classification layers. Fortuntately, there is a shortcut to doing this in TensorFlow,
# **```include_top=False```**. By passing in this parameter we instantiate a MobileNet V1 model
# pre-loaded with weights trained on ImageNet that doesn't include the classification layers at the top, which is ideal for feature extraction.


# Create the base model from the pre-trained model MobileNet V2
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNet(input_shape=IMG_SHAPE,
                                             include_top=False,
                                             weights='imagenet')


image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)
print(feature_batch.shape)



# ### Freeze the convolutional layers
#
# It is important to freeze the convolutional layers before you compile and train the model with transfer learning.
# Freezing (by setting layer.trainable = False) prevents the weights in a given layer from being updated during training.
# We want to perform this operation because we want to leverage the pre-trained values in the convolutional layers and
# only learn new classification layer values. We can do this by setting the entire model's `trainable` flag to False.

base_model.trainable = False


# Let's take a look at the base model architecture
print(base_model.summary())


# ### Add a classification head

# To begin the process of generating classifications from the pretrained features, we use a
# `tf.keras.layers.GlobalAveragePooling2D` layer to convert the 5x5 spatial features into a single
# 1024-element feature vector per image.

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)


# We then apply a `tf.keras.layers.Dense` layer to convert the feature vector into a single prediction per image.
# You don't need an activation function here because this prediction will be treated as a `logit`, or a raw prediction value.
# Positive numbers predict class 1, negative numbers predict class 0.

prediction_layer = tf.keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)


# We can then build our final model by chaining together the data augmentation, rescaling, base_model and feature extractor
# layers using the [Keras Functional API](https://www.tensorflow.org/guide/keras/functional). Importantly we remind Tensorflow
# that we do not want to train the base_model!

inputs = tf.keras.Input(shape=(96, 96, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)


# ### Compile the model
#
# Compile the model before training it. Since there are two classes, use a binary cross-entropy loss with `from_logits=True` since the model provides a linear output. You can see below how we have included the Mobilenet V1 model after the input layer and before our classification layers!


base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

print(model.summary())



# ## Your Turn: Train and Evaluate the model
#
# Now that we have our model we can train it! You will see that since we are leveraging all of the pre-trained features
# we can improve our model from a random initialization (accuracy of ~50%) to a model with over 95% accuracy quite quickly.
# **How many epochs do you think we'll need?**

# ### Train the model
# First print the initial accuracy

loss0, accuracy0 = model.evaluate(validation_dataset)
print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))


# **Now its your turn to pick the number of epochs of training! Remmeber we are aiming for at least 95% accuracy on the test set!**
#
# *Hint: Despite the fact that it would take more than a day to train this model from scratch, it requires far
# fewer epochs to train it with transfer learning than you might suspect! You probably have more fingers and toes
# than the number of epochs you will need!*


EPOCHS = 2
history = model.fit(train_dataset,
                    epochs=EPOCHS,
                    validation_data=validation_dataset)


# Did you hit you accuracy goal? Did you overshoot and spend some extra time training? When might you have been
# able to quit? Lets take a look at the learning curves of the training and validation accuracy/loss to analyze our results.

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


# ### Evaluate your model
#
# The last thing we need to do is check if the model is overfitting or if it actually learned the problem
# that fast! Does you model still perform well on the test set? Lets find out!

loss, accuracy = model.evaluate(test_dataset)
print('Test accuracy :', accuracy)


# And assuming your model passed the accuracy threshold you are now are all set to use this model to
# predict if the person is wearing a mask or not. Lets print the results from a bunch of the images in our test dataset!

#Retrieve a batch of images from the test set
image_batch, label_batch = test_dataset.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch).flatten()

# Apply a sigmoid since our model returns logits
predictions = tf.nn.sigmoid(predictions)
predictions = tf.where(predictions < 0.5, 0, 1)

print('Predictions:\n', predictions.numpy())
print('Labels:\n', label_batch)

plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image_batch[i].astype("uint8"))
    plt.title(class_names[predictions[i]])
    plt.axis("off")
    plt.show()

# ## Summary
#
# * **Using transfer learning for Mask Detection**: In this colab, we learned how we can use transfer learning to
# detect if a person is wearing mask or not. When working with a small dataset, it is a common practice to take advantage
# of features learned by a model trained on a larger dataset in the same domain.
