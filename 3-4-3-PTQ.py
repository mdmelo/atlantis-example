#!/usr/bin/env python
# coding: utf-8

# # Post Training Quantization
# In this Colab we are going to explore Post Training Quantization (PTQ) in more detail. In particular we will use Python to get a sense of what is going on during quantization (effectively peeking under the hood of TensorFlow). We will also visualize the weight distributions to gain intuition for why quantization is often so successful (hint: the weights are often closely clustered around 0).

# ### First import the needed packages


# using TF2.x with Keras 2.x see https://keras.io/getting_started/ and https://github.com/tensorflow/tensorflow/issues/63849
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"


# For Numpy
import matplotlib.pyplot as plt
import numpy as np
import pprint
import re
import sys
# For TensorFlow Lite (also uses some of the above)
import logging
logging.getLogger("tensorflow").setLevel(logging.DEBUG)
import tensorflow as tf
from tensorflow import keras
import pathlib



# ### Exploring Post Training Quantization Algorithms in Python
#
# Let us assume we have a weight array of size (256, 256).

weights = np.random.randn(256, 256)


# In Post Training Quantization, we map the 32-bit floating point numbers to 8-bit integers.
# To do this, we need to find a very important value, the scale. The scale value is used to
# convert numbers back and forth between the various representations. For example,  32-bit floating
# point numbers can be contructed from 8-bit Integers by the following formula:
#
# $ FP32_Reconstructed_Value = Scale * Int_value

# To make sure we can cover the complete weight distribution, the scale value needs to take into account
# the full range of weight values which we can compute using the following formula. The denominator is 256
# because that is the range of values that can be represented using 8-bits (2^8 = 256).
#
# scale = frac{max(weights) - min(weights)}{256}
#
# Now lets code this up!
#
# We can then use this function to quantize our weights and then reconstruct them back to floating point format.
# We can then see what kinds of errors are introduced by this process. Our hope is that the errors in general are
# small showing that this process does a good job representing our weights in a more compact format. **In general
# if our scale is smaller it is more likely to have smaller errors as we are not lumping as many numbers into the
# same bin.**



def quantizeAndReconstruct(weights):
    """
    @param W: np.ndarray

    This function computes the scale value to map fp32 values to int8. The function returns a weight matrix in fp32,
    that is representable using 8-bits.
    """
    # Compute the range of the weight.
    max_weight = np.max(weights)
    min_weight = np.min(weights)
    range = max_weight - min_weight

    max_int8 = 2**8

    # Compute the scale
    scale = range / max_int8

    # Compute the midpoint
    midpoint = np.mean([max_weight, min_weight])

    # Next, we need to map the real fp32 values to the integers. For this, we make use of the computed scale.
    # By dividing the weight matrix with the scale, the weight matrix has a range between (-128, 127). Now, we can
    # simply round the full precision numbers to the closest integers.
    centered_weights = weights - midpoint
    quantized_weights = np.rint(centered_weights / scale)

    # Now, we can reconstruct the values back to fp32.
    reconstructed_weights = scale * quantized_weights + midpoint
    return reconstructed_weights




reconstructed_weights = quantizeAndReconstruct(weights)
print("Original weight matrix\n", weights)
print("Weight Matrix after reconstruction\n", reconstructed_weights)
errors = reconstructed_weights-weights
max_error = np.max(errors)
print("Max Error  : ", max_error)
print("reconstructed weights shape: ", reconstructed_weights.shape)

# QAT - quantization aware training - the quantization happens during the training
# PTQ - post training quantization - like it sounds, quantization happens after training the model is complete


# The quantized representation should not have more than 256 unique floating numbers, lets do a sanity check.
# We can use np.unique to check the number of unique floating point numbers in the weight matrix.
print("reconstructed weights shape [unique]: ", np.unique(quantizeAndReconstruct(weights)).shape)


# ### Exploring Post Training Quantization using TFLite

# Now that we know how PTQ works under the hood, lets move over to seeing the actual benefits in terms of memory and speed.
# Since in numpy, we were representing our final weight matrix in full precision, the memory occupied was still the same.
# However, in TFLite, we only store the matrix in an 8-bit format. As you have seen in previous Colabs, this can lead to a
# decrease in size of the model by a factor of up to 4!
#
# Note: We however do not save a perfect factor of 4 in total memory usage as we now also have to store the scale
# (and potentially other factors needed to properly convert the numbers).
#
# Lets explore this again looking at the file sizes of the MNIST model using the [TFLite Converter]
# (https://www.tensorflow.org/lite/convert)


# Load MNIST dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the input image so that each pixel value is between 0 to 1.
train_images = train_images / 255.0
test_images = test_images / 255.0



# Define the model architecture
model = keras.Sequential([keras.layers.InputLayer(input_shape=(28, 28)),
                          keras.layers.Reshape(target_shape=(28, 28, 1)),
                          keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.relu),
                          keras.layers.MaxPooling2D(pool_size=(2, 2)),
                          keras.layers.Flatten(),
                          keras.layers.Dense(10)])



# Train the digit classification model.  Adam is used instead of more common stochastic gradient descent
# to update network weights.  Adam extends SGD, maintaining a single learning rate.  see
# https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images,
          train_labels,
          epochs=1,
          validation_data=(test_images, test_labels))



# convert the model to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

tflite_models_dir = pathlib.Path("/files/pico/ML/atlantis-example/content/mnist_tflite_models/")
tflite_models_dir.mkdir(exist_ok=True, parents=True)
tflite_model_file = tflite_models_dir/"mnist_model.tflite"
nb1 = tflite_model_file.write_bytes(tflite_model)




# Convert the model using DEFAULT optimizations:
# https://github.com/tensorflow/tensorflow/blob/v2.4.1/tensorflow/lite/python/lite.py#L91-L130
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()
tflite_model_quant_file = tflite_models_dir / "mnist_model_quant.tflite"
nb2 = tflite_model_quant_file.write_bytes(tflite_quant_model)


# (venv) mike@t430sDebianBackup:/files/pico/ML/atlantis-example$ tree content/
# content/
# ├── mnist_tflite_models
# │   ├── mnist_model_quant.tflite
# │   └── mnist_model.tflite

# sizes 438420 vs 114024
print("sizes {} vs {}".format(nb1, nb2))


# **Notice the size difference - the quantized model is smaller by a factor of ~4 as expected**

# ### Software Installation to Inspect TFLite Models
#
# Before we can inspect TF Lite files in detail we need to build and install software to read the file format.
# First we’ll build and install the Flatbuffer compiler, which takes in a schema definition and outputs Python files to read files with that format.
#
# **Note: This will take a few minutes to run.**



# get_ipython().run_cell_magic('bash', '', '\n'
# 'cd /content/\ngit clone https://github.com/google/flatbuffers\ncd flatbuffers\ngit checkout 0dba63909fb2959994fec11c704c5d5ea45e8d83\ncmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release\nmake\ncp flatc /usr/local/bin/\n',
# 'cd /content/\ngit clone --depth 1 https://github.com/tensorflow/tensorflow\nflatc --python --gen-object-api tensorflow/tensorflow/lite/schema/schema_v3.fbs\n'
# 'pip install flatbuffers\n')

# mike@t430sDebianBackup:/files/pico/ML/flatbuffers$  git clone https://github.com/google/flatbuffers
# mike@t430sDebianBackup:/files/pico/ML/flatbuffers$  cd flatbuffers/
# mike@t430sDebianBackup:/files/pico/ML/flatbuffers$  cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release
# mike@t430sDebianBackup:/files/pico/ML/flatbuffers$  sudo cp flatc /usr/local/bin/
#
#     flatc:
#       FILEs may be schemas (must end in .fbs), binary schemas (must end in .bfbs) or
#       JSON files (conforming to preceding schema). BINARY_FILEs after the -- must be
#       binary flatbuffer format files. Output files are named using the base file name
#       of the input, and written to the current directory or the path given by -o.
#       example: flatc -c -b schema1.fbs schema2.fbs data.json
#
# (venv) mike@t430sDebianBackup:/files/pico/ML$ flatc --python --gen-object-api tensorflow/tensorflow/lite/schema/schema_v3.fbs
# (venv) mike@t430sDebianBackup:/files/pico/ML$ env |grep VIRT
# VIRTUAL_ENV=/media/mike/MMBKUPDRV/pico/ML/venv
# VIRTUAL_ENV_PROMPT=(venv)
# (venv) mike@t430sDebianBackup:/files/pico/ML$ python -m pip install flatbuffers
#
# flatc creates output in /files/pico/ML/tflite
#     (venv) mike@t430sDebianBackup:/files/pico/ML$ ls -lt tflite
#     total 200
#     -rw-r--r-- 1 mike mike  3956 Apr  5 07:30 Buffer.py
#     -rw-r--r-- 1 mike mike     0 Apr  5 07:30 __init__.py
#     -rw-r--r-- 1 mike mike  9887 Apr  5 07:30 Model.py






# To allow us to import the Python files we've just generated we need to update the path env variable
sys.path.append("/files/pico/ML/tflite")
import tflite
import Model


# Then we define some utility functions that will help us convert the model into a dictionary that's easy to work with in Python.



def CamelCaseToSnakeCase(camel_case_input):
    """Converts an identifier in CamelCase to snake_case."""
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", camel_case_input)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

def FlatbufferToDict(fb, attribute_name=None):
    """Converts a hierarchy of FB objects into a nested dict."""
    if hasattr(fb, "__dict__"):
        result = {}
        for attribute_name in dir(fb):
            attribute = fb.__getattribute__(attribute_name)
            if not callable(attribute) and attribute_name[0] != "_":
                snake_name = CamelCaseToSnakeCase(attribute_name)
                result[snake_name] = FlatbufferToDict(attribute, snake_name)
        return result
    elif isinstance(fb, str):
        return fb
    elif attribute_name == "name" and fb is not None:
        result = ""
        for entry in fb:
            result += chr(FlatbufferToDict(entry))
        return result
    elif hasattr(fb, "__len__"):
        result = []
        for entry in fb:
            result.append(FlatbufferToDict(entry))
        return result
    else:
        return fb

def CreateDictFromFlatbuffer(buffer_data):
    model_obj = Model.Model.GetRootAsModel(buffer_data, 0)
    model = Model.ModelT.InitFromObj(model_obj)
    return FlatbufferToDict(model)


# ### Visualizing TFLite model weight distributions
#
# This example uses the Inception v3 model, dating back to 2015, but you can replace it with your own file by updating the variables. To load in any TFLite model.

# MODEL_ARCHIVE_NAME = 'inception_v3_2015_2017_11_10.zip'
# MODEL_ARCHIVE_URL = 'https://storage.googleapis.com/download.tensorflow.org/models/tflite/' + MODEL_ARCHIVE_NAME
# MODEL_FILE_NAME = 'inceptionv3_non_slim_2015.tflite'
# get_ipython().system('curl -o {MODEL_ARCHIVE_NAME} {MODEL_ARCHIVE_URL}')
# get_ipython().system('unzip {MODEL_ARCHIVE_NAME}')

# (venv) mike@t430sDebianBackup:/files/pico/ML/atlantis-example$ curl -o inception_v3_2015_2017_11_10.zip https://storage.googleapis.com/download.tensorflow.org/models/tflite/inception_v3_2015_2017_11_10.zip
# (venv) mike@t430sDebianBackup:/files/pico/ML/atlantis-example/tflite-models$ unzip ../inception_v3_2015_2017_11_10.zip
# Archive:  ../inception_v3_2015_2017_11_10.zip
#   inflating: inceptionv3_non_slim_2015.tflite
#   inflating: imagenet_2015_label_strings.txt
MODEL_FILE_NAME = "/files/pico/ML/atlantis-example/tflite-models/inceptionv3_non_slim_2015.tflite"

with open(MODEL_FILE_NAME, 'rb') as file:
    model_data = file.read()


# Once we have the raw bytes of the file, we need to convert them into an understandable form. The utility functions and Python schema code
# we generated earlier will help us create a dictionary holding the file contents in a structured form.
#
# **Note: since it's a large file, this will take several minutes to run.**

model_dict = CreateDictFromFlatbuffer(model_data)



# Now that we have the model file in a dictionary, we can examine its contents using standard Python commands. In this case we're interested in
# examining the tensors (arrays of values) in the first subgraph, so we're printing them out.

pprint.pprint(model_dict['subgraphs'][0]['tensors'])


# Let's inspect the weight parameters of a typical convolution layer, so looking at the output above we can see that
# the tensor with the name 'Conv2D' has a buffer index of 212. This index points to where the raw bytes for the trained weights are stored.
# From the tensor properties I can see its type is '0', which [corresponds to a type of float32]
# (https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/schema/schema_v3.fbs#L30).
#
# This means we have to cast the bytes into a numpy array using the frombuffer() function.

param_bytes = bytearray(model_dict['buffers'][212]['data'])
params = np.frombuffer(param_bytes, dtype=np.float32)


# With the weights loaded into a numpy array, we can now use all the standard functionality to analyze them. To start, let's
# print out the minimum and maximum values to understand the range.

print("params min: ", params.min())
print("params max: ", params.max())



# This gives us the total range of the weight values, but how are those parameters distributed across that range?

plt.figure(figsize=(8,8))
plt.hist(params, 100)
plt.show()




# **This shows a distribution that's heavily concentrated around zero. This explains why quantization can work quite well.
# With values so concentrated around zero, our scale can be quite small and therefore it is much easier to do an accurate
# reconstruction as we do not need to represent a large number of values!**



# ### More Models to Explore

# Text Classification
# get_ipython().system('wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/text_classification/text_classification_v2.tflite')

# Post Estimation
# get_ipython().system('wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite')



TEXT_CLASSIFICATION_MODEL_FILE_NAME = "tflite-models/text_classification_v2.tflite"
POSE_ESTIMATION_MODEL_FILE_NAME = "tflite-models/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite"

with open(TEXT_CLASSIFICATION_MODEL_FILE_NAME, 'rb') as file:
    text_model_data = file.read()

with open(POSE_ESTIMATION_MODEL_FILE_NAME, 'rb') as file:
    pose_model_data = file.read()


def aggregate_all_weights(buffers):
    weights = []
    for i in range(len(buffers)):
        raw_data = buffers[i]['data']
        if raw_data is not None:
            param_bytes = bytearray(raw_data)
            params = np.frombuffer(param_bytes, dtype=np.float32)
            weights.extend(params.flatten().tolist())

    weights = np.asarray(weights)
    weights = weights[weights<50]
    weights = weights[weights>-50]

    return weights


# Lets plot the distribution of the Text Classification Model in log scale

model_dict_temp = CreateDictFromFlatbuffer(text_model_data)
weights = aggregate_all_weights(model_dict_temp['buffers'])

plt.figure(figsize=(8,8))
plt.hist(weights, 256, log=True)


# Lets plot the distribution of the Post Net Model in log scale

model_dict_temp = CreateDictFromFlatbuffer(pose_model_data)
weights = aggregate_all_weights(model_dict_temp['buffers'][:-1])

plt.figure(figsize=(8,8))
plt.hist(weights, 256, log=True)


# **Again we find that most model weights are closely packed around 0.**
