#!/usr/bin/env python
# coding: utf-8


# In[3]:


import tensorflow as tf
import numpy as np

import keras


l0 = keras.layers.Dense(units=1, input_shape=[1])
model = keras.Sequential([l0])
model.compile(optimizer='sgd', loss='mean_squared_error')

# linear regression y = 2x -1
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, ys, epochs=5)

# https://www.tensorflow.org/api_docs/python/tf/keras/Model
print(model.predict(np.array([10.0], dtype=float)))
print("Here is what I learned: {}".format(l0.get_weights()))


# mike@t430sDebianBackup:/files/pico/ML/atlantis-example$ tree saved_model/
# saved_model/
# └── 1
#     ├── assets
#     ├── fingerprint.pb
#     ├── saved_model.pb
#     └── variables
#         ├── variables.data-00000-of-00001
#         └── variables.ind

export_dir = 'saved_model/1'

# avoid python error missing attribute 'value', use model.export
# see  https://github.com/keras-team/keras/issues/19108
# prev: tf.saved_model.save(model, export_dir)
model.export(export_dir, "tf_saved_model")

# Convert the model.
converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)


# TFLite model written out just above...
#
# import pathlib
# tflite_model_file = pathlib.Path('model.tflite')
# tflite_model_file.write_bytes(tflite_model)


# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)
print(output_details)


# use TFLite model (pruned, quantized) to check accuracy, compare to TF model
to_predict = np.array([[10.0]], dtype=np.float32)
print(to_predict)
interpreter.set_tensor(input_details[0]['index'], to_predict)
interpreter.invoke()
tflite_results = interpreter.get_tensor(output_details[0]['index'])
print(tflite_results)

