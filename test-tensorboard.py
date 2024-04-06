# using TF2.x with Keras 2.x see https://keras.io/getting_started/ and https://github.com/tensorflow/tensorflow/issues/63849
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"


import logging
logging.getLogger("tensorflow").setLevel(logging.DEBUG)

import tensorflow as tf
import pathlib
import datetime

# (venv) mike@t430sDebianBackup:/files/pico/ML/atlantis-example$ python -m tensorboard.main --logdir $PWD/logs
# 2024-04-05 10:19:23.811219: I external/local_tsl/tsl/cuda/cudart_stub.cc:31] Could not find cuda drivers on your machine, GPU will not be used.
# 2024-04-05 10:19:23.861601: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
# 2024-04-05 10:19:23.861645: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
# 2024-04-05 10:19:23.863033: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
# 2024-04-05 10:19:23.870133: I external/local_tsl/tsl/cuda/cudart_stub.cc:31] Could not find cuda drivers on your machine, GPU will not be used.
# 2024-04-05 10:19:25.075865: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
# 
# NOTE: Using experimental fast data loading logic. To disable, pass
#     "--load_fast=false" and report issues on GitHub. More details:
#     https://github.com/tensorflow/tensorboard/issues/4784
# 
# Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
# TensorBoard 2.15.2 at http://localhost:6006/ (Press CTRL+C to quit)





mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def create_model():
    return tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28), name='layers_flatten'),
                                       tf.keras.layers.Dense(512, activation='relu', name='layers_dense'),
                                       tf.keras.layers.Dropout(0.2, name='layers_dropout'),
                                       tf.keras.layers.Dense(10, activation='softmax', name='layers_dense_2')])


log_dir = "/files/pico/ML/atlantis-example/logs"


model = create_model()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(x=x_train, 
          y=y_train, 
          epochs=5, 
          validation_data=(x_test, y_test), 
          callbacks=[tensorboard_callback])




train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
train_dataset = train_dataset.shuffle(60000).batch(64)
test_dataset = test_dataset.batch(64)


loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()


# Define our metrics
train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')



# Define the training and test functions:

def train_step(model, optimizer, x_train, y_train):

    with tf.GradientTape() as tape:
        predictions = model(x_train, training=True)
        loss = loss_object(y_train, predictions)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    train_loss(loss)
    train_accuracy(y_train, predictions)

def test_step(model, x_test, y_test):
    predictions = model(x_test)
    loss = loss_object(y_test, predictions)

    test_loss(loss)
    test_accuracy(y_test, predictions)





# Define the training and test functions:

def train_step(model, optimizer, x_train, y_train):

    with tf.GradientTape() as tape:
        predictions = model(x_train, training=True)
        loss = loss_object(y_train, predictions)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    train_loss(loss)
    train_accuracy(y_train, predictions)

def test_step(model, x_test, y_test):
    predictions = model(x_test)
    loss = loss_object(y_test, predictions)

    test_loss(loss)
    test_accuracy(y_test, predictions)



    
    
# Set up summary writers to write the summaries to disk in a different logs directory:

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)





model = create_model() # reset our model

EPOCHS = 5

for epoch in range(EPOCHS):
    for (x_train, y_train) in train_dataset:
        train_step(model, optimizer, x_train, y_train)
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', train_loss.result(), step=epoch)
        tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

    for (x_test, y_test) in test_dataset:
        test_step(model, x_test, y_test)
    with test_summary_writer.as_default():
        tf.summary.scalar('loss', test_loss.result(), step=epoch)
        tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print (template.format(epoch+1,
                         train_loss.result(), 
                         train_accuracy.result()*100,
                         test_loss.result(), 
                         test_accuracy.result()*100))

    # Reset metrics every epoch
    train_loss.reset_states()
    test_loss.reset_states()
    train_accuracy.reset_states()
    test_accuracy.reset_states()



# see https://stackoverflow.com/questions/42158694/how-to-run-tensorboard-from-python-scipt-in-virtualenv
# 
# def launchTensorBoard():
#     import os
#     os.system('tensorboard --logdir=' + tensorBoardPath)
#     return
# 
# import threading
# t = threading.Thread(target=launchTensorBoard, args=([]))
# t.start()
#
# -- or --
#
# if __name__ == "__main__":
#     tb = program.TensorBoard()
#     tb.configure(argv=[None, '--logdir', tracking_address])
#     url = tb.launch()
#     print(f"Tensorflow listening on {url}")