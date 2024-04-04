import tensorflow as tf
import keras

# from https://github.com/keras-team/keras/issues/19108

(train_images, train_labels), (
    test_images,
    test_labels,
) = keras.datasets.fashion_mnist.load_data()

test_model = keras.Sequential(
    [
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ]
)
test_model.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)
test_model.fit(train_images, train_labels, epochs=1)

# replace tf.saved_model.save with this line
test_model.export("test", "tf_saved_model")


# WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
# W0000 00:00:1712234239.842753 2953893 tf_tfl_flatbuffer_helpers.cc:390] Ignored output_format.
# W0000 00:00:1712234239.842779 2953893 tf_tfl_flatbuffer_helpers.cc:393] Ignored drop_control_dependency.
# 2024-04-04 08:37:19.843880: I tensorflow/cc/saved_model/reader.cc:83] Reading SavedModel from: test
# 2024-04-04 08:37:19.844822: I tensorflow/cc/saved_model/reader.cc:51] Reading meta graph with tags { serve }
# 2024-04-04 08:37:19.844854: I tensorflow/cc/saved_model/reader.cc:146] Reading SavedModel debug info (if present) from: test
# 2024-04-04 08:37:19.852722: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:388] MLIR V1 optimization pass is not enabled
# 2024-04-04 08:37:19.853586: I tensorflow/cc/saved_model/loader.cc:234] Restoring SavedModel bundle.
# 2024-04-04 08:37:19.879092: I tensorflow/cc/saved_model/loader.cc:218] Running initialization op on SavedModel bundle at path: test
# 2024-04-04 08:37:19.891308: I tensorflow/cc/saved_model/loader.cc:317] SavedModel load for tags { serve }; Status: success: OK. Took 47437 microseconds.
# 2024-04-04 08:37:19.903497: I tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc:268] disabling MLIR crash reproducer, set env var `MLIR_CRASH_REPRODUCER_DIRECTORY` to enable.
# 2024-04-04 08:37:19.941253: I tensorflow/compiler/mlir/lite/flatbuffer_export.cc:3064] Estimated count of arithmetic ops: 3.275 M  ops, equivalently 1.638 M  MACs

converter = tf.lite.TFLiteConverter.from_saved_model("test")
tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)