import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_model_optimization as tfmot

def main():

    def test_model(final_depth = 24):
        inputs = layers.Input(shape=(None, None,24))
        input_shape = tf.shape(inputs)
        rs_1 = tf.reshape(inputs, [input_shape[0], input_shape[1] * input_shape[2], input_shape[3], 1])
        conv1 = layers.ReLU(negative_slope = 0.2)(layers.Conv2D(1, (1,3), activation = None, padding = 'same', kernel_initializer = 'glorot_uniform')(rs_1))
        out = tf.reshape(conv1, [input_shape[0], input_shape[1], input_shape[2], input_shape[3]])
        model = tf.keras.Model(inputs = inputs, outputs = out)
        return model

    model = test_model()

    def annotate(layer):
        if layer._name.startswith('tf_op_layer_Reshape'):
            return layer
        elif layer._name.startswith('tf_op_layer_Shape'):
            return layer
        elif layer._name.startswith('concatenate'):
            return layer
        elif layer._name.startswith('tf_op_layer_strided_slice'):
            return layer
        # quantize everything else
        return tfmot.quantization.keras.quantize_annotate_layer(layer)

    annotated_model = tf.keras.models.clone_model(model, clone_function=annotate)
    model = tfmot.quantization.keras.quantize_apply(annotated_model)

    model.summary()

if __name__ == "__main__":
    main()