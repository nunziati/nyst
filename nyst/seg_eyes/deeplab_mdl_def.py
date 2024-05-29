import keras
import tensorflow as tf
from tensorflow import keras
from keras import layers

# Utilizza pkgutil qui...


class DynamicUpsample(tf.keras.layers.Layer):
    def __init__(self, method='bilinear', **kwargs):
        super().__init__(**kwargs)
        self.method = method

    def call(self, inputs, ref_tensor):
        return tf.image.resize(inputs, (tf.shape(ref_tensor)[1], tf.shape(ref_tensor)[2]), method=self.method)

def DeeplabV3Plus(num_classes, filters_conv1=24, filters_conv2=24, filters_spp=128, filters_final=128, dilated_conv_rates =[1, 4, 8, 16]):
    model_input = keras.Input(shape=(None, None, 3))
    preprocessed = keras.applications.resnet50.preprocess_input(model_input)
    resnet50 = keras.applications.ResNet50(weights="imagenet", include_top=False, input_tensor=preprocessed)

    x = resnet50.get_layer("conv4_block6_2_relu").output
    input_b = resnet50.get_layer("conv2_block3_2_relu").output

    x1 = layers.GlobalAveragePooling2D()(x)
    x1 = layers.Reshape((1, 1, x.shape[-1]))(x1)
    x1 = layers.Conv2D(filters=filters_conv1, kernel_size=1, padding="same")(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = DynamicUpsample()(x1, x)  # Custom layer for dynamic upsampling

    # Multiple dilated convolutions
    pyramids = []
    for rate in dilated_conv_rates:
        if rate == 1: #TODO testato con != 1, proviamo con == 1
            pyramid = layers.Conv2D(filters=filters_spp, kernel_size=3, dilation_rate=rate, padding="same")(x)
            pyramid = layers.BatchNormalization()(pyramid)
            pyramids.append(pyramid)
        else:
            pyramid = layers.Conv2D(filters=filters_spp, kernel_size=3 + int(rate/2), dilation_rate=rate, padding="same")(x)
            pyramid = layers.BatchNormalization()(pyramid)
            pyramids.append(pyramid)
    '''
    pyramids = [layers.Conv2D(filters=filters_spp, kernel_size=3 + 2 * (rate - 1), dilation_rate=rate, padding="same")(x) for rate in rates]
    pyramids = [layers.BatchNormalization()(p) for p in pyramids]
    '''

    x = layers.Concatenate(axis=-1)([x1] + pyramids)
    x = layers.Conv2D(filters=filters_spp, kernel_size=1, padding="same")(x)
    x = layers.BatchNormalization()(x)

    input_b = layers.Conv2D(filters=filters_conv2, kernel_size=1, padding="same")(input_b)
    input_b = layers.BatchNormalization()(input_b)

    input_a = DynamicUpsample()(x, input_b)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = layers.Conv2D(filters=filters_final, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=filters_final, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = DynamicUpsample()(x, model_input)

    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
    model = keras.Model(inputs=model_input, outputs=model_output)
    return model
