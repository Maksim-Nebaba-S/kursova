import tensorflow as tf
from keras.layers import Conv2D, DepthwiseConv2D, ReLU, BatchNormalization, Add, GlobalAveragePooling2D, Dense, Input, Dropout
from keras.models import Model


class ReluActiovationModel:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def conv_block(self, inputs, filters, kernel_size, strides):
        x = Conv2D(filters, kernel_size, padding='same', strides=strides)(inputs)
        x = BatchNormalization()(x)
        x = ReLU(6.)(x)
        return x
    
    def depthwise_conv_block(self, inputs, filters, strides):
        x = DepthwiseConv2D(kernel_size=3, padding='same', strides=strides)(inputs)
        x = BatchNormalization()(x)
        x = ReLU(6.)(x)
        x = Conv2D(filters, kernel_size=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU(6.)(x)
        return x    

    def inverted_bottleneck_block(self, inputs, in_filters, out_filters, expansion_factor, strides):
        expanded_filters = in_filters * expansion_factor

        x = self.conv_block(inputs, expanded_filters, kernel_size=1, strides=1)
        x = self.depthwise_conv_block(x, expanded_filters, strides=strides)
        if strides == 1 and in_filters == out_filters:
            x = Add()([inputs, x])

        return x

    def build_model(self):
        input_tensor = Input(shape=self.input_shape)

    
        x = self.conv_block(input_tensor, filters=32, kernel_size=(3, 3), strides=(2, 2))
        

        x = self.inverted_bottleneck_block(x, in_filters=32, out_filters=16, expansion_factor=1, strides=1)
        x = self.inverted_bottleneck_block(x, in_filters=16, out_filters=24, expansion_factor=6, strides=2)

        x = self.inverted_bottleneck_block(x, in_filters=24, out_filters=24, expansion_factor=6, strides=1)
        x = self.inverted_bottleneck_block(x, in_filters=24, out_filters=32, expansion_factor=6, strides=2)

        x = self.inverted_bottleneck_block(x, in_filters=32, out_filters=32, expansion_factor=6, strides=1)
        x = self.inverted_bottleneck_block(x, in_filters=32, out_filters=64, expansion_factor=6, strides=2)

        x = self.inverted_bottleneck_block(x, in_filters=64, out_filters=64, expansion_factor=6, strides=1)
        x = self.inverted_bottleneck_block(x, in_filters=64, out_filters=96, expansion_factor=6, strides=1)

        x = self.inverted_bottleneck_block(x, in_filters=96, out_filters=96, expansion_factor=6, strides=1)
        x = self.inverted_bottleneck_block(x, in_filters=96, out_filters=160, expansion_factor=6, strides=2)

        x = self.inverted_bottleneck_block(x, in_filters=160, out_filters=160, expansion_factor=6, strides=1)
        x = self.inverted_bottleneck_block(x, in_filters=160, out_filters=320, expansion_factor=6, strides=1)

 
        x = Conv2D(1280, kernel_size=1, padding='same', strides=1)(x)
        x = BatchNormalization()(x)
        x = ReLU(6.)(x)
        x = Dropout(0.3)(x)
        
        x = GlobalAveragePooling2D()(x)
        output = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=input_tensor, outputs=output)
        return model



class ReluActiovationModel_64x64:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def conv_block(self, inputs, filters, kernel_size, strides):
        x = Conv2D(filters, kernel_size, padding='same', strides=strides)(inputs)
        x = BatchNormalization()(x)
        x = ReLU(6.)(x)
        return x
    
    def depthwise_conv_block(self, inputs, filters, strides):
        x = DepthwiseConv2D(kernel_size=3, padding='same', strides=strides)(inputs)
        x = BatchNormalization()(x)
        x = ReLU(6.)(x)
        x = Conv2D(filters, kernel_size=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU(6.)(x)
        return x    

    def inverted_bottleneck_block(self, inputs, in_filters, out_filters, expansion_factor, strides):
        expanded_filters = in_filters * expansion_factor

        x = self.conv_block(inputs, expanded_filters, kernel_size=1, strides=1)
        x = self.depthwise_conv_block(x, expanded_filters, strides=strides)
        if strides == 1 and in_filters == out_filters:
            x = Add()([inputs, x])

        return x

    def build_model(self):
        input_tensor = Input(shape=self.input_shape)


        x = self.conv_block(input_tensor, filters=32, kernel_size=(3, 3), strides=(2, 2))
      
        x = self.inverted_bottleneck_block(x, in_filters=32, out_filters=64, expansion_factor=3, strides=2)
        x = self.inverted_bottleneck_block(x, in_filters=64, out_filters=64, expansion_factor=3, strides=1)
 
        x = self.inverted_bottleneck_block(x, in_filters=64, out_filters=128, expansion_factor=6, strides=2)
        x = self.inverted_bottleneck_block(x, in_filters=128, out_filters=128, expansion_factor=6, strides=1)

        x = self.inverted_bottleneck_block(x, in_filters=128, out_filters=256, expansion_factor=6, strides=2)
        x = self.inverted_bottleneck_block(x, in_filters=256, out_filters=256, expansion_factor=6, strides=1)
 
        x = Conv2D(640, kernel_size=1, padding='same', strides=1)(x)
        x = BatchNormalization()(x)
        x = ReLU(6.)(x)
        x = Dropout(0.4)(x)

        x = GlobalAveragePooling2D()(x)
        output = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=input_tensor, outputs=output)
        return model
    

