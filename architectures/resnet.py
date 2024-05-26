from tensorflow import keras
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten, Dense, Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D
from tensorflow.keras import Sequential
from tensorflow.keras.applications.resnet import ResNet50, ResNet101


def resnet_layer(inputs, num_filters=16, kernel_size=3, strides=1, activation='relu', batch_normalization=True, conv_first=True):
    conv = Conv2D(num_filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))
    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


# ResNet20
# [2 'conv2d', 'batch_normalization', 'activation',
# 5 'conv2d_1', 'batch_normalization_1', 'activation_1',
# 8 'conv2d_2', 'batch_normalization_2', 'add', 'activation_2',
# 12 'conv2d_3', 'batch_normalization_3', 'activation_3',
# 15 'conv2d_4', 'batch_normalization_4', 'add_1', 'activation_4',
# 19 'conv2d_5', 'batch_normalization_5', 'activation_5',
# 22 'conv2d_6', 'batch_normalization_6', 'add_2', 'activation_6',
# 26 'conv2d_7', 'batch_normalization_7', 'activation_7',
# 29 'conv2d_8', 30 'conv2d_9', 'batch_normalization_8', 'add_3', 'activation_8',
# 34 'conv2d_10', 'batch_normalization_9', 'activation_9',
# 37 'conv2d_11', 'batch_normalization_10', 'add_4', 'activation_10',
# 41 'conv2d_12', 'batch_normalization_11', 'activation_11',
# 44 'conv2d_13', 'batch_normalization_12', 'add_5', 'activation_12',
# 48 'conv2d_14', 'batch_normalization_13', 'activation_13',
# 51 'conv2d_15', 52 'conv2d_16', 'batch_normalization_14', 'add_6', 'activation_14',
# 56 'conv2d_17', 'batch_normalization_15', 'activation_15',
# 59 'conv2d_18', 'batch_normalization_16', 'add_7', 'activation_16',
# 63 'conv2d_19', 'batch_normalization_17', 'activation_17',
# 66 'conv2d_20', 'batch_normalization_18', 'add_8', 'activation_18',
# 70 'average_pooling2d',
# 71 'dense']

# Resnet56
# [2 'conv2d', 'batch_normalization', 'activation',
# 5 'conv2d_1', 'batch_normalization_1', 'activation_1',
# 8 'conv2d_2', 'batch_normalization_2', 'add', 'activation_2',
# 12 'conv2d_3', 'batch_normalization_3', 'activation_3',
# 15 'conv2d_4', 'batch_normalization_4', 'add_1', 'activation_4',
# 19 'conv2d_5', 'batch_normalization_5', 'activation_5',
# 22 'conv2d_6', 'batch_normalization_6', 'add_2', 'activation_6',
# 26 'conv2d_7', 'batch_normalization_7', 'activation_7',
# 29 'conv2d_8', 'batch_normalization_8', 'add_3', 'activation_8',
# 33 'conv2d_9', 'batch_normalization_9', 'activation_9',
# 36 'conv2d_10', 'batch_normalization_10', 'add_4', 'activation_10',
# 40 'conv2d_11', 'batch_normalization_11', 'activation_11',
# 43 'conv2d_12', 'batch_normalization_12', 'add_5', 'activation_12',
# 47 'conv2d_13', 'batch_normalization_13', 'activation_13',
# 50 'conv2d_14', 'batch_normalization_14', 'add_6', 'activation_14',
# 54 'conv2d_15', 'batch_normalization_15', 'activation_15',
# 57 'conv2d_16', 'batch_normalization_16', 'add_7', 'activation_16',
# 61 'conv2d_17', 'batch_normalization_17', 'activation_17',
# 64 'conv2d_18', 'batch_normalization_18', 'add_8', 'activation_18',
# 68 'conv2d_19', 'batch_normalization_19', 'activation_19',
# 71 'conv2d_20', 72 'conv2d_21', 'batch_normalization_20', 'add_9', 'activation_20',
# 76 'conv2d_22', 'batch_normalization_21', 'activation_21',
# 79 'conv2d_23', 'batch_normalization_22', 'add_10', 'activation_22',
# 83 'conv2d_24', 'batch_normalization_23', 'activation_23',
# 86 'conv2d_25', 'batch_normalization_24', 'add_11', 'activation_24',
# 90 'conv2d_26', 'batch_normalization_25', 'activation_25',
# 93 'conv2d_27', 'batch_normalization_26', 'add_12', 'activation_26',
# 97 'conv2d_28', 'batch_normalization_27', 'activation_27',
# 100 'conv2d_29', 'batch_normalization_28', 'add_13', 'activation_28',
# 104 'conv2d_30', 'batch_normalization_29', 'activation_29',
# 107 'conv2d_31', 'batch_normalization_30', 'add_14', 'activation_30',
# 111 'conv2d_32', 'batch_normalization_31', 'activation_31',
# 114 'conv2d_33', 'batch_normalization_32', 'add_15', 'activation_32',
# 118 'conv2d_34', 'batch_normalization_33', 'activation_33',
# 121 'conv2d_35', 'batch_normalization_34', 'add_16', 'activation_34',
# 125 'conv2d_36', 'batch_normalization_35', 'activation_35',
# 128 'conv2d_37', 'batch_normalization_36', 'add_17', 'activation_36',
# 132 'conv2d_38', 'batch_normalization_37', 'activation_37',
# 135 'conv2d_39', 136 'conv2d_40', 'batch_normalization_38', 'add_18', 'activation_38',
# 140 'conv2d_41', 'batch_normalization_39', 'activation_39',
# 143 'conv2d_42', 'batch_normalization_40', 'add_19', 'activation_40',
# 147 'conv2d_43', 'batch_normalization_41', 'activation_41',
# 150 'conv2d_44', 'batch_normalization_42', 'add_20', 'activation_42',
# 154 'conv2d_45', 'batch_normalization_43', 'activation_43',
# 157 'conv2d_46', 'batch_normalization_44', 'add_21', 'activation_44',
# 161 'conv2d_47', 'batch_normalization_45', 'activation_45',
# 164 'conv2d_48', 'batch_normalization_46', 'add_22', 'activation_46',
# 168 'conv2d_49', 'batch_normalization_47', 'activation_47',
# 171 'conv2d_50', 'batch_normalization_48', 'add_23', 'activation_48',
# 175 'conv2d_51', 'batch_normalization_49', 'activation_49',
# 178 'conv2d_52', 'batch_normalization_50', 'add_24', 'activation_50',
# 182 'conv2d_53', 'batch_normalization_51', 'activation_51',
# 185 'conv2d_54', 'batch_normalization_52', 'add_25', 'activation_52',
# 189 'conv2d_55', 'batch_normalization_53', 'activation_53',
# 192 'conv2d_56', 'batch_normalization_54', 'add_26', 'activation_54',
# 'average_pooling2d',
# 'dense']


def resnet_v1(input_shape, depth, num_classes=10):
    """
    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """

    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x, num_filters=num_filters, strides=strides)
            y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x, num_filters=num_filters, kernel_size=1, strides=strides, activation=None, batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = keras.models.Model(inputs=inputs, outputs=outputs)
    return model


def Resnet50(input_shape, num_classes=100):
    pre_model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape, pooling=None, classes=num_classes)
    pre_model.trainable = True

    model = Sequential()
    model.add(pre_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(num_classes, activation='softmax'))
    return model


def Resnet101(input_shape, num_classes=100):
    pre_model = ResNet101(include_top=False, weights='imagenet', input_shape=input_shape, pooling=None, classes=num_classes)
    pre_model.trainable = True

    model = Sequential()
    model.add(pre_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(num_classes, activation='softmax'))
    return model