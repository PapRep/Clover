from tensorflow import keras


def Lenet4(input_shape, num_classes=10):
    # ['block1_conv1', 'block1_pool1', 'block2_conv1', 'block2_pool1', 'fc', 'before_softmax', 'predictions']
    input_tensor = keras.layers.Input(shape=input_shape)  # input_shape=(32, 32, 1)

    x = keras.layers.Convolution2D(4, (5, 5), activation='relu', padding='same', name='block1_conv1')(input_tensor)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2), name='block1_pool1')(x)

    x = keras.layers.Convolution2D(16, (5, 5), activation='relu', padding='same', name='block2_conv1')(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2), name='block2_pool1')(x)

    x = keras.layers.Flatten(name='flatten')(x)
    x = keras.layers.Dense(120, activation='relu', name='fc')(x)
    x = keras.layers.Dense(num_classes, name='before_softmax')(x)
    x = keras.layers.Activation('softmax', name='predictions')(x)

    return keras.models.Model(input_tensor, x)


def Lenet5(input_shape, num_classes=10):
    # ['block1_conv1', 'block1_pool1', 'block2_conv1', 'block2_pool1', 'fc1', 'fc2', 'before_softmax', 'predictions']
    input_tensor = keras.layers.Input(shape=input_shape)  # input_shape=(32, 32, 1)

    x = keras.layers.Convolution2D(6, (5, 5), activation='relu', padding='same', name='block1_conv1')(input_tensor)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2), name='block1_pool1')(x)

    x = keras.layers.Convolution2D(16, (5, 5), activation='relu', padding='same', name='block2_conv1')(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2), name='block2_pool1')(x)

    x = keras.layers.Flatten(name='flatten')(x)
    x = keras.layers.Dense(120, activation='relu', name='fc1')(x)
    x = keras.layers.Dense(84, activation='relu', name='fc2')(x)
    x = keras.layers.Dense(num_classes, name='before_softmax')(x)
    x = keras.layers.Activation('softmax', name='predictions')(x)

    return keras.models.Model(input_tensor, x)


