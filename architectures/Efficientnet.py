import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras import Sequential


def EfficientNetB7(input_shape, num_classes=10):
    pre_model = tf.keras.applications.EfficientNetB7(include_top=False, weights=None, input_shape=input_shape)
    pre_model.trainable = True
    model = Sequential()
    model.add(pre_model)
    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units=num_classes, activation="softmax"))
    return model

