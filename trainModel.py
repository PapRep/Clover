import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from architectures import vgg, lenet, resnet
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def lr_schedule_svhn_cifar10(epoch, start_lr=0.001):
    lr = start_lr
    if epoch == 130:
        lr = start_lr * 0.5e-3
    elif epoch == 110:
        lr = start_lr * 1e-3
    elif epoch == 80:
        lr = start_lr * 1e-2
    elif epoch == 50:
        lr = start_lr * 1e-1
    print('Learning rate: ', lr)
    return lr


def lr_schedule_cifar100_tiny(epoch, start_lr=0.0001):
    lr = start_lr
    if epoch == 250:
        lr = start_lr * 0.5e-3
    elif epoch == 200:
        lr = start_lr * 1e-3
    elif epoch == 150:
        lr = start_lr * 1e-2
    elif epoch == 100:
        lr = start_lr * 1e-1
    print('Learning rate: ', lr)
    return lr


if __name__ == '__main__':
    # all parameters
    param_dict = {"dataset_name": ["fashion_mnist", "svhn", "cifar10", "cifar100"],
                  "model_architecture": ["lenet4", "lenet5", "resnet20", "resnet56"],
                  "num_classes": [10, 10, 10, 100],
                  "batch_size": [256, 256, 256, 256],
                  "epochs": [150, 150, 150, 300],
                  "init_lr": [0.001, 0.001, 0.001, 0.001]}

    # parameters
    param_index = 0
    dataset_name = param_dict["dataset_name"][param_index]
    model_architecture = param_dict["model_architecture"][param_index]
    num_classes = param_dict["num_classes"][param_index]
    batch_size = param_dict["batch_size"][param_index]
    epochs = param_dict["epochs"][param_index]
    init_lr = param_dict["init_lr"][param_index]

    # path
    data_architecture = dataset_name + "_" + model_architecture
    base_path = "./checkpoint/" + data_architecture + "/"
    original_data_path = base_path + "/dataset/" + dataset_name + ".npz"
    save_model_path = base_path + '/saved_models/'
    if not os.path.isdir(save_model_path):
        os.makedirs(save_model_path)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    with np.load(original_data_path) as f:
        x_train, y_train, x_test, y_test = f['x_train'], f['y_train'], f['x_test'], f['y_test']
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    if dataset_name == "fashion_mnist":
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    if dataset_name == "fashion_mnist" and model_architecture == "vgg16":
        model = vgg.vgg16(input_shape=x_train.shape[1:], num_classes=num_classes)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr_schedule_svhn_cifar10(0, init_lr)), metrics=['accuracy'])
        lr_scheduler = LearningRateScheduler(lr_schedule_svhn_cifar10)
    elif dataset_name == "svhn" and model_architecture == "lenet5":
        model = lenet.Lenet5(input_shape=x_train.shape[1:], num_classes=num_classes)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr_schedule_svhn_cifar10(0, init_lr)), metrics=['accuracy'])
        lr_scheduler = LearningRateScheduler(lr_schedule_svhn_cifar10)
    elif dataset_name == "cifar10" and model_architecture == "resnet20":
        model = resnet.resnet_v1(input_shape=x_train.shape[1:], depth=20, num_classes=num_classes)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr_schedule_svhn_cifar10(0, init_lr)), metrics=['accuracy'])
        lr_scheduler = LearningRateScheduler(lr_schedule_svhn_cifar10)
    elif dataset_name == "cifar100" and model_architecture == "resnet56":
        model = resnet.resnet_v1(input_shape=x_train.shape[1:], depth=56, num_classes=num_classes)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr_schedule_cifar100_tiny(0, init_lr)), metrics=['accuracy'])
        lr_scheduler = LearningRateScheduler(lr_schedule_cifar100_tiny)

    model_name = dataset_name + '_' + model_architecture + '_model.{epoch:03d}.h5'
    filepath = os.path.join(save_model_path, model_name)

    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=20, min_lr=0.5e-6)
    callbacks = [checkpoint, lr_reducer, lr_scheduler]

    datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), validation_data=(x_test, y_test), epochs=epochs, verbose=2,
                                  callbacks=callbacks, steps_per_epoch=x_train.shape[0] // batch_size, shuffle=True)

    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=2)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
