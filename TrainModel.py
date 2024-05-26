import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from sklearn.model_selection import train_test_split
from architectures import vgg, lenet, resnet
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def lr_schedule(epoch, start_lr=0.001):
    lr = start_lr
    if epoch == 110:
        lr = start_lr * 1e-3
    elif epoch == 80:
        lr = start_lr * 1e-2
    elif epoch == 50:
        lr = start_lr * 1e-1
    print('Learning rate: ', lr)
    return lr


if __name__ == '__main__':
    # all parameters
    param_dict = {"dataset_name": ["fashion_mnist", "svhn", "cifar10", "cifar100"],
                  "model_architecture": ["vgg16", "lenet5", "resnet20", "resnet56"],
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
    base_path = "./checkpoint/" + data_architecture
    original_data_path = base_path + "/dataset/" + dataset_name + ".npz"
    train_valid_data_path = base_path + "/dataset/train_valid.npz"
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

    if not os.path.isdir(train_valid_data_path):
        with np.load(original_data_path) as f:
            x_train, y_train, x_test, y_test = f['x_train'], f['y_train'], f['x_test'], f['y_test']

        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=5000, random_state=2023)
        np.savez(train_valid_data_path, x_train=x_train, x_valid=x_valid, y_train=y_train, y_valid=y_valid)
    else:
        with np.load(train_valid_data_path) as f:
            x_train, x_valid, y_train, y_valid = f['x_train'], f['x_valid'], f['y_train'], f['y_valid']
        with np.load(original_data_path) as f:
            x_test, y_test = f['x_test'], f['y_test']

    x_train = x_train.astype('float32') / 255
    x_valid = x_valid.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    if dataset_name == "fashion_mnist":
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
        x_valid = x_valid.reshape(x_valid.shape[0], x_valid.shape[1], x_valid.shape[2], 1)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_valid = tf.keras.utils.to_categorical(y_valid, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    if dataset_name == "fashion_mnist" and model_architecture == "vgg16":
        model = vgg.vgg16(input_shape=x_train.shape[1:], num_classes=num_classes)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr_schedule(0, init_lr)), metrics=['accuracy'])
        lr_scheduler = LearningRateScheduler(lr_schedule)
    elif dataset_name == "svhn" and model_architecture == "lenet5":
        model = lenet.Lenet5(input_shape=x_train.shape[1:], num_classes=num_classes)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr_schedule(0, init_lr)), metrics=['accuracy'])
        lr_scheduler = LearningRateScheduler(lr_schedule)
    elif dataset_name == "cifar10" and model_architecture == "resnet20":
        model = resnet.resnet_v1(input_shape=x_train.shape[1:], depth=20, num_classes=num_classes)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr_schedule(0, init_lr)), metrics=['accuracy'])
        lr_scheduler = LearningRateScheduler(lr_schedule)
    elif dataset_name == "cifar100" and model_architecture == "resnet56":
        model = resnet.resnet_v1(input_shape=x_train.shape[1:], depth=56, num_classes=num_classes)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr_schedule(0, init_lr)), metrics=['accuracy'])
        lr_scheduler = LearningRateScheduler(lr_schedule)

    filepath = os.path.join(save_model_path, data_architecture + '_model.{epoch:03d}.h5')
    checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
    lr_reducer = ReduceLROnPlateau(factor=0.1, cooldown=0, patience=20, min_lr=0.5e-6)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)
    callbacks = [checkpoint, lr_reducer, lr_scheduler, early_stopping]

    datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    datagen.fit(x_train)
    history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), validation_data=(x_valid, y_valid), epochs=epochs, verbose=2,
                                  callbacks=callbacks, steps_per_epoch=x_train.shape[0] // batch_size, shuffle=True)

    scores_train = model.evaluate(x_train, y_train, verbose=2)
    print(f"Training loss: {scores_train[0]}, Training accuracy: {scores_train[1]}")

    scores_valid = model.evaluate(x_valid, y_valid, verbose=2)
    print(f"Validation loss: {scores_valid[0]}, Validation accuracy: {scores_valid[1]}, Train-Valid: {scores_train[1] - scores_valid[1]}")

    scores_test = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test loss: {scores_test[0]}, Test accuracy: {scores_test[1]}, Train-Test: {scores_train[1] - scores_test[1]}")
