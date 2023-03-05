import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from retrain import existing_selection, context_selection
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def lr_schedule_retrain(epoch):
    # lr = 1e-3         # lenet5 and svhn
    lr = 1e-4
    print('Learning rate: ', lr)
    return lr


if __name__ == '__main__':
    model_architecture = "efficientB7"    # [vgg16, lenet5, efficientB7, resnet20, resnet56]
    dataset_name = "svhn"  # [fashion_mnist, svhn, svhn, cifar10, cifar100]
    fuzz_type = "Context_Fuzzing"       # [FOL_Fuzzing, Context_Fuzzing]
    dataset_prefix = 'Context_Fuzz_'    # [FOL_Fuzz_, Context_Fuzz_]
    strategy = 'mean_confidence'               # [kmst, mean_confidence]
    metric_name = 'mocs'            # [fols, mocs]

    print(f"Model architecture: {model_architecture}, dataset: {dataset_name}, fuzz_type: {fuzz_type}")

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    if model_architecture == "vgg16" and dataset_name == "fashion_mnist":
        num_classes = 10
        train_data_path = './data/fashion_mnist/fashion_mnist.npz'
        ae_data_path = './checkpoint/fashion_mnist_vgg16/fuzzing/' + str(fuzz_type)
        original_model_path = './checkpoint/fashion_mnist_vgg16/saved_models/fashion_mnist_vgg19_model.064.h5'
        # lr = 1e-4

    elif model_architecture == "lenet5" and dataset_name == "svhn":
        num_classes = 10
        train_data_path = './data/svhn/svhn.npz'
        ae_data_path = './checkpoint/svhn_lenet5/fuzzing/' + str(fuzz_type)
        original_model_path = './checkpoint/svhn_lenet5/saved_models/svhn_lenet5_model.053.h5'
        # lr = 1e-4

    elif model_architecture == "efficientB7" and dataset_name == "svhn":
        num_classes = 10
        train_data_path = './data/svhn/svhn.npz'
        ae_data_path = './checkpoint/svhn_efficientB7/fuzzing/' + str(fuzz_type)
        original_model_path = './checkpoint/svhn_efficientB7/saved_models/svhn_efficientB7_model.078.h5'
        # lr = 1e-4

    elif model_architecture == "resnet20" and dataset_name == "cifar10":
        num_classes = 10
        train_data_path = './data/cifar10/cifar10.npz'
        ae_data_path = './checkpoint/cifar10_resnet20/fuzzing/' + str(fuzz_type)
        original_model_path = './checkpoint/cifar10_resnet20/saved_models/cifar10_resnet20_model.120.h5'
        # lr = 1e-4

    elif model_architecture == "resnet56" and dataset_name == "cifar100":
        num_classes = 100
        train_data_path = './data/cifar100/cifar100.npz'
        ae_data_path = './checkpoint/cifar100_resnet56/fuzzing/' + str(fuzz_type)
        original_model_path = './checkpoint/cifar100_resnet56/saved_models/cifar100_resnet56_model.086.h5'
        # lr = 1e-4

    with np.load(train_data_path) as f:
        x_train, y_train, x_test, y_test = f['x_train'], f['y_train'], f['x_test'], f['y_test']
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    if model_architecture == "vgg16" and dataset_name == "fashion_mnist":
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # Load the generated adversarial inputs for testing. FGSM and PGD.
    with np.load(ae_data_path + "/FGSM_Test.npz") as f:
        fgsm_test, fgsm_test_labels, fgsm_test_fols, fgsm_test_ginis, fgsm_test_mean_confidence, fgsm_test_mean_l2_dist, fgsm_test_mean_ce = \
            f['advs'], f['labels'], f['fols'], f['ginis'], f['mean_confidence'], f['mean_l2_dist'], f['mean_ce']

    with np.load(ae_data_path + "/PGD_Test.npz") as f:
        pgd_test, pgd_test_labels, pgd_test_fols, pgd_test_ginis, pgd_test_mean_confidence, pgd_test_mean_l2_dist, pgd_test_mean_ce = \
            f['advs'], f['labels'], f['fols'], f['ginis'], f['mean_confidence'], f['mean_l2_dist'], f['mean_ce']

    fp_test = np.concatenate((fgsm_test, pgd_test))
    fp_test_labels = np.concatenate((fgsm_test_labels, pgd_test_labels))

    datasetNums = [50000]
    for dNums in datasetNums:
        # Load the generated adversarial examples generated from fuzzing techniques for training.
        with np.load(ae_data_path + '/' + dataset_prefix + str(dNums) + '.npz') as f:
            fuzz_train, fuzz_labels, fuzz_metric = f['advs'], f['labels'], f[metric_name]

        print(len(np.where(fuzz_metric > 0.8)[0]))
        print(len(np.where(fuzz_metric > 0.6)[0]) - len(np.where(fuzz_metric > 0.8)[0]))
        print(len(np.where(fuzz_metric > 0.4)[0]) - len(np.where(fuzz_metric > 0.6)[0]))
        print(len(np.where(fuzz_metric > 0.2)[0]) - len(np.where(fuzz_metric > 0.4)[0]))
        print(len(np.where(fuzz_metric > 0.0)[0]) - len(np.where(fuzz_metric > 0.2)[0]))

        fuzz_train = np.array([np.array(item).reshape(x_train.shape[1], x_train.shape[2], x_train.shape[3]) for item in fuzz_train])
        sNums = [5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]

        for num in sNums:
            print(num, round(num/len(fuzz_train), 2))

            model_path = "./checkpoint/" + dataset_name + "_" + model_architecture + "/fuzzing/" + fuzz_type + "/best_Resnet_MIX_%d_%d_%s.h5" % (dNums, num, strategy)
            checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_accuracy', verbose=1, save_best_only=True)
            lr_scheduler = LearningRateScheduler(lr_schedule_retrain)
            lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
            callbacks = [checkpoint, lr_reducer, lr_scheduler]

            if strategy == 'random':
                indexes = existing_selection(fuzz_train, num, select_metric=strategy)
            elif strategy == 'gini':
                indexes = existing_selection(fuzz_train, num, select_metric=strategy)
            elif strategy == 'best' or strategy == 'kmst':
                indexes = existing_selection(fuzz_train, num, select_metric=strategy)
            elif strategy == "mean_confidence" or strategy == "mean_l2_dist" or strategy == "mean_ce":
                indexes = context_selection(data=fuzz_metric, x_idxs=None, select_num=num)
            else:
                indexes = []

            print(f"The # of selected test cases {len(indexes)}")
            selectAdvs = fuzz_train[indexes]
            selectAdvsLabels = fuzz_labels[indexes]

            x_train_mix = np.concatenate((x_train, selectAdvs), axis=0)
            y_train_mix = np.concatenate((y_train, selectAdvsLabels), axis=0)

            # load old model
            model = keras.models.load_model(original_model_path)
            datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
            datagen.fit(x_train_mix)
            batch_size = 64
            history = model.fit_generator(datagen.flow(x_train_mix, y_train_mix, batch_size=batch_size), validation_data=(fp_test, fp_test_labels), epochs=40, verbose=2,
                                          callbacks=callbacks, steps_per_epoch=x_train_mix.shape[0] // batch_size)

            scores = model.evaluate(x_test, y_test, verbose=2)
            print(f'Standard accuracy with {num} adversarial examples: ', scores[1])
