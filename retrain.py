import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def lr_schedule_retrain(epoch):
    lr = 1e-4
    # lr = 1e-3
    print('Learning rate: ', lr)
    return lr


def existing_selection(data, select_num, select_metric='best', k=4):
    """
    data: the data used for ranking
    select_num: the number of selected test cases.
    select_metric: strategy, ['best', 'random', 'kmst', 'gini']
    k: for KM-ST, the number of ranges.
    """
    ranks = np.argsort(data)
    if select_metric == 'random':
        return np.array(random.sample(list(ranks), min(select_num, len(data))))
    elif select_metric == 'gini':
        return ranks[::-1][:select_num]
    elif select_metric == 'best':
        h = select_num // 2
        return np.concatenate((ranks[:h], ranks[-h:]))
    elif select_metric == 'kmst':
        fol_max = data.max()
        th = fol_max / k
        section_nums = select_num // k
        indexes_list = []
        for i in range(k):
            section_indexes = np.intersect1d(np.where(data < th * (i + 1)), np.where(data >= th * i))
            if section_nums < len(section_indexes):
                index = random.sample(list(section_indexes), section_nums)
                indexes_list.append(index)
            else:
                indexes_list.append(section_indexes)
                index = random.sample(list(ranks), section_nums - len(section_indexes))
                indexes_list.append(index)
        return np.concatenate(np.array(indexes_list))


def context_selection(data, x_idxs, select_num):
    """
    data: the data used for ranking
    select_num: the number of selected test cases
    y_ae: the prediction label of aes
    """
    rank = np.argsort(data)
    if x_idxs is None:
        selected_index_array = rank[0:select_num]
    else:
        sorted_x_idxs = x_idxs[rank]
        sorted_index_array = np.array([], dtype=int)
        all_sections = [list(rank[np.where(sorted_x_idxs == idx)[0]]) for idx in range(max(sorted_x_idxs)+1)]
        while len(sorted_index_array) != len(data):
            cur_index_array = np.array([section.pop(0) for section in all_sections if len(section) > 0])
            cur_index_array_sorted = cur_index_array[np.argsort(data[cur_index_array])]
            sorted_index_array = np.concatenate((sorted_index_array, cur_index_array_sorted))
        selected_index_array = sorted_index_array[0:select_num]
    return selected_index_array


if __name__ == '__main__':
    # all parameters
    param_dict = {"dataset_name": ["fashion_mnist", "svhn", "cifar10", "cifar100"],
                  "model_architecture": ["vgg16", "lenet5", "resnet20", "resnet56"],
                  "num_classes": [10, 10, 10, 100]}

    # parameters
    param_index = 2
    dataset_name = param_dict["dataset_name"][param_index]
    model_architecture = param_dict["model_architecture"][param_index]
    num_classes = param_dict["num_classes"][param_index]
    ae_generation_technique = "attack"      # ["attack", "adapt", "folfuzz", "contextfuzz"]

    # path
    data_architecture = dataset_name + "_" + model_architecture
    base_path = "./checkpoint/" + data_architecture + "/"
    original_data_path = base_path + "/dataset/" + dataset_name + ".npz"
    original_model_path = base_path + "/saved_models/" + data_architecture + ".h5"
    ae_data_path = base_path + ae_generation_technique + "/ae/"
    selection_retrain_path = base_path + ae_generation_technique + "/retrain/"
    if not os.path.isdir(selection_retrain_path):
        os.makedirs(selection_retrain_path)

    print(f"Dataset: {dataset_name}, Model architecture: {model_architecture}, num_classes: {num_classes}, ae_generation_technique:{ae_generation_technique}")

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

    # Load the generated adversarial inputs for training
    if ae_generation_technique == "attack":
        with np.load(ae_data_path + "/fgsm_train_ae.npz") as f:
            fgsm_train_idx, fgsm_train, fgsm_train_label, fgsm_train_fol, fgsm_train_gini, fgsm_train_distance = f['idx'], f['ae'], f['ae_label'], f['fol'], f['gini'], f['distance']
        with np.load(ae_data_path + "/pgd_train_ae.npz") as f:
            pgd_train_idx, pgd_train, pgd_train_label, pgd_train_fol, pgd_train_gini, pgd_train_distance = f['idx'], f['ae'], f['ae_label'], f['fol'], f['gini'], f['distance']

        # Mix the adversarial inputs
        ae_train_idx = np.concatenate((fgsm_train_idx, pgd_train_idx))
        ae_train = np.concatenate((fgsm_train, pgd_train))
        ae_train_label = np.concatenate((fgsm_train_label, pgd_train_label))
        ae_train_fol = np.concatenate((fgsm_train_fol, pgd_train_fol))
        ae_train_gini = np.concatenate((fgsm_train_gini, pgd_train_gini))
        ae_train_distance = np.concatenate((fgsm_train_distance, pgd_train_distance))
        print(f"length of idx: {len(set(ae_train_idx))}, length of ae: {len(ae_train)}")

    elif ae_generation_technique == "adapt" or ae_generation_technique == "folfuzz" or ae_generation_technique == "contextfuzz":
        with np.load(ae_data_path + "/train_ae.npz") as f:
            ae_train_idx, ae_train, ae_train_label, ae_train_fol, ae_train_gini, ae_train_distance, ae_time = f['idx'], f['ae'], f['ae_label'], f['fol'], f['gini'], f['distance'], f['time']
        print(f"length of idx: {len(set(ae_train_idx))}, length of ae: {len(ae_train)}, time: {ae_time.max()}")

    # Load the generated adversarial inputs for testing. FGSM and PGD.
    with np.load(base_path + "attack/ae/env1/fgsm_test_ae.npz") as f:
        fgsm_test_idx, fgsm_test, fgsm_test_label = f['idx'], f['ae'], f['ae_label']

    with np.load(base_path + "attack/ae/env1/pgd_test_ae.npz") as f:
        pgd_test_idxs, pgd_test, pgd_test_label = f['idx'], f['ae'], f['ae_label']

    ae_test = np.concatenate((fgsm_test, pgd_test))
    ae_test_label = np.concatenate((fgsm_test_label, pgd_test_label))

    # sNums = [1000, 2000, 4000, 6000, 8000, 10000, 20000]
    # metrics = ['random', 'gini', 'best', 'kmst', 'distance']

    sNums = [6000, 8000, 10000, 20000, 1000, 2000, 4000]
    metrics = ['random']
    print(len(ae_train_idx), len(set(ae_train_idx)), len(ae_train_idx)/len(set(ae_train_idx)))

    for num in sNums:
        print(num, round(num/len(ae_train), 2))
        for metric in metrics:
            retrained_model_path = selection_retrain_path + "/best_retrain_%d_%s.h5" % (num, metric)
            checkpoint = ModelCheckpoint(filepath=retrained_model_path, monitor='val_accuracy', verbose=1, save_best_only=True)
            lr_scheduler = LearningRateScheduler(lr_schedule_retrain)
            lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
            callbacks = [checkpoint, lr_reducer, lr_scheduler]

            indexes = None
            if metric == 'random' or metric == 'best' or metric == 'kmst':
                indexes = existing_selection(data=ae_train_fol, select_num=num, select_metric=metric)
            elif metric == 'gini':
                indexes = existing_selection(data=ae_train_gini, select_num=num, select_metric=metric)
            elif metric == "distance":
                indexes = context_selection(data=ae_train_distance, x_idxs=ae_train_idx, select_num=num)
            else:
                print(f"The {metric} metric selection is not supported!")
                sys.exit()
            print("*" * 10 + f"Selection Parameters: select_num: {len(indexes)}, select_metric: {metric}" + "*" * 10)

            x_train_mix = np.concatenate((x_train, ae_train[indexes]), axis=0)
            y_train_mix = np.concatenate((y_train, ae_train_label[indexes]), axis=0)

            # load old model
            model = keras.models.load_model(original_model_path)
            datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
            datagen.fit(x_train_mix)
            batch_size = 64
            history = model.fit(datagen.flow(x_train_mix, y_train_mix, batch_size=batch_size), validation_data=(ae_test, ae_test_label), epochs=40, verbose=2,
                                callbacks=callbacks, steps_per_epoch=x_train_mix.shape[0] // batch_size, shuffle=True)

            scores = model.evaluate(x_test, y_test, verbose=2)
            print(f'Standard accuracy with {num} adversarial examples: {scores[1]}')
