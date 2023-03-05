import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

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
    # lr = 1e-3
    lr = 1e-4
    print('Learning rate: ', lr)
    return lr


def equally_divide_selection(data, y_ae, select_num, select_metric="moc", classes=10, by_class=True, sections=5):
    rank = np.argsort(data)
    if select_metric == "distance":
        pass
    elif select_metric == "gini":
        rank = rank[::-1]

    selected_index_all_sections_list = []
    for i in range(sections):
        start_idx = len(np.where(data > (1-i*(1./sections)))[0])
        end_idx = len(np.where(data > (1-(i+1)*(1./sections)))[0])
        rank_section = rank[start_idx:end_idx]

        if by_class:
            sorted_y_ae = y_ae[rank_section]
            selected_index = np.array([], dtype=np.int64)
            for c in range(classes):
                selected_index = np.concatenate((selected_index, np.where(sorted_y_ae == c)[0][0:int(select_num / classes)]))
        else:
            selected_index = rank_section[0:select_num]
        selected_index_all_sections_list.append(selected_index)
        print(f"start_idx:{start_idx}, end_idx:{end_idx}, num in section {i}:{len(rank_section)}, select num in section {i}:{len(selected_index)}")
    return selected_index_all_sections_list


if __name__ == '__main__':
    # all parameters
    param_dict = {"dataset_name": ["fashion_mnist", "svhn", "cifar10", "cifar100"],
                  "model_architecture": ["vgg16", "lenet5", "resnet20", "resnet56"],
                  "num_classes": [10, 10, 10, 100]}

    # parameters
    param_index = 0
    dataset_name = param_dict["dataset_name"][param_index]
    model_architecture = param_dict["model_architecture"][param_index]
    num_classes = param_dict["num_classes"][param_index]
    ae_generation_technique = "attack"      # ["attack", "dlfuzz", "adapt"]

    # path
    data_architecture = dataset_name + "_" + model_architecture
    base_path = "./checkpoint/" + data_architecture + "/"
    original_data_path = base_path + "/dataset/" + dataset_name + ".npz"
    original_model_path = base_path + "/saved_models/" + data_architecture + ".h5"
    ae_data_path = base_path + "/attack/ae/"
    equally_divide_selection_path = base_path + "/equally_divide_selection/"
    if not os.path.exists(equally_divide_selection_path):
        os.makedirs(equally_divide_selection_path)

    print(f"Model architecture: {model_architecture}, dataset: {dataset_name}, num_classes: {num_classes}")

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

    ae_train_idxs, ae_trains, ae_train_labels, ae_train_fols, ae_train_ginis, ae_train_mocs, ae_tests, ae_test_labels = None, None, None, None, None, None, None, None
    if ae_generation_technique == "attack":
        # Load the generated adversarial inputs for training. FGSM and PGD.
        with np.load(ae_data_path + "/fgsm_train_ae.npz") as f:
            fgsm_train_idx, fgsm_train, fgsm_train_label, fgsm_train_fol, fgsm_train_gini, fgsm_train_moc = f['idx'], f['ae'], f['ae_label'], f['fol'], f['gini'], 1 - f['distance']

        with np.load(ae_data_path + "/pgd_train_ae.npz") as f:
            pgd_train_idx, pgd_train, pgd_train_label, pgd_train_fol, pgd_train_gini, pgd_train_moc = f['idx'], f['ae'], f['ae_label'], f['fol'], f['gini'], 1 - f['distance']

        # Load the generated adversarial inputs for testing. FGSM and PGD.
        with np.load(ae_data_path + "/fgsm_test_ae.npz") as f:
            fgsm_test_idx, fgsm_test, fgsm_test_labels, fgsm_test_fols, fgsm_test_ginis, fgsm_test_distance = f['idx'], f['ae'], f['ae_label'], f['fol'], f['gini'], f['distance']

        with np.load(ae_data_path + "/pgd_test_ae.npz") as f:
            pgd_test_idx, pgd_test, pgd_test_labels, pgd_test_fols, pgd_test_ginis, pgd_test_distance = f['idx'], f['ae'], f['ae_label'], f['fol'], f['gini'], f['distance']

        # Mix the adversarial inputs
        ae_train_idxs = np.concatenate((fgsm_train_idx, pgd_train_idx))
        ae_trains = np.concatenate((fgsm_train, pgd_train))
        ae_train_labels = np.concatenate((fgsm_train_label, pgd_train_label))
        ae_train_fols = np.concatenate((fgsm_train_fol, pgd_train_fol))
        ae_train_ginis = np.concatenate((fgsm_train_gini, pgd_train_gini))
        ae_train_mocs = np.concatenate((fgsm_train_moc, pgd_train_moc))

        ae_tests = np.concatenate((fgsm_test, pgd_test))
        ae_test_labels = np.concatenate((fgsm_test_labels, pgd_test_labels))
    elif ae_generation_technique == "dlfuzz":
        pass
    elif ae_generation_technique == "adapt":
        pass

    all_sections = [1.0, 0.8, 0.6, 0.4, 0.2]
    sNums = [int(len(ae_trains) * 0.01 * i) for i in [1, 2, 4, 6]]
    metrics = ['distance', 'gini', 'fol']
    for num in sNums:
        for metric in metrics:
            if metric == "distance":
                indexes = equally_divide_selection(data=ae_train_mocs, y_ae=ae_train_labels, sections=5, select_num=num, select_metric=metric, classes=num_classes, by_class=True)
            elif metric == "gini":
                indexes = equally_divide_selection(data=ae_train_ginis, y_ae=ae_train_labels, sections=5, select_num=num, select_metric=metric, classes=num_classes, by_class=False)
            elif metric == "fol":
                indexes = equally_divide_selection(data=ae_train_fols, y_ae=ae_train_labels, sections=5, select_num=num, select_metric=metric, classes=num_classes, by_class=False)
            else:
                indexes = []

            for idx, indexes in enumerate(indexes):
                retrained_model_path = equally_divide_selection_path + "best_retrain_%d_%s_%s.h5" % (num, metric, str(all_sections[idx]))
                checkpoint = ModelCheckpoint(filepath=retrained_model_path, monitor='val_accuracy', verbose=1, save_best_only=True)
                lr_scheduler = LearningRateScheduler(lr_schedule_retrain)
                lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
                callbacks = [checkpoint, lr_reducer, lr_scheduler]

                x_train_mix = np.concatenate((x_train, ae_trains[indexes]), axis=0)
                y_train_mix = np.concatenate((y_train, ae_train_labels[indexes]), axis=0)

                # load old model
                model = keras.models.load_model(original_model_path)
                datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
                datagen.fit(x_train_mix)
                batch_size = 64
                history = model.fit_generator(datagen.flow(x_train_mix, y_train_mix, batch_size=batch_size), validation_data=(ae_tests, ae_test_labels), epochs=40, verbose=2,
                                              callbacks=callbacks, steps_per_epoch=x_train_mix.shape[0] // batch_size, shuffle=True)

                scores = model.evaluate(x_test, y_test, verbose=2)
                print(f'Standard accuracy with {num} adversarial examples: ', scores[1])
