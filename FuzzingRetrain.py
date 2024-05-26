import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from Retrain import existing_selection, context_selection
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def lr_schedule_retrain(epoch):
    lr = 1e-4
    print('Learning rate: ', lr)
    return lr


def lr_schedule_retrain_svhn(epoch):
    lr = 1e-3
    print('Learning rate: ', lr)
    return lr


if __name__ == '__main__':
    param_dict = {"dataset_name": ["fashion_mnist", "svhn", "cifar10", "cifar100"],
                  "model_architecture": ["vgg16", "lenet5", "resnet20", "resnet56"],
                  "num_classes": [10, 10, 10, 100],
                  "batch_size": [64, 64, 64, 64],
                  "duration": [10, 10, 10, 20]}

    for param_index in [0, 1, 2, 3]:
        dataset_name = param_dict["dataset_name"][param_index]
        model_architecture = param_dict["model_architecture"][param_index]
        num_classes = param_dict["num_classes"][param_index]
        batch_size = param_dict["batch_size"][param_index]
        adapt_duration = param_dict["duration"][param_index]
        fuzzer = "contextfuzz"         # ["adapt", "folfuzz", "contextfuzz", "contextfuzz_gini", "contextfuzz_fol"]
        select_metric = "cc"           # ["kmst", "kmst", "cc", "gini", "kmst"]

        data_architecture = f"{dataset_name}_{model_architecture}"
        base_path = f"./checkpoint/{data_architecture}"
        original_data_path = f"{base_path}/dataset/{dataset_name}.npz"
        train_valid_data_path = f"{base_path}/dataset/train_valid.npz"
        original_model_path = f"{base_path}/saved_models/{data_architecture}.h5"
        ae_data_path = f"{base_path}/{fuzzer}_runs0/ae/"
        selection_retrain_path = f"{base_path}/{fuzzer}_runs0/retrain_by_time/"
        if not os.path.isdir(selection_retrain_path):
            os.makedirs(selection_retrain_path)

        print(f"Model architecture: {model_architecture}, dataset: {dataset_name}, fuzz_type: {fuzzer}")
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)

        with np.load(train_valid_data_path) as f:
            x_train, x_valid, y_train, y_valid = f['x_train'], f['x_valid'], f['y_train'], f['y_valid']
        with np.load(original_data_path) as f:
            x_test, y_test = f['x_test'], f['y_test']

        if dataset_name == "fashion_mnist":
            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
            x_valid = x_valid.reshape(x_valid.shape[0], x_valid.shape[1], x_valid.shape[2], 1)
            x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

        x_train = x_train.astype('float32') / 255
        x_valid = x_valid.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_valid = tf.keras.utils.to_categorical(y_valid, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)

        with np.load(base_path + "/attack/ae/fgsm_valid_ae.npz") as f:
            fgsm_valid_idx, fgsm_valid, fgsm_valid_label = f['idx'], f['ae'], f['ae_label']

        with np.load(base_path + "/attack/ae/pgd_valid_ae.npz") as f:
            pgd_valid_idx, pgd_valid, pgd_valid_label = f['idx'], f['ae'], f['ae_label']

        with np.load(base_path + "/attack/ae/fgsm_test_ae.npz") as f:
            fgsm_test_idx, fgsm_test, fgsm_test_label = f['idx'], f['ae'], f['ae_label']

        with np.load(base_path + "/attack/ae/pgd_test_ae.npz") as f:
            pgd_test_idx, pgd_test, pgd_test_label = f['idx'], f['ae'], f['ae_label']

        ae_valid = np.concatenate((fgsm_valid, pgd_valid))
        ae_valid_label = np.concatenate((fgsm_valid_label, pgd_valid_label))
        ae_test = np.concatenate((fgsm_test, pgd_test))
        ae_test_label = np.concatenate((fgsm_test_label, pgd_test_label))

        durations = [1800, 3600, 7200, 18000]
        for duration in durations:
            if fuzzer == "attack_control":
                with np.load(ae_data_path + "/fgsm_train_ae.npz") as f:
                    fgsm_train_idx, fgsm_train, fgsm_train_labels, fgsm_train_fols, fgsm_train_ginis, fgsm_train_cc = f['idx'], f['ae'], f['ae_label'], f['fol'], f['gini'], f['cc']
                with np.load(ae_data_path + "/pgd_train_ae.npz") as f:
                    pgd_train_idx, pgd_train, pgd_train_labels, pgd_train_fols, pgd_train_ginis, pgd_train_cc = f['idx'], f['ae'], f['ae_label'], f['fol'], f['gini'], f['cc']

                ae_train_idx_all = np.concatenate((fgsm_train_idx, pgd_train_idx))
                ae_train_all = np.concatenate((fgsm_train, pgd_train))
                ae_train_label_all = np.concatenate((fgsm_train_labels, pgd_train_labels))
                ae_train_fol_all = np.concatenate((fgsm_train_fols, pgd_train_fols))
                ae_train_gini_all = np.concatenate((fgsm_train_ginis, pgd_train_ginis))
                ae_train_cc_all = np.concatenate((fgsm_train_cc, pgd_train_cc))
                cur_idxs = np.array(range(len(ae_train_all)))
            elif fuzzer == "adapt":
                with np.load(ae_data_path + "/train_ae.npz") as f:
                    ae_train_idx_all, ae_train_all, ae_train_label_all, ae_train_fol_all, ae_train_gini_all, ae_train_cc_all = f['idx'], f['ae'], f['ae_label'], f['fol'], f['gini'], f['cc']
                cur_idxs = np.where(ae_train_idx_all < (duration // adapt_duration))[0]
            elif fuzzer == "folfuzz":
                with np.load(ae_data_path + "/train_ae.npz") as f:
                    ae_train_idx_all, ae_train_all, ae_train_label_all, ae_train_fol_all, ae_train_gini_all, ae_train_cc_all, ae_time_all = f['idx'], f['ae'], f['ae_label'], f['fol'], f['gini'], f['cc'], f['time']
                cur_idxs = np.where(ae_time_all <= duration)[0]
            elif fuzzer == "contextfuzz":
                with np.load(ae_data_path + "/train_ae_" + str(duration) + ".npz") as f:
                    ae_train_idx_all, ae_train_all, ae_train_label_all, ae_train_fol_all, ae_train_cc_all, ae_train_gini_all, ae_time_all = f['idx'], f['ae'], f['ae_label'], f['fol'], f['cc'], f['gini'], f['time']
                cur_idxs = np.where(ae_time_all <= duration)[0]
            elif fuzzer == "contextfuzz_gini":
                with np.load(ae_data_path + "/train_ae_" + str(duration) + ".npz") as f:
                    ae_train_idx_all, ae_train_all, ae_train_label_all, ae_train_fol_all, ae_train_cc_all, ae_train_gini_all, ae_time_all = f['idx'], f['ae'], f['ae_label'], f['fol'], f['cc'], f['gini'], f['time']
                cur_idxs = np.where(ae_time_all <= duration)[0]
            elif fuzzer == "contextfuzz_fol":
                with np.load(ae_data_path + "/train_ae_" + str(duration) + ".npz") as f:
                    ae_train_idx_all, ae_train_all, ae_train_label_all, ae_train_fol_all, ae_train_cc_all, ae_train_gini_all, ae_time_all = f['idx'], f['ae'], f['ae_label'], f['fol'], f['cc'], f['gini'], f['time']
                cur_idxs = np.where(ae_time_all <= duration)[0]
            else:
                print(f"The {fuzzer} ae generation technique is not supported!")

            print(f"ae_train_idx:{len(ae_train_idx_all)} ae_train:{len(ae_train_all)} ae_train_label:{len(ae_train_label_all)} ae_train_cc:{len(ae_train_cc_all)}")
            np.random.shuffle(cur_idxs)
            ae_train_idx, ae_train, ae_train_label, ae_train_fol, ae_train_cc, ae_train_gini = ae_train_idx_all[cur_idxs], ae_train_all[cur_idxs], ae_train_label_all[cur_idxs], \
                                                                                      ae_train_fol_all[cur_idxs], ae_train_cc_all[cur_idxs], ae_train_gini_all[cur_idxs]

            sNums = [1000, 2000, 4000, 6000]
            for num in sNums:
                retrained_model_path = selection_retrain_path + "/best_retrain_%d_%d_%s.h5" % (duration, num, select_metric)
                checkpoint = ModelCheckpoint(filepath=retrained_model_path, monitor='val_accuracy', verbose=1, save_best_only=True)
                if param_index == 1:
                    lr_scheduler = LearningRateScheduler(lr_schedule_retrain_svhn)
                else:
                    lr_scheduler = LearningRateScheduler(lr_schedule_retrain)
                lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
                early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)
                callbacks = [checkpoint, lr_reducer, lr_scheduler, early_stopping]

                if select_metric == 'random':
                    indexes = existing_selection(data=ae_train_fol, select_num=num, select_metric=select_metric)
                elif select_metric == 'kmst':
                    indexes = existing_selection(data=ae_train_fol, select_num=num, select_metric=select_metric)
                elif select_metric == 'gini':
                    indexes = existing_selection(data=ae_train_gini, select_num=num, select_metric=select_metric)
                elif select_metric == "cc":
                    indexes = context_selection(data=ae_train_cc, x_idxs=ae_train_idx, y_ae=ae_train_label, select_num=num)
                elif select_metric == "first":
                    indexes = np.array(list(range(len(ae_train))))[0:num]
                else:
                    print(f"The {select_metric} metric selection is not supported!")
                    sys.exit()
                print("*" * 10 + f"Duration: {duration}, Selection Parameters: select_num: {len(indexes)}, select_metric: {select_metric}" + "*" * 10)

                x_train_mix = np.concatenate((x_train, ae_train[indexes]), axis=0)
                y_train_mix = np.concatenate((y_train, ae_train_label[indexes]), axis=0)

                x_valid_mix = np.concatenate((x_valid, ae_valid), axis=0)
                y_valid_mix = np.concatenate((y_valid, ae_valid_label), axis=0)

                # load old model
                model = tf.keras.models.load_model(original_model_path)
                datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
                datagen.fit(x_train_mix)
                history = model.fit_generator(datagen.flow(x_train_mix, y_train_mix, batch_size=batch_size), validation_data=(x_valid_mix, y_valid_mix), epochs=40, verbose=2,
                                              callbacks=callbacks, steps_per_epoch=x_train_mix.shape[0] // batch_size, shuffle=True)

                model = tf.keras.models.load_model(retrained_model_path)
                scores_std_train = model.evaluate(x_train, y_train, verbose=2)
                scores_std_test = model.evaluate(x_test, y_test, verbose=2)
                scores_ae_test = model.evaluate(ae_test, ae_test_label, verbose=2)
                print(f"Num: {num}, Std train: {round(scores_std_train[1], 4) * 100}, Std test: {round(scores_std_test[1], 4) * 100}, AE test: {round(scores_ae_test[1], 4) * 100}")
