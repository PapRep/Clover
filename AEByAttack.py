import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from utils.attack import FGSM, PGD
import tensorflow as tf
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


if __name__ == '__main__':
    param_dict = {"dataset_name": ["fashion_mnist", "svhn", "cifar10", "cifar100"],
                  "model_architecture": ["vgg16", "lenet5", "resnet20", "resnet56"],
                  "num_classes": [10, 10, 10, 100],
                  "ep": [0.03, 0.03, 0.01, 0.01]}

    for param_index in [0, 1, 2, 3]:
        dataset_name = param_dict["dataset_name"][param_index]
        model_architecture = param_dict["model_architecture"][param_index]
        num_classes = param_dict["num_classes"][param_index]
        ep = param_dict["ep"][param_index]

        data_architecture = dataset_name + "_" + model_architecture
        base_path = "./checkpoint/" + data_architecture
        original_data_path = base_path + "/dataset/" + dataset_name + ".npz"
        train_valid_data_path = base_path + "/dataset/train_valid.npz"
        original_model_path = base_path + "/saved_models/" + data_architecture + ".h5"
        ae_data_path = base_path + "/attack/ae/"
        if not os.path.exists(ae_data_path):
            os.makedirs(ae_data_path)

        print(f"Generate by PGD and FGSM. Model architecture: {model_architecture}, dataset: {dataset_name}, num_classes: {num_classes}, ep: {ep}")

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

        # generate adversarial examples at once.
        data_type_list = ["train", "valid", "test"]      # train, valid or test
        attack_method_list = ["fgsm", "pgd"]    # FGSM or "PGD"

        for data_type in data_type_list:
            if data_type == "train":
                data_for_ae = x_train
                label_for_ae = y_train
                ae_total_num = 50000
            elif data_type == "valid":
                data_for_ae = x_valid
                label_for_ae = y_valid
                ae_total_num = 10000
            elif data_type == "test":
                data_for_ae = x_test
                label_for_ae = y_test
                ae_total_num = 10000

            for attack_method in attack_method_list:
                model = tf.keras.models.load_model(original_model_path)
                if attack_method == "fgsm":
                    attack_tech = FGSM(model, ep=ep, isRand=True)
                elif attack_method == "pgd":
                    attack_tech = PGD(model, ep=ep, epochs=10, isRand=True)

                print(f"dataset: {dataset_name}, data_type:{data_type}, attack_method:{attack_method}, ae_total_num:{ae_total_num}")
                idx_all = []
                ae_all = []
                ae_label_all = []
                fol_all = []
                gini_all = []
                similarity_all = []
                cc_all = []
                sample_prediction_all = []
                time_all = []

                batch_size = 100
                while len(ae_all) < ae_total_num:
                    for i in range(round(len(data_for_ae)/batch_size)):
                        idx, ae, ae_label, fol, gini, similarity, cc, sample_prediction, gen_time = attack_tech.generate(data_for_ae[i*batch_size: (i+1)*batch_size], label_for_ae[i*batch_size: (i+1)*batch_size])
                        idx_all.extend(idx+i*batch_size)
                        ae_all.extend(ae)
                        ae_label_all.extend(ae_label)
                        fol_all.extend(fol)
                        gini_all.extend(gini)
                        similarity_all.extend(similarity)
                        cc_all.extend(cc)
                        sample_prediction_all.extend(sample_prediction)
                        time_all.extend(gen_time)

                        if len(ae_all) > ae_total_num:
                            break
                np.savez(ae_data_path + attack_method + "_" + data_type + "_ae.npz", idx=np.array(idx_all)[0:ae_total_num], ae=np.array(ae_all)[0:ae_total_num],
                         ae_label=np.array(ae_label_all)[0:ae_total_num], fol=np.array(fol_all)[0:ae_total_num], gini=np.array(gini_all)[0:ae_total_num],
                         similarity=np.array(similarity_all)[0:ae_total_num], cc=np.array(cc_all)[0:ae_total_num],
                         sample_prediction=np.array(sample_prediction_all)[0:ae_total_num], time=np.array(time_all))
