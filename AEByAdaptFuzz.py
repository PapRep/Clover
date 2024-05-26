import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
from tensorflow import keras
from utils.adapt import Network
from utils.adapt.metric import NC, TKNC
from utils.adapt.fuzzer import WhiteBoxFuzzer
from utils.adapt.strategy import AdaptiveParameterizedStrategy, UncoveredRandomStrategy, MostCoveredStrategy
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


if __name__ == '__main__':
    param_dict = {"dataset_name": ["fashion_mnist", "svhn", "cifar10", "cifar100"],
                  "model_architecture": ["vgg16", "lenet5", "resnet20", "resnet56"],
                  "num_classes": [10, 10, 10, 100],
                  "duration": [10, 10, 10, 20]}

    for param_index in [0, 1, 2, 3]:
        dataset_name = param_dict["dataset_name"][param_index]
        model_architecture = param_dict["model_architecture"][param_index]
        num_classes = param_dict["num_classes"][param_index]
        duration = param_dict["duration"][param_index]
        ae_generation_technique = "adapt"
        num_input = 2000
        budget = 18000

        data_architecture = dataset_name + "_" + model_architecture
        base_path = "./checkpoint/" + data_architecture
        original_data_path = base_path + "/dataset/" + dataset_name + ".npz"
        train_valid_data_path = base_path + "/dataset/train_valid.npz"
        original_model_path = base_path + "/saved_models/" + data_architecture + ".h5"
        ae_data_path = base_path + "/" + ae_generation_technique + "/ae/"
        if not os.path.exists(ae_data_path):
            os.makedirs(ae_data_path)

        print(f"Generate by {ae_generation_technique}, Model architecture: {model_architecture}, dataset: {dataset_name}, num_input:{num_input}, num_classes: {num_classes}")

        with np.load(train_valid_data_path) as f:
            x_train, x_valid, y_train, y_valid = f['x_train'], f['x_valid'], f['y_train'], f['y_valid']
        with np.load(original_data_path) as f:
            x_test, y_test = f['x_test'], f['y_test']

        if dataset_name == "fashion_mnist":
            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
            x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

        # preprocess cifar dataset
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

        # generate adversarial examples at once.
        data_type_list = ["train"]

        for data_type in data_type_list:
            if data_type == "train":
                data_for_ae = x_train
                label_for_ae = y_train
            else:
                data_for_ae = x_test
                label_for_ae = y_test

            network = Network(keras.models.load_model(original_model_path))
            idx_all = []
            ae_all = []
            ae_label_all = []
            fol_all = []
            gini_all = []
            similarity_all = []
            cc_all = []
            sample_prediction_all = []

            metric = NC(0.5)
            for i in range(num_input):
                if np.argmax(network.predict(np.expand_dims(data_for_ae[i], axis=0))[1]) != np.argmax(label_for_ae[i]):
                    continue

                if ae_generation_technique == "adapt":
                    strategy = AdaptiveParameterizedStrategy(network)
                else:
                    strategy = None

                fuzzer = WhiteBoxFuzzer(network=network, input=data_for_ae[i], ground_truth=label_for_ae[i], metric=metric, strategy=strategy, k=10, class_weight=0.5, neuron_weight=0.5, trail=3, decode=None)
                ae, ae_label, fol, gini, similarity, cc, sample_prediction = fuzzer.start(seconds=duration, append='min_dist')
                idx_all.extend(np.array([i]*len(ae)))
                ae_all.extend(ae)
                ae_label_all.extend(ae_label)
                fol_all.extend(fol)
                gini_all.extend(gini)
                similarity_all.extend(similarity)
                cc_all.extend(cc)
                sample_prediction_all.extend(sample_prediction)

                print(f"---------------------{i}, {len(ae)}, {len(ae_all)}----------------------")

                if (i+1) * duration >= budget:
                    print(f"New ae {len(ae)}, Current total ae {len(ae_all)}")
                    np.savez(ae_data_path + data_type + "_ae.npz", idx=np.array(idx_all), ae=np.array(ae_all), ae_label=np.array(ae_label_all), fol=np.array(fol_all),
                             gini=np.array(gini_all), similarity=np.array(similarity_all), cc=np.array(cc_all), sample_prediction=np.array(sample_prediction_all))
                    break

            print(f"length of AE {len(ae_all)}")
            np.savez(ae_data_path + data_type + "_ae.npz", idx=np.array(idx_all), ae=np.array(ae_all), ae_label=np.array(ae_label_all), fol=np.array(fol_all),
                     gini=np.array(gini_all), similarity=np.array(similarity_all), cc=np.array(cc_all), sample_prediction=np.array(sample_prediction_all))

