import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import time
import numpy as np
import tensorflow as tf
from Context import Context

np.set_printoptions(threshold=np.inf)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class ContextFuzzing:
    def __init__(self, model, max_step_size, ep=0.05, delta=0.01, gc_num=5):
        self.model = model
        self.max_step = max_step_size
        self.ep = ep
        self.delta = delta
        self.gc_num = gc_num
        self.time_start = time.time()

    def step(self, current_step):
        return self.max_step * np.sin(current_step*np.pi/(int(self.ep/self.delta)+1))

    def fuzzing_process(self, x, y, budget):
        ae_idx_list, ae_list, ae_label_list, ae_similarity_list, ae_cc_list, ae_sample_prediction_list, ae_time_list, ae_gini_list, ae_fol_list = [], [], [], [], [], [], [], [], []
        x_ctx_label_list, x_ctx_similarity_list, x_ctx_cc_list, x_gini_list, x_fol_list, x_ctx_grad_list, x_ctx_inc_list = [], [], [], [], [], [], []

        xi = x.copy()
        target = tf.constant(y, dtype=float)
        original_image_ground_truth = np.argmax(y, axis=1)
        original_image_pred_label = np.argmax(self.model(x), axis=1)
        original_correct_idxs = np.where(original_image_pred_label == original_image_ground_truth)[0]
        seed_num = 900
        x_adv = tf.Variable(x, dtype=float)
        with tf.GradientTape() as tape:
            loss = tf.keras.losses.categorical_crossentropy(target, self.model(x_adv))
            grads = tape.gradient(loss, x_adv)
        gradient_matrix = tf.sign(grads)
        x_adv = x + gradient_matrix * self.ep
        x_adv = tf.clip_by_value(x_adv, clip_value_min=xi-self.ep, clip_value_max=xi+self.ep)
        x_adv = tf.clip_by_value(x_adv, clip_value_min=0.0, clip_value_max=1.0)

        x_adv_pred_vector = self.model(x_adv)
        x_adv_pred_label = np.argmax(x_adv_pred_vector, axis=1)
        x_aes_incorrect_idxs = np.where(x_adv_pred_label != original_image_ground_truth)[0]
        idxs = np.array(list(set(original_correct_idxs).intersection(set(x_aes_incorrect_idxs))))

        x_ctx_grad_list = list(gradient_matrix.numpy())
        for idx, (original_x, ae, ground_truth_vector, ae_label) in enumerate(zip(xi, x_adv.numpy(), target.numpy(), x_adv_pred_label)):
            x_ctx_label_list.append(ae_label)
            x_ctx_similarity_list.append(-np.inf)
            x_ctx_cc_list.append(np.inf)
            x_gini_list.append(-np.inf)
            x_fol_list.append(0)
            x_ctx_inc_list.append(1)
            if idx in idxs:
                ae_idx_list.append(idx)
                ae_list.append(ae)
                ae_label_list.append(ground_truth_vector)
                new_context = Context(self.model, ae=np.expand_dims(ae, axis=0), ae_label=ae_label, k=20)
                ae_similarity_list.append(new_context.similarity())
                ae_cc_list.append(new_context.cc())
                ae_sample_prediction_list.append(new_context.sample_prediction())
                ae_time_list.append(int(time.time() - self.time_start))
                new_gini = 1 - np.sum(np.square(x_adv_pred_vector[idx]))
                ae_gini_list.append(new_gini)
                new_fol = np.linalg.norm(grads.numpy()[idx])
                ae_fol_list.append(new_fol)

                # update ctx if it is an ae
                x_ctx_similarity_list[idx] = new_context.similarity()
                x_ctx_cc_list[idx] = new_context.cc()
                x_gini_list[idx] = new_gini
                x_fol_list[idx] = new_fol
                x_ctx_inc_list[idx] += 1
        print(f"first stage: ae length: {len(ae_list)}, ae list shape: {np.array(ae_list).shape}, ae label shape: {np.array(ae_label_list).shape} ae dist shape: {np.array(x_ctx_cc_list).shape},  time: {int(time.time()-self.time_start)}")

        while True:
            energy = self.gc_num * np.array(x_ctx_inc_list) / np.array(x_ctx_inc_list).mean()   # calculate energy
            for batch_idx in range(int(len(x)/seed_num)):
                cur_idxs = np.array(range(batch_idx*seed_num, (batch_idx+1)*seed_num))
                equivalent_class_dict = {(c_i, c_j):np.array(list(set(np.where(original_image_ground_truth == c_i)[0]).intersection(set(np.where(np.array(x_ctx_label_list) == c_j)[0])))) for c_i in range(y.shape[1]) for c_j in range(y.shape[1])}
                gc_grads_all = [list(np.array(x_ctx_grad_list)[equivalent_class_dict[(original_image_ground_truth[idx], x_ctx_label_list[idx])][np.random.choice(len(equivalent_class_dict[(original_image_ground_truth[idx], x_ctx_label_list[idx])]), min(len(equivalent_class_dict[(original_image_ground_truth[idx], x_ctx_label_list[idx])]), int(np.ceil(energy[idx]))), replace=False)]]) for idx in cur_idxs]

                while True:
                    left_gc_num = np.array([len(item) for item in gc_grads_all])
                    left_gc_idxs = np.where(left_gc_num > 0)[0]
                    if len(left_gc_idxs) == 0:
                        break

                    left_cur_idxs = cur_idxs[left_gc_idxs]
                    left_gc_delta = np.array([gc_grads_all[left_gc_idx].pop() for left_gc_idx in left_gc_idxs])
                    feedback_grad = np.array(x_ctx_grad_list)[left_cur_idxs]

                    x_ae = tf.Variable(x[left_cur_idxs], dtype=float)
                    target = tf.constant(y[left_cur_idxs], dtype=float)
                    for step_idx in range(0, int(self.ep/self.delta)+1):
                        with tf.GradientTape() as tape:
                            loss = tf.keras.losses.categorical_crossentropy(target, self.model(x_ae))
                            grads = tape.gradient(loss, x_ae)
                        grads_sign = tf.sign(grads)
                        if step_idx == 0:
                            feedback_grad = grads_sign + left_gc_delta + feedback_grad
                            x_ae.assign_add(self.max_step * feedback_grad * self.ep)
                        else:
                            x_ae.assign_add(self.step(current_step=step_idx) * grads_sign * self.ep)
                        x_ae = tf.clip_by_value(x_ae, clip_value_min=xi[left_cur_idxs]-self.ep, clip_value_max=xi[left_cur_idxs]+self.ep)
                        x_ae = tf.clip_by_value(x_ae, clip_value_min=0.0, clip_value_max=1.0)
                        x_ae = tf.Variable(x_ae)

                        new_ae_vector = self.model(x_ae)
                        new_ae_label = np.argmax(new_ae_vector, axis=1)
                        for idx, left_cur_idx in enumerate(left_cur_idxs):
                            if original_image_pred_label[left_cur_idx] == np.argmax(y[left_cur_idx]) and new_ae_label[idx] != np.argmax(y[left_cur_idx]):
                                ae_idx_list.append(left_cur_idx)
                                ae_list.append(x_ae[idx])
                                ae_label_list.append(y[left_cur_idx])
                                new_context = Context(self.model, ae=np.expand_dims(x_ae[idx], axis=0), ae_label=new_ae_label[idx], k=20)
                                ae_similarity_list.append(new_context.similarity())
                                ae_cc_list.append(new_context.cc())
                                ae_sample_prediction_list.append(new_context.sample_prediction())
                                ae_time_list.append(int(time.time() - self.time_start))
                                new_gini = 1 - np.sum(np.square(new_ae_vector[idx]))
                                ae_gini_list.append(new_gini)
                                new_fol = np.linalg.norm(grads.numpy()[idx])
                                ae_fol_list.append(new_fol)

                                if new_fol > x_fol_list[left_cur_idx]:
                                    x_ctx_label_list[left_cur_idx] = new_ae_label[idx]
                                    x_ctx_similarity_list[left_cur_idx] = new_context.similarity()
                                    x_ctx_cc_list[left_cur_idx] = new_context.cc()
                                    x_gini_list[left_cur_idx] = new_gini
                                    x_fol_list[left_cur_idx] = new_fol
                                    x_ctx_grad_list[left_cur_idx] = x_ae[idx] - xi[left_cur_idx]
                                    x_ctx_inc_list[left_cur_idx] += 1

                        # time out
                        if (time.time() - self.time_start) >= budget:
                            print(f"second stage: ae length: {len(ae_list)}, ae list shape: {np.array(ae_list).shape}, ae label shape: {np.array(ae_label_list).shape} "
                                  f"ae dist shape: {np.array(x_ctx_cc_list).shape}, time: {int(time.time() - self.time_start)}")
                            return np.array(ae_idx_list), np.array(ae_list), np.array(ae_label_list), np.array(ae_similarity_list), \
                                   np.array(ae_cc_list), np.array(ae_sample_prediction_list), np.array(ae_time_list), np.array(ae_gini_list), np.array(ae_fol_list)
            print(f"time for one epoch is {time.time() - self.time_start}")


if __name__ == "__main__":
    param_dict = {"dataset_name": ["fashion_mnist", "svhn", "cifar10", "cifar100"],
                  "model_architecture": ["vgg16", "lenet5", "resnet20", "resnet56"],
                  "num_classes": [10, 10, 10, 100],
                  "ep": [0.05, 0.05, 0.05, 0.05],
                  "delta": [0.01, 0.01, 0.01, 0.01]}

    for param_index in [0, 1, 2, 3]:
        dataset_name = param_dict["dataset_name"][param_index]
        model_architecture = param_dict["model_architecture"][param_index]
        num_classes = param_dict["num_classes"][param_index]
        ep = param_dict["ep"][param_index]
        delta = param_dict["delta"][param_index]

        data_architecture = f"{dataset_name}_{model_architecture}"
        base_path = f"./checkpoint/{data_architecture}"
        original_data_path = f"{base_path}/dataset/{dataset_name}.npz"
        train_valid_data_path = f"{base_path}/dataset/train_valid.npz"
        original_model_path = f"{base_path}/saved_models/{data_architecture}.h5"
        ae_data_path = f"{base_path}/contextfuzz_fol/ae/"
        if not os.path.exists(ae_data_path):
            os.makedirs(ae_data_path)

        print(f"Generate by ContextFuzz. Model architecture: {model_architecture}, dataset: {dataset_name}, num_classes: {num_classes}, ep: {ep}")

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

        original_model = tf.keras.models.load_model(original_model_path)

        time_budgets = [18000]
        seed_numbers = [18000]
        for dur_idx in range(len(time_budgets)):
            time_budget = time_budgets[dur_idx]
            seed_number = seed_numbers[dur_idx]
            context_fuzzing = ContextFuzzing(model=original_model, max_step_size=0.2, delta=delta, ep=ep, gc_num=5)
            all_ae_idx, all_ae, all_ae_label, all_ae_similarity, all_ae_cc, all_ae_sample_prediction, all_ae_time, all_ae_gini, all_ae_fol = \
                context_fuzzing.fuzzing_process(x=x_train[:seed_number], y=y_train[:seed_number], budget=time_budget)
            np.savez(ae_data_path + '/train_ae_' + str(time_budget) + '.npz', idx=all_ae_idx, ae=all_ae, ae_label=all_ae_label, similarity=all_ae_similarity, cc=all_ae_cc,
                     sample_prediction=all_ae_sample_prediction, time=all_ae_time, gini=all_ae_gini, fol=all_ae_fol)

