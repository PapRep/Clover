import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import copy
import time
import random
import numpy as np
import tensorflow as tf
from Context import Context
np.set_printoptions(threshold=np.inf)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


if __name__ == '__main__':
    param_dict = {"dataset_name": ["fashion_mnist", "svhn", "cifar10", "cifar100"],
                  "model_architecture": ["vgg16", "lenet5", "resnet20", "resnet56"],
                  "num_classes": [10, 10, 10, 100],
                  "ep": [0.05, 0.05, 0.05, 0.05]}

    for param_index in [0, 1, 2, 3]:
        dataset_name = param_dict["dataset_name"][param_index]
        model_architecture = param_dict["model_architecture"][param_index]
        num_classes = param_dict["num_classes"][param_index]
        ep = param_dict["ep"][param_index]
        ae_generation_technique = "folfuzz"
        num_input = 18000
        budget = 18000

        data_architecture = dataset_name + "_" + model_architecture
        base_path = "./checkpoint/" + data_architecture
        original_data_path = base_path + "/dataset/" + dataset_name + ".npz"
        train_valid_data_path = base_path + "/dataset/train_valid.npz"
        original_model_path = base_path + "/saved_models/" + data_architecture + ".h5"
        ae_data_path = base_path + "/" + ae_generation_technique + "/ae/"
        if not os.path.exists(ae_data_path):
            os.makedirs(ae_data_path)

        print(f"Generate by {ae_generation_technique}, Model architecture: {model_architecture}, dataset: {dataset_name}, num_input:{num_input}, num_classes: {num_classes}, ep: {ep}")

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

        x_train = x_train[0:num_input]
        y_train = y_train[0:num_input]

        batch_size = 1000
        total_sets = []
        time_start = time.time()
        model = tf.keras.models.load_model(original_model_path)

        for i in range(int(np.ceil(len(x_train)/batch_size))):
            seeds = np.array(range(x_train.shape[0]))[i*batch_size: (i+1)*batch_size]
            images = x_train[seeds]
            labels = y_train[seeds]

            # some training samples is static, i.e., grad=<0>, hard to generate.
            gen_img = tf.Variable(images)
            with tf.GradientTape() as g:
                loss = tf.keras.losses.categorical_crossentropy(labels, model(gen_img))
                grads = g.gradient(loss, gen_img)

            fols = np.linalg.norm((grads.numpy()+1e-20).reshape(images.shape[0], -1), ord=2, axis=1)
            seeds_filter = np.where(fols > 1e-3)[0]

            lam = 1
            top_k = 5
            steps = 3
            for idx in seeds_filter:
                img_list = []
                tmp_img = images[[idx]]
                orig_img = copy.deepcopy(tmp_img)
                orig_norm = np.linalg.norm(orig_img)
                img_list.append(tf.identity(tmp_img))
                logits = model(tmp_img)
                orig_index = np.argmax(logits[0])
                target = tf.keras.utils.to_categorical([orig_index], num_classes)
                label_top5 = np.argsort(logits[0])[-top_k:-1][::-1]
                folMAX = 0.0

                while len(img_list) > 0:
                    gen_img = img_list.pop(0)

                    for _ in range(steps):
                        gen_img = tf.Variable(gen_img, dtype=float)
                        with tf.GradientTape(persistent=True) as g:
                            loss = tf.keras.losses.categorical_crossentropy(target, model(gen_img))
                            grads = g.gradient(loss, gen_img)
                            fol = tf.norm(grads+1e-20)
                            g.watch(fol)
                            logits = model(gen_img)
                            obj = lam*fol - logits[0][orig_index]
                            dl_di = g.gradient(obj, gen_img)
                        del g

                        gen_img = gen_img + dl_di * 0.1 * (random.random() + 0.5)
                        gen_img = tf.clip_by_value(gen_img, clip_value_min=0.0, clip_value_max=1.0)

                        with tf.GradientTape() as t:
                            t.watch(gen_img)
                            loss = tf.keras.losses.categorical_crossentropy(target, model(gen_img))
                            grad = t.gradient(loss, gen_img)
                            fol = np.linalg.norm(grad.numpy())  # L2 adaption

                        distance = np.linalg.norm(gen_img.numpy() - orig_img) / orig_norm
                        if fol > folMAX and distance < ep:
                            folMAX = fol
                            img_list.append(tf.identity(gen_img))

                        preds = model(gen_img).numpy()
                        gen_index = np.argmax(preds[0])
                        if orig_index == np.argmax(labels[idx]) and gen_index != orig_index:
                            gini = 1 - np.sum(np.square(preds))
                            new_context = Context(model=model, ae=gen_img, ae_label=gen_index, k=20)
                            total_sets.append((idx+batch_size*i, time.time()-time_start, fol, gen_img.numpy(), labels[idx], gini, new_context.similarity(), new_context.cc(), new_context.sample_prediction()))

                    print(f"---------------------Time: {time.time()-time_start}, {idx + i*batch_size}, {len(total_sets)}----------------------")
                    if time.time() - time_start > budget:
                        print(f"Current length of total_sets: {len(total_sets)}")
                        idx_all = np.array([item[0] for item in total_sets])
                        time_all = np.array([item[1] for item in total_sets])
                        fol_all = np.array([item[2] for item in total_sets])
                        ae_all = np.array([item[3][0] for item in total_sets])
                        label_all = np.array([item[4] for item in total_sets])
                        gini_all = np.array([item[5] for item in total_sets])
                        similarity_all = np.array([item[6] for item in total_sets])
                        cc_all = np.array([item[7] for item in total_sets])
                        sample_prediction_all = np.array([item[8] for item in total_sets])
                        np.savez(ae_data_path + '/train_ae.npz', idx=idx_all, time=time_all, ae=ae_all, ae_label=label_all, fol=fol_all, gini=gini_all,
                                 similarity=np.array(similarity_all), cc=np.array(cc_all), sample_prediction=np.array(sample_prediction_all))
                        break
                if time.time()-time_start > budget:
                    break
            if time.time()-time_start > budget:
                break