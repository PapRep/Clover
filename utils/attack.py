import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from Context import Context


class FGSM:
    """
    We use FGSM to generate a batch of adversarial examples. 
    """
    def __init__(self, model, ep=0.01, isRand=True):
        """
        isRand is set True to improve the attack success rate. 
        """
        self.isRand = isRand
        self.model = model
        self.ep = ep
        self.time_start = time.time()

    def generate(self, x, y, randRate=1):
        """
        x: clean inputs, shape of x: [batch_size, width, height, channel] 
        y: ground truth, one hot vectors, shape of y: [batch_size, N_classes] 
        """
        ground_truths = tf.constant(y, dtype=float)

        original_xs = x.copy()
        original_correct_idxs = np.where(np.argmax(self.model(original_xs), axis=1) == np.argmax(y, axis=1))[0]

        if self.isRand:
            x = x + np.random.uniform(-self.ep * randRate, self.ep * randRate, x.shape)
            x = np.clip(x, 0.0, 1.0)

        x = tf.Variable(x, dtype=float)
        with tf.GradientTape() as tape:
            loss = keras.losses.categorical_crossentropy(ground_truths, self.model(x))
            grads = tape.gradient(loss, x)
        delta = tf.sign(grads)

        x_aes = x + self.ep * delta
        x_aes = tf.clip_by_value(x_aes, clip_value_min=original_xs-self.ep, clip_value_max=original_xs+self.ep)
        x_aes = tf.clip_by_value(x_aes, clip_value_min=0, clip_value_max=1)

        ae_prediction_labels = np.argmax(self.model(x_aes), axis=1)
        x_aes_incorrect_idxs = np.where(ae_prediction_labels != np.argmax(y, axis=1))[0]
        idxs = np.array(list(set(original_correct_idxs).intersection(set(x_aes_incorrect_idxs))))
        print(f"The number of successful ae is {len(idxs)}, time {time.time()-self.time_start}")

        selected_original_xs, selected_x_aes, selected_ground_truths = original_xs[idxs], x_aes.numpy()[idxs], ground_truths.numpy()[idxs]
        fol_list, similarity_list, cc_list, sample_prediction_list, time_list = [], [], [], [], []
        for selected_x_ae, selected_x_ae_label in zip(selected_x_aes, ae_prediction_labels[idxs]):
            new_context = Context(self.model, ae=np.expand_dims(selected_x_ae, axis=0), ae_label=selected_x_ae_label, delta=0.01, k=20)
            similarity_list.append(new_context.similarity())
            cc_list.append(new_context.cc())
            sample_prediction_list.append(new_context.sample_prediction())
            time_list.append(time.time() - self.time_start)

        selected_x_aes_var, selected_ground_truths_var = tf.Variable(selected_x_aes), tf.constant(selected_ground_truths)
        gini_array = 1 - np.sum(np.square(self.model(selected_x_aes_var).numpy()), axis=1)
        with tf.GradientTape() as tape:
            loss = keras.losses.categorical_crossentropy(selected_ground_truths_var, self.model(selected_x_aes_var))
            grads = tape.gradient(loss, selected_x_aes_var)
            grad_norm = np.linalg.norm(grads.numpy().reshape(selected_x_aes_var.shape[0], -1), ord=1, axis=1)
            grads_flat = grads.numpy().reshape(selected_x_aes_var.shape[0], -1)
            diff = (selected_x_aes_var.numpy() - selected_original_xs).reshape(selected_x_aes_var.shape[0], -1)
            for i in range(selected_x_aes_var.shape[0]):
                i_fol = -np.dot(grads_flat[i], diff[i]) + self.ep * grad_norm[i]
                fol_list.append(i_fol)

        return np.array(idxs), selected_x_aes, selected_ground_truths, np.array(fol_list), gini_array, np.array(similarity_list), np.array(cc_list), np.array(sample_prediction_list), np.array(time_list)


class PGD:
    """
    We use PGD to generate a batch of adversarial examples. PGD could be seen as iterative version of FGSM.
    """
    def __init__(self, model, ep=0.01, step=None, epochs=10, isRand=True):
        """
        isRand is set True to improve the attack success rate. 
        """
        self.isRand = isRand
        self.model = model
        self.ep = ep
        if step is None:
            self.step = ep/6
        self.epochs = epochs
        self.time_start = time.time()
        
    def generate(self, x, y, randRate=1):
        """
        x: clean inputs, shape of x: [batch_size, width, height, channel] 
        y: ground truth, one hot vectors, shape of y: [batch_size, N_classes] 
        """
        ground_truths = tf.constant(y, dtype=float)

        original_xs = x.copy()
        original_correct_idxs = np.where(np.argmax(self.model(original_xs), axis=1) == np.argmax(y, axis=1))[0]

        if self.isRand:
            x = x + np.random.uniform(-self.ep * randRate, self.ep * randRate, x.shape)
            x = np.clip(x, 0.0, 1.0)

        x_aes = tf.Variable(x, dtype=float)
        for i in range(self.epochs):
            with tf.GradientTape() as tape:
                loss = keras.losses.categorical_crossentropy(ground_truths, self.model(x_aes))
                grads = tape.gradient(loss, x_aes)
            delta = tf.sign(grads)
            x_aes.assign_add(self.step * delta)
            x_aes = tf.clip_by_value(x_aes, clip_value_min=original_xs-self.ep, clip_value_max=original_xs+self.ep)
            x_aes = tf.clip_by_value(x_aes, clip_value_min=0.0, clip_value_max=1.0)
            x_aes = tf.Variable(x_aes)

        ae_prediction_labels = np.argmax(self.model(x_aes), axis=1)
        x_aes_incorrect_idxs = np.where(ae_prediction_labels != np.argmax(y, axis=1))[0]
        idxs = np.array(list(set(original_correct_idxs).intersection(set(x_aes_incorrect_idxs))))
        print(f"The number of successful ae is {len(idxs)}, time {time.time()-self.time_start}")

        selected_original_xs, selected_x_aes, selected_ground_truths = original_xs[idxs], x_aes.numpy()[idxs], ground_truths.numpy()[idxs]
        fol_list, similarity_list, cc_list, sample_prediction_list, time_list = [], [], [], [], []
        for selected_x_ae, selected_x_ae_label in zip(selected_x_aes, ae_prediction_labels[idxs]):
            new_context = Context(self.model, ae=np.expand_dims(selected_x_ae, axis=0), ae_label=selected_x_ae_label, delta=0.01, k=20)
            similarity_list.append(new_context.similarity())
            cc_list.append(new_context.cc())
            sample_prediction_list.append(new_context.sample_prediction())
            time_list.append(time.time() - self.time_start)

        selected_x_aes_var, selected_ground_truths_var = tf.Variable(selected_x_aes), tf.constant(selected_ground_truths)
        gini_array = 1 - np.sum(np.square(self.model(selected_x_aes_var).numpy()), axis=1)
        with tf.GradientTape() as tape:
            loss = keras.losses.categorical_crossentropy(selected_ground_truths_var, self.model(selected_x_aes_var))
            grads = tape.gradient(loss, selected_x_aes_var)
            grad_norm = np.linalg.norm(grads.numpy().reshape(selected_x_aes_var.shape[0], -1), ord=1, axis=1)
            grads_flat = grads.numpy().reshape(selected_x_aes_var.shape[0], -1)
            diff = (selected_x_aes_var.numpy() - selected_original_xs).reshape(selected_x_aes_var.shape[0], -1)
            for i in range(selected_x_aes_var.shape[0]):
                i_fol = -np.dot(grads_flat[i], diff[i]) + self.ep * grad_norm[i]
                fol_list.append(i_fol)

        return np.array(idxs), selected_x_aes, selected_ground_truths, np.array(fol_list), gini_array, np.array(similarity_list), np.array(cc_list), np.array(sample_prediction_list), np.array(time_list)
