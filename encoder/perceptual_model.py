import PIL.Image
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input
import keras.backend as K


def load_images(images_list, img_size):
    loaded_images = list()
    for img_path in images_list:
      img = PIL.Image.open(img_path).convert('RGB').resize((img_size,img_size),PIL.Image.LANCZOS)
      img = np.array(img)
      img = np.expand_dims(img, 0)
      loaded_images.append(img)
    loaded_images = np.vstack(loaded_images)
    return loaded_images

class PerceptualModel:
    def __init__(self, args, batch_size=1, sess=None):
        self.sess = tf.get_default_session() if sess is None else sess
        K.set_session(self.sess)
        self.img_size = args.image_size
        self.vgg_loss = args.use_vgg_loss
        self.pixel_loss = args.use_pixel_loss
        self.lr = args.lr
        self.decay_steps = args.decay_steps
        self.decay_rate = args.decay_rate
        self.batch_size = batch_size
        self.layer = 9

        self.perceptual_model = None
        self.ref_img_features = None
        self.features_weight = None
        self.loss = None

    def build_perceptual_model(self, generator):
        self.loss = 0
        # Learning rate
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")
        incremented_global_step = tf.assign_add(global_step, 1)
        self._reset_global_step = tf.assign(global_step, 0)
        self.learning_rate = tf.train.exponential_decay(self.lr, incremented_global_step,
                self.decay_steps, self.decay_rate, staircase=True)
        self.sess.run([self._reset_global_step])

        generated_image = tf.image.resize_nearest_neighbor(generator.generated_image, (self.img_size, self.img_size), align_corners=True)
        self.ref_img_features = tf.get_variable('ref_img_features', shape=generated_image.shape,
                                                dtype='float32', initializer=tf.initializers.zeros())
        self.features_weight = tf.get_variable('features_weight', shape=generated_image.shape,
                                               dtype='float32', initializer=tf.initializers.zeros())
        self.sess.run([self.features_weight.initializer, self.features_weight.initializer])
        vgg16 = VGG16(include_top=False, input_shape=(self.img_size, self.img_size, 3))
        self.perceptual_model = Model(vgg16.input, vgg16.layers[self.layer].output)
        generated_img_features = self.perceptual_model(preprocess_input(self.features_weight * generated_image))
        self.ref_img_features_vgg = tf.get_variable('ref_img_features_vgg', shape=generated_img_features.shape,
                                                dtype='float32', initializer=tf.initializers.zeros())
        self.features_weight_vgg = tf.get_variable('features_weight_vgg', shape=generated_img_features.shape,
                                               dtype='float32', initializer=tf.initializers.zeros())
        self.sess.run([self.features_weight_vgg.initializer, self.ref_img_features_vgg.initializer])
        self.loss += self.vgg_loss * tf.reduce_mean(tf.abs(self.features_weight_vgg * self.ref_img_features_vgg - self.features_weight_vgg * generated_img_features), axis=None)
        self.loss += self.pixel_loss * tf.reduce_mean(tf.keras.losses.logcosh(self.features_weight * self.ref_img_features, self.features_weight * generated_image))


    def set_reference_images(self, images_list):
        assert(len(images_list) != 0 and len(images_list) <= self.batch_size)
        loaded_image = load_images(images_list, self.img_size)
        if self.perceptual_model is not None:
            image_features = self.perceptual_model.predict_on_batch(preprocess_input(loaded_image))
            weight_mask = np.ones(self.features_weight_vgg.shape)
            self.sess.run(tf.assign(self.features_weight_vgg, weight_mask))
            self.sess.run(tf.assign(self.ref_img_features_vgg, image_features))
        # in case if number of images less than actual batch size
        # can be optimized further
        weight_mask = np.ones(self.features_weight.shape)
        if len(images_list) != self.batch_size:
            features_space = list(self.features_weight.shape[1:])
            existing_features_shape = [len(images_list)] + features_space
            empty_features_shape = [self.batch_size - len(images_list)] + features_space

            existing_examples = np.ones(shape=existing_features_shape)
            empty_examples = np.zeros(shape=empty_features_shape)
            weight_mask = np.vstack([existing_examples, empty_examples])

            loaded_image = np.vstack([loaded_image, np.zeros(empty_features_shape)])

        self.sess.run(tf.assign(self.features_weight, weight_mask))
        self.sess.run(tf.assign(self.ref_img_features, loaded_image))

    def optimize(self, vars_to_optimize, iterations=500):
        vars_to_optimize = vars_to_optimize if isinstance(vars_to_optimize, list) else [vars_to_optimize]
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        min_op = optimizer.minimize(self.loss, var_list=[vars_to_optimize])
        self.sess.run(tf.variables_initializer(optimizer.variables()))
        for _ in range(iterations):
            _, loss = self.sess.run([min_op, self.loss])
            yield {"loss":loss}

