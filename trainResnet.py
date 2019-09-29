import cv2
import pickle
import numpy as np
from dnnlib import tflib
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, Reshape, Dense
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input


tflib.init_tf()

def generate_dataset(n=50000, seed=None, image_size=256, minibatch_size=16):
    with open("./model/stylegan.pkl")as f:
        generator_network, discriminator_network, Gs_network = pickle.load(f)
    if seed is not None:
        latents = np.random.RandomState(seed).randn(n, Gs_network.input_shape[1])
    else:
        latents = np.random.randn(n, Gs_network.input_shape[1])
    dlatents = Gs_network.components.mapping.run(latents, None, minibatch_size=minibatch_size)
    images = Gs_network.components.synthesis.run(dlatents, randomize_noise = False, minibatch_size = minibatch_size, print_progress = True, output_transform = dict(func = tflib.convert_images_to_uint8, nchw_to_nhwc = True))
    images = np.array([cv2.resize(image, (image_size, image_size), interpolation = cv2.INTER_AREA) for image in images])
    images = preprocess_input(images)
    return dlatents, images


def createModel(image_size=256):
    resnet = ResNet50(include_top=False, pooling=None, weights='imagenet', input_shape=(image_size, image_size, 3))
    input_layer = Input(shape=(image_size, image_size, 3))
    layer = resnet(input_layer)
    layer = Conv2D(144, 1, activation="elu")(layer)
    layer = Reshape((18, 512))(layer)
    model = Model(inputs=input_layer, outputs=layer)
    return model


def data_generator(data, targets, batch_size):
    batches = (len(data) + batch_size - 1) // batch_size
    while (True):
        for i in range(batches):
            X = data[i * batch_size: (i + 1) * batch_size]
            Y = targets[i * batch_size: (i + 1) * batch_size]
            yield (X, Y)


def train(seed=0, num=10000, model_path='./model/resnet50.h5', freeze_first=True, batch_size=16):
    # Iterate on batches of size batch_size
    print('Generating training set:')
    W_train, X_train = generate_dataset(num, image_size = 256, seed = seed, minibatch_size = 16)
    model = createModel(image_size = 256)
    if freeze_first:
        model.layers[1].trainable = False
        model.compile(loss = "logcosh", optimizer = "adam", metrics = [])
    model.fit_generator(generator = data_generator(X_train, W_train, batch_size), steps_per_epoch = (num + batch_size - 1) // batch_size, epochs = 100, verbose = True)
    print('Saving model.')
    model.save(model_path)
    model.layers[1].trainable = True
    model.compile(loss = "logcosh", optimizer = "adam", metrics = [])
    W_train, X_train = generate_dataset(num, image_size = 256, seed = seed, minibatch_size = 16)
    model.fit(X_train, W_train, epochs = 100, verbose = True, batch_size = batch_size)
    print('Saving model.')
    model.save(model_path)


if __name__ == '__main__':
    train(seed=1)