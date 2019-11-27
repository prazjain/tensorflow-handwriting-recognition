#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

# import mnist data
from tflow import input_data
import pathlib
import matplotlib.pyplot as plot


def load_handwritten_digit_image(filename):
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.rgb_to_grayscale(img)
    img = tf.image.crop_to_bounding_box(img, 0, 0, 560, 560)
    img = tf.image.resize(img, [28, 28])
    img = tf.squeeze(img)
    img = 1 - img
    # plot.imshow(img)
    # plot.show()
    img = tf.reshape(img, [784])
    return image_tensor, int(pathlib.Path(filename).stem)


# store mnist data in /tmp/data
mnist = input_data.read_data_sets("mnist_data/", one_hot=True)

# read 5000 digit images from mnist db, and use this to train
training_digits, training_labels = mnist.train.next_batch(5000)

# read next 200 digit images from mnist db, and use this for test
test_digits, test_labels = mnist.test.next_batch(10)

accuracy = 0.
print('---------------------------------------------------------')
print('Predicting 20 digits after learning from 5000 handwritten digits database')
print('---------------------------------------------------------')
# loop over all test digits from MNIST database
for i in range(len(test_digits)):

    # multi dimensional matrix is flattened into 1D array, we will apply L1 Distance/ Manhattan distance on this array
    l1_distance = tf.abs(tf.add(training_digits, tf.negative(test_digits[i, :])))

    # now sum all values from this vector to produce a scalar value
    distance = tf.reduce_sum(l1_distance, axis=1)

    # nearest neighbour is the one where distance is minimum
    pred = tf.argmin(distance, 0)

    nn_index = pred.numpy()

    # labels are in one-hot notation, so we can use argmax to get them
    print('Test :', i, ' Prediction: ', np.argmax(training_labels[nn_index]), ' True Label: ',
          np.argmax(test_labels[i]))

    # for every correct prediction we add up accuracy
    if np.argmax(training_labels[nn_index]) == np.argmax(test_labels[i]):
        accuracy += 1. / len(test_digits)

print('---------------------------------------------------------')
print('Now Predicting my handwritten digits from trained labels')
print('---------------------------------------------------------')
handwritten_accuracy = 0.
count = 0
matches = 0
# Now lets check my hand written digits and test them against the trained model.
for file in pathlib.Path('./test').glob('*.jpg'):
    if file.is_file():
        image_tensor, label = load_handwritten_digit_image(str(file))
        count += 1

        l1_distance = tf.abs(tf.add(training_digits, tf.negative(image_tensor)))

        # now sum all values from this vector to produce a scalar value
        distance = tf.reduce_sum(l1_distance, axis=1)

        # nearest neighbour is the one where distance is minimum
        pred = tf.argmin(distance, 0)
        nn_index = pred.numpy()
        trained_label = np.argmax(training_labels[nn_index])
        print('Prediction :', trained_label, ', Handwritten label :', label)
        if trained_label == label:
            matches += 1

handwritten_accuracy = matches / count
print('Done')
print('Accuracy :', accuracy)
print('Handwritten accuracy :', handwritten_accuracy)
