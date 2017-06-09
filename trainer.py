import tensorflow as tf
import data_helper

# Convolutional layer 1
filter_size_1 = 3
num_filters_1 = 32

# Convolutional layer 2
filter_size_1 = 3
num_filters_1 = 32

# Convolutional layer 3
filter_size_1 = 3
num_filters_1 = 64

# Fully connected layer
fc_size = 128   # Number of neurons in fully connected layer

# Number of color channels in image
num_channels = 3    #RGB

# Image dimensions
img_size = 128

# Size of image after flattening to a single dimension
img_size_flat = img_size * img_size * num_channels

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# Class info

classes = ['dogs', 'cats']
num_classes = len(classes)

# Batch size
batch_size = 16

# Validation split
validation_size = .2

# How long to wait after validation loss stops improving before terminating training
early_stopping = None  # use None if you don't want to implement early stoping

train_path = 'training_data'
test_path = 'testing_data'

data = data_helper.read_training_sets(train_path, img_size, classes, validation_size = validation_size)
test_images, test_ids = data_helper.read_testing_set(test_path, img_size, classes)

print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(test_images)))
print("- Validation-set:\t{}".format(len(data.valid.labels)))


def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev = 0.05)) # Standard Deviation = 0.05

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))
