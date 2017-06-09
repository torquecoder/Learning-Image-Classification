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

# Number of color channels in an image
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
    return tf.Variable(tf.constant(0.05, shape = [length]))


def new_conv_layer(input,              # The previous layer
               num_input_channels,     # Number of channels in previous layer
               filter_size,            # Width and height of each filter
               num_filters,            # Number of filters
               use_pooling = True):    # Use 2x2 max-pooling

    # Shape of the filter-weights for the convolution
    # This format is determined by the TensorFlow API
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape
    weights = new_weights(shape = shape)

    # Create new biases, one for each filter
    biases = new_biases(length = num_filters)

    # Create the TensorFlow operation for convolution
    # Note the strides are set to 1 in all dimensions
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same
    layer = tf.nn.conv2d(input = input, filter = weights, strides=[1, 1, 1, 1], padding = 'SAME')

    # Add the biases to the results of the convolution
    # A bias-value is added to each filter-channel
    layer += biases

    # Use pooling to down-sample the image resolution
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize = [1, 2, 2, 1],       # Size of max-pooling window (2x2)
                               strides = [1, 2, 2, 1],     # stride on a single image (2x2)
                               padding = 'SAME')

    # Rectified Linear Unit (ReLU)  (Some alien name as said by Siraj Raval :P)
    # It calculates max(x, 0) for each input pixel x
    # This adds some non-linearity to the formula and allows us to learn more complicated functions
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights


def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()

    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features

def new_fc_layer(input,             # The previous layer
                 num_inputs,        # Number of inputs from previous layer
                 num_outputs,       # Number of outputs
                 use_relu = True):  # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape = [num_inputs, num_outputs])
    biases = new_biases(length = num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

session = tf.Session()
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)
