import tensorflow as tf

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
