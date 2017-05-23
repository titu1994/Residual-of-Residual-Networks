from keras.models import Model
from keras.layers import Input, Add, Activation, Dropout, Flatten, Dense
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K


def initial_conv(input):
    x = Convolution2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(input)

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    return x


def conv1_block(input, dropout=0.0, initial=False):
    init = input

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    # Check if input number of filters is same as 16, else create convolution2d for this input
    if initial:
        if K.image_data_format() == "th":
            init = Convolution2D(16, (1, 1), kernel_initializer='he_normal', padding='same')(init)
        else:
            init = Convolution2D(16, (1, 1), kernel_initializer='he_normal', padding='same')(init)

    x = Convolution2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(input)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = Convolution2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)

    m = Add()([init, x])
    x = Activation('relu')(m)
    return x


def conv2_block(input, dropout=0.0):
    init = input

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    # Check if input number of filters is same as 32, else create convolution2d for this input
    if K.image_data_format() == "channels_first":
        if init._keras_shape[1] != 32:
            init = Convolution2D(32, (1, 1), kernel_initializer='he_normal', padding='same')(init)
    else:
        if init._keras_shape[-1] != 32:
            init = Convolution2D(32, (1, 1), kernel_initializer='he_normal', padding='same')(init)

    x = Convolution2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(input)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = Convolution2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)

    m = Add()([init, x])
    x = Activation('relu')(m)
    return x


def conv3_block(input, dropout=0.0, is_last=False):
    global count
    init = input

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    # Check if input number of filters is same as 64, else create convolution2d for this input
    if K.image_data_format() == "channels_first":
        if init._keras_shape[1] != 64:
            init = Convolution2D(64, (1, 1), kernel_initializer='he_normal', padding='same')(init)
    else:
        if init._keras_shape[-1] != 64:
            init = Convolution2D(64, (1, 1), kernel_initializer='he_normal', padding='same')(init)

    x = Convolution2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(input)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = Convolution2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)

    m = Add()([init, x])
    if not is_last:
        m = Activation('relu')(m)
    return m


def create_residual_of_residual(input_dim, nb_classes=100, N=2, dropout=0.0, verbose=1):
    """
    Creates a Residual Network of Residual Network with specified parameters

    Example : To create a RoR-3-110 model for CIFAR-10:
              model = create_pre_residual_of_residual((3, 32, 32), 10, N=2)

              Note : The ResNet 101 model is the RoR-3-110 model

    :param input: Input Keras object
    :param nb_classes: Number of output classes
    :param N: Depth of the network. Compute n = 6 * (N * 9 - 1) + 8.
              Example1: For a depth of 56, N = 1, n = 6 * (1 * 9 - 1) + 8 = 56
              Example2: For a depth of 110, N = 2, n = 6 * (2 * 9 - 1) + 8 = 110
              Example3: For a depth of 164, N = 3, N = 6 * (3 * 9 - 1) + 8 = 164
    :param dropout: Adds dropout if value is greater than 0.0.
                    Note : Generally not used in RoR
    :param verbose: Debug info to describe created WRN
    :return:
    """
    ip = Input(shape=input_dim)

    x = initial_conv(ip)
    nb_conv = 8

    conv0_level1_shortcut = Convolution2D(64, (1, 1), kernel_initializer='he_normal', padding='same', strides=(4, 4),
                                          name='conv0_level1_shortcut')(x)

    conv1_level2_shortcut = Convolution2D(16, (1, 1), kernel_initializer='he_normal', padding='same',
                                          name='conv1_level2_shortcut')(x)
    for i in range(N * 9 - 1):
        initial = (i == 0)
        x = conv1_block(x, dropout, initial=initial)
        nb_conv += 2

    # Add Level 2 shortcut
    x = Add()([x, conv1_level2_shortcut])
    x = Activation('relu')(x)

    x = MaxPooling2D((2, 2))(x)

    conv2_level2_shortcut = Convolution2D(32, (1, 1), kernel_initializer='he_normal', padding='same',
                                          name='conv2_level2_shortcut')(x)
    for i in range(N * 9 - 1):
        x = conv2_block(x, dropout)
        nb_conv += 2

    # Add Level 2 shortcut
    x = Add()([x, conv2_level2_shortcut])
    x = Activation('relu')(x)

    x = MaxPooling2D((2, 2))(x)

    conv3_level2_shortcut = Convolution2D(64, (1, 1), kernel_initializer='he_normal', padding='same',
                                          name='conv3_level2_shortcut')(x)
    for i in range(N * 9 - 1):
        is_last = (i == N - 1)
        x = conv3_block(x, dropout, is_last=is_last)
        nb_conv += 2

    # Add Level 2 shortcut
    x = Add()([x, conv3_level2_shortcut])

    # Add Level 1 shortcut
    x = Add()([x, conv0_level1_shortcut])
    x = Activation('relu')(x)

    x = AveragePooling2D((8, 8))(x)
    x = Flatten()(x)

    x = Dense(nb_classes, activation='softmax')(x)

    model = Model(ip, x)

    if verbose: print("Residual-in-Residual-Network-%d created." % (nb_conv))
    return model


if __name__ == '__main__':
    model = create_residual_of_residual((3, 32, 32), 10, N=2)

    conv_count = 0
    for layer in model.layers:
        if 'conv' in layer.name:
            conv_count += 1

    print('Number of convolution layers : ', conv_count)
