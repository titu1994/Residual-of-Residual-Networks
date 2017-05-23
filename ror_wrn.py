from keras.models import Model
from keras.layers import Input, Concatenate, Add, Activation, Dropout, Flatten, Dense
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K


def initial_conv(input):
    x = Convolution2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(input)
    return x


def conv1_block(input, k=1, dropout=0.0, initial=False):
    init = input

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    # Check if input number of filters is same as 16 * k, else create convolution2d for this input
    if initial:
        if K.image_data_format() == "channels_first":
            init = Convolution2D(16 * k, (1, 1), kernel_initializer='he_normal', padding='same')(init)
        else:
            init = Convolution2D(16 * k, (1, 1), kernel_initializer='he_normal', padding='same')(init)

    x = BatchNormalization(axis=channel_axis)(input)
    x = Activation('relu')(x)
    x = Convolution2D(16 * k, (3, 3), padding='same', kernel_initializer='he_normal')(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    x = Convolution2D(16 * k, (3, 3), padding='same', kernel_initializer='he_normal')(x)

    m = Add()([init, x])
    return m


def conv2_block(input, k=1, dropout=0.0):
    init = input

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    # Check if input number of filters is same as 32 * k, else create convolution2d for this input
    if K.image_data_format() == "channels_first":
        if init._keras_shape[1] != 32 * k:
            init = Convolution2D(32 * k, (1, 1), kernel_initializer='he_normal', padding='same')(init)
    else:
        if init._keras_shape[-1] != 32 * k:
            init = Convolution2D(32 * k, (1, 1), kernel_initializer='he_normal', padding='same')(init)

    x = BatchNormalization(axis=channel_axis)(input)
    x = Activation('relu')(x)
    x = Convolution2D(32 * k, (3, 3), padding='same', kernel_initializer='he_normal')(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    x = Convolution2D(32 * k, (3, 3), padding='same', kernel_initializer='he_normal')(x)

    m = Add()([init, x])
    return m


def conv3_block(input, k=1, dropout=0.0):
    init = input

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    # Check if input number of filters is same as 64 * k, else create convolution2d for this input
    if K.image_data_format() == "channels_first":
        if init._keras_shape[1] != 64 * k:
            init = Convolution2D(64 * k, (1, 1), kernel_initializer='he_normal', padding='same')(init)
    else:
        if init._keras_shape[-1] != 64 * k:
            init = Convolution2D(64 * k, (1, 1), kernel_initializer='he_normal', padding='same')(init)

    x = BatchNormalization(axis=channel_axis)(input)
    x = Activation('relu')(x)
    x = Convolution2D(64 * k, (3, 3), kernel_initializer='he_normal', padding='same')(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    x = Convolution2D(64 * k, (3, 3), kernel_initializer='he_normal', padding='same')(x)

    m = Add()([init, x])
    return m


def create_pre_residual_of_residual(input_dim, nb_classes=100, N=2, k=1, dropout=0.0, verbose=1):
    """
    Creates a Residual Network of Residual Network with specified parameters

    Example : To create a Pre-RoR model, use k = 1
              model = create_pre_residual_of_residual((3, 32, 32), 10, N=4, k=1) # Pre-RoR-3

              To create a RoR-WRN model, use k > 1
              model = create_pre_residual_of_residual((3, 32, 32), 10, N=4, k=10) # RoR-3-WRN-28-10


    :param input: Input Keras object
    :param nb_classes: Number of output classes
    :param N: Depth of the network. Compute N = (n - 4) / 6.
              Example : For a depth of 16, n = 16, N = (16 - 4) / 6 = 2
              Example2: For a depth of 28, n = 28, N = (28 - 4) / 6 = 4
              Example3: For a depth of 40, n = 40, N = (40 - 4) / 6 = 6
    :param k: Width of the network.
    :param dropout: Adds dropout if value is greater than 0.0.
                    Note : Generally not used in RoR
    :param verbose: Debug info to describe created WRN
    :return:
    """
    ip = Input(shape=input_dim)

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    x = initial_conv(ip)
    nb_conv = 4 # Dont count 4 long residual connections in WRN models

    conv0_level1_shortcut = Convolution2D(64 * k, (1, 1), padding='same', strides=(4, 4),
                                          name='conv0_level1_shortcut')(x)

    conv1_level2_shortcut = Convolution2D(16 * k, (1, 1), padding='same',
                                          name='conv1_level2_shortcut')(x)
    for i in range(N):
        initial = (i == 0)
        x = conv1_block(x, k, dropout, initial=initial)
        nb_conv += 2

    # Add Level 2 shortcut
    x = Add()([x, conv1_level2_shortcut])

    x = MaxPooling2D((2, 2))(x)

    conv2_level2_shortcut = Convolution2D(32 * k, (1, 1), padding='same',
                                          name='conv2_level2_shortcut')(x)
    for i in range(N):
        x = conv2_block(x, k, dropout)
        nb_conv += 2

    # Add Level 2 shortcut
    x = Add()([x, conv2_level2_shortcut])

    x = MaxPooling2D((2, 2))(x)

    conv3_level2_shortcut = Convolution2D(64 * k, (1, 1), padding='same',
                                          name='conv3_level2_shortcut')(x)
    for i in range(N):
        x = conv3_block(x, k, dropout)
        nb_conv += 2

    # Add Level 2 shortcut
    x = Add()([x, conv3_level2_shortcut])

    # Add Level 1 shortcut
    x = Add()([x, conv0_level1_shortcut])
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x = AveragePooling2D((8, 8))(x)
    x = Flatten()(x)

    x = Dense(nb_classes, activation='softmax')(x)

    model = Model(ip, x)

    if verbose: print("Residual-in-Residual-Network-%d-%d created." % (nb_conv, k))
    return model

if __name__ == "__main__":
    model = create_pre_residual_of_residual((3, 32, 32), 10, N=6, k=2)

    conv_count = 0
    for layer in model.layers:
        if 'conv' in layer.name:
            conv_count += 1

    print('Number of convolution layers : ', conv_count)