from keras.layers import Input, Conv2D, MaxPooling2D
from keras import backend as K

def nn_base(input_tensor=None, trainable=False):
    input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='vgg_block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='vgg_block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='vgg_block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='vgg_block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='vgg_block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='vgg_block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='vgg_block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='vgg_block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='vgg_block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='vgg_block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='vgg_block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='vgg_block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='vgg_block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='vgg_block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='vgg_block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='vgg_block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='vgg_block5_conv3')(x)

    return x
