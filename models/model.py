import tensorflow.keras.regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, MaxPooling2D, Conv2D, BatchNormalization, \
    GlobalAveragePooling2D, Input
from tensorflow.keras.applications import ResNet50, ResNet50V2, VGG16, InceptionResNetV2, MobileNetV2


# Resnet 50
def resnet_50(input_tensor, input_shape, weights):
    base = ResNet50(weights=weights, input_tensor=input_tensor, include_top=False, input_shape=input_shape)
    base.trainable = False
    return base


def resnet_50_v2(input_tensor, input_shape, weights):
    base = ResNet50V2(weights=weights, input_tensor=input_tensor, include_top=False, input_shape=input_shape)
    base.trainable = False
    return base


def vgg_16(input_tensor, input_shape, weights):
    base = VGG16(weights=weights, input_tensor=input_tensor, include_top=False, input_shape=input_shape)
    base.trainable = False
    return base


def inception_res(input_tensor, input_shape, weights):
    base = InceptionResNetV2(weights=weights, input_tensor=input_tensor, include_top=False, input_shape=input_shape)
    base.trainable = False
    return base


def mobilenet(input_tensor, input_shape, num_classes):
    base = MobileNetV2(weights="imagenet",
                       input_shape=input_shape,
                       input_tensor=input_tensor,
                       include_top=False)
    base.trainable = False
    model = Sequential()
    model.add(base)
    model.add(GlobalAveragePooling2D())
    # model.add(Dense(128, activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(64, activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def my_net(num_classes, input_shape):
    model = Sequential(name='DCNN')
    model.add(Conv2D(filters=64, kernel_size=(5, 5), input_shape=input_shape, activation='elu', padding='same',
                     kernel_initializer='he_normal',
                     name='conv2d_1'))
    model.add(BatchNormalization(name='batchnorm_1'))
    model.add(
        Conv2D(filters=64, kernel_size=(5, 5), activation='elu', padding='same', kernel_initializer='he_normal',
               name='conv2d_2'))
    model.add(BatchNormalization(name='batchnorm_2'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='maxpool2d_1'))
    model.add(Dropout(0.4, name='dropout_1'))
    model.add(
        Conv2D(filters=128, kernel_size=(3, 3), activation='elu', padding='same', kernel_initializer='he_normal',
               name='conv2d_3'))
    model.add(BatchNormalization(name='batchnorm_3'))
    model.add(
        Conv2D(filters=128, kernel_size=(3, 3), activation='elu', padding='same', kernel_initializer='he_normal',
               name='conv2d_4'))
    model.add(BatchNormalization(name='batchnorm_4'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='maxpool2d_2'))
    model.add(Dropout(0.4, name='dropout_2'))
    model.add(
        Conv2D(filters=256, kernel_size=(3, 3), activation='elu', padding='same', kernel_initializer='he_normal',
               name='conv2d_5'))
    model.add(BatchNormalization(name='batchnorm_5'))
    model.add(
        Conv2D(filters=256, kernel_size=(3, 3), activation='elu', padding='same', kernel_initializer='he_normal',
               name='conv2d_6'))
    model.add(BatchNormalization(name='batchnorm_6'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='maxpool2d_3'))
    model.add(Dropout(0.5, name='dropout_3'))
    model.add(Flatten(name='flatten'))
    model.add(Dropout(0.2, name='dropout_4'))
    model.add(Dense(128, activation='elu', kernel_initializer='he_normal', name='dense_1'))
    model.add(BatchNormalization(name='batchnorm_7'))
    model.add(Dense(num_classes, activation='softmax', name='out_layer'))

    return model


def my_net2(num_classes, input_shape):
    model = Sequential()

    model.add(Conv2D(filters=16, input_shape=input_shape, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(3, 3))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    # model.add(Dense(32,activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    return model


# Feed forward
def fully_connected(num_classes):
    model = Sequential()
    model.add(Flatten())
    # model.add(BatchNormalization())
    # model.add(Dense(2048, activation='relu'))
    # model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def fully_connected_2(num_classes):
    model = Sequential()
    # model.add(GlobalAveragePooling2D())
    model.add(Flatten())
    # model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax'))
    return model


def fully_connected_3(num_classes):
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def build_model(base, forward):
    model = Sequential()
    model.add(base)
    model.add(forward)
    return model


def build_net(num_classes, input_shape):
    """
    This is a Deep Convolutional Neural Network (DCNN). For generalization purpose I used dropouts in regular intervals.
    I used `ELU` as the activation because it avoids dying relu problem but also performed well as compared to LeakyRelu
    atleast in this case. `he_normal` kernel initializer is used as it suits ELU. BatchNormalization is also used for better
    results.
    """
    model = Sequential(name='DCNN')

    model.add(Conv2D(filters=64,
                     kernel_size=(5, 5),
                     input_shape=input_shape,
                     activation='relu', padding='same',
                     kernel_initializer='he_normal',
                     name='conv2d_1'
                     )
              )
    model.add(BatchNormalization(name='batchnorm_1'))
    model.add(
        Conv2D(
            filters=64,
            kernel_size=(5, 5),
            activation='relu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_2'
        )
    )
    model.add(BatchNormalization(name='batchnorm_2'))

    model.add(MaxPooling2D(pool_size=(2, 2), name='maxpool2d_1'))
    model.add(Dropout(0.4, name='dropout_1'))

    model.add(
        Conv2D(
            filters=128,
            kernel_size=(3, 3),
            activation='relu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_3'
        )
    )
    model.add(BatchNormalization(name='batchnorm_3'))
    model.add(
        Conv2D(
            filters=128,
            kernel_size=(3, 3),
            activation='relu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_4'
        )
    )
    model.add(BatchNormalization(name='batchnorm_4'))

    model.add(MaxPooling2D(pool_size=(2, 2), name='maxpool2d_2'))
    model.add(Dropout(0.4, name='dropout_2'))

    model.add(
        Conv2D(
            filters=256,
            kernel_size=(3, 3),
            activation='relu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_5'
        )
    )
    model.add(BatchNormalization(name='batchnorm_5'))
    model.add(
        Conv2D(
            filters=256,
            kernel_size=(3, 3),
            activation='relu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_6'
        )
    )
    model.add(BatchNormalization(name='batchnorm_6'))

    model.add(MaxPooling2D(pool_size=(2, 2), name='maxpool2d_3'))
    model.add(Dropout(0.5, name='dropout_3'))

    model.add(Flatten(name='flatten'))

    model.add(
        Dense(
            128,
            activation='relu',
            kernel_initializer='he_normal',
            name='dense_1'
        )
    )
    model.add(BatchNormalization(name='batchnorm_7'))

    model.add(Dropout(0.2, name='dropout_4'))

    model.add(
        Dense(
            num_classes,
            activation='softmax',
            name='out_layer'
        )
    )
    return model
