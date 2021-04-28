
from keras import Sequential, optimizers
from keras.models import Input, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, Activation, Dropout


def build(input_shape):
    """
    Args:
        input_shape : number of features in the dataframe.

    Returns:
        model (keras Model): a compiled Model object.
    """
    # Since I am working with frames and their features per frame
    # I used a fully connected NN with 2 hidden layers

    model = Sequential(
        [
            Input(shape=(input_shape,)),
            Dense(120, use_bias=True),
            BatchNormalization(center=True, scale=False),
            Activation('relu'),
            Dropout(0.3),
            Dense(60, use_bias=True),
            BatchNormalization(center=True, scale=False),
            Activation('relu'),
            Dropout(0.3),
            Dense(3, activation='softmax')
         ])
    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=0.01),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model

