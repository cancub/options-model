# Tensorflow/Keras
import tensorflow as tf
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import (Add, Conv2D, Dense, Flatten, Input,
                                     Lambda, Subtract)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop

def build_q_network(
    n_actions,
    input_length,
    history_length=4,
    learning_rate=0.00001
):
    '''
    Builds a dueling DQN as a Keras model

    Arguments:
        n_actions: Number of possible action the agent can take

        input_shape: Shape of the options data the model sees

        history_length: Number of historical timepoints the agent can see

        learning_rate: Learning rate

    Returns:
        A compiled Keras model
    '''
    model_input = Input(shape=(input_length * history_length, 1))

    x = model_input

    for _ in range(3):
        x = Dense(
            32,
            kernel_initializer=VarianceScaling(scale=2.),
            activation='relu'
        )(x)

    # Split into value and advantage streams
    val_stream, adv_stream = Lambda(
        lambda w: tf.split(w, 2, 3))(x)  # custom splitting layer

    val_stream = Flatten()(val_stream)
    val = Dense(
        1,
        kernel_initializer=VarianceScaling(scale=2.)
    )(val_stream)

    adv_stream = Flatten()(adv_stream)
    adv = Dense(
        n_actions,
        kernel_initializer=VarianceScaling(scale=2.)
    )(adv_stream)

    # Combine streams into Q-Values
    reduce_mean = Lambda(lambda w: tf.reduce_mean(
        w, axis=1, keepdims=True))  # custom layer for reduce mean

    q_vals = Add()([val, Subtract()([adv, reduce_mean(adv)])])

    # Build model
    model = Model(model_input, q_vals)
    model.compile(Adam(learning_rate), loss=tf.keras.losses.Huber())

    return model
