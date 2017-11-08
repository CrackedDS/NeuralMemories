from keras.layers import *
from keras.models import Model

from .SpatialMemory import NeuralMap, NeuralMapCell

def NeuralNNModel(s_t_size, feature_size, nb_actions):
    '''
    Architecture of the MQN.
    Initialize by calling:
    model = NeuralNNmodel(s_t_size, feature_size, nb_actions)
    where s_t_size is the dimension of the
    encoding of the convolutional layer, and
    feature_size is the dimension of the
    memory output/operating size of the memory.
    nb_actions is the number of actions
    in the environment.
    '''
    input_layer = Input((1,None,None))

    provider = Conv2D(filters=3, kernel_size=(1,1), padding="same", data_format="channels_first")(input_layer)
    provider = Conv2D(filters=32, kernel_size=(2,2), strides=(2,2), padding="valid", data_format="channels_first")(provider)
    provider = Conv2D(filters=64, kernel_size=(2,2), strides=(1,1), padding="valid", data_format="channels_first")(provider)
    provider = GlobalMaxPooling2D(data_format="channels_first")(provider)
    s = Dense(s_t_size)(provider)
    s = Dropout(rate=0.5)(s)

    memory = NeuralMap(feature_size, memory_size=[15,15])(s)

    output_layer = Dense(256, activation="softmax")(memory)
    output_layer = Dropout(rate=0.5)(output_layer)
    output_layer = Dense(128, activation="softmax")(output_layer)
    output_layer = Dense(64, activation="softmax")(output_layer)
    output_layer = Dense(nb_actions, activation="softmax")(output_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model