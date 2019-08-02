import pickle
import os

from ndf.layers import Conv1D, Linear, TanH, Dense, Input, BatchNorm, \
    Flatten
from ndf.model import Model

# location of this file
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))


def encoder():
    # load weights
    with open(os.path.join(
            __location__, "weights", "encoder_weights.pkl"), "rb") as f:
        weights = pickle.load(f)

    input = Input(shape=(120, 35))

    x = Conv1D(9, 9, name="conv1d_1", kernel_weights=weights[0], bias_weights=weights[1])(input)
    x = TanH(name="tanh_1")(x)
    x = BatchNorm(name="batchnorm1", gamma=weights[2], beta=weights[3], moving_mu=weights[18], moving_var=weights[19])(x)

    x = Conv1D(9, 9, name="conv1d_2", kernel_weights=weights[4], bias_weights=weights[5])(x)
    x = TanH(name="tanh_2")(x)
    x = BatchNorm(name="batchnorm_2", gamma=weights[6], beta=weights[7], moving_mu=weights[20], moving_var=weights[21])(
        x)

    x = Conv1D(10, 11, name="conv1d_3", kernel_weights=weights[8], bias_weights=weights[9])(x)
    x = TanH(name="tanh_3")(x)
    x = BatchNorm(name="batchnorm_3", gamma=weights[10], beta=weights[11], moving_mu=weights[22],
                  moving_var=weights[23])(x)

    x = Flatten(name="flatten")(x)

    x = Dense(name="dense_1", kernel_weights=weights[12], bias_weights=weights[13])(x)
    x = TanH(name="tanh_4")(x)

    x = BatchNorm(name="batchnorm_4", gamma=weights[14], beta=weights[15], moving_mu=weights[24],
                  moving_var=weights[25])(x)

    z_mean = Dense(name="encoder_output", kernel_weights=weights[16], bias_weights=weights[17])(x)
    #z_mean = Linear(name="linear")(z_mean)

    model = Model([input], [z_mean])
    return model