from keras.layers import *
from keras import backend as K
import tensorflow as tf
from keras.initializers import Zeros, glorot_normal, glorot_uniform


class FM(Layer):
    """Factorization Machine models pairwise (order-2) feature interactions
     without linear term and bias.

      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.

      Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.

      References
        - [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
    """

    def __init__(self, **kwargs):

        super(FM, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError("Unexpected inputs dimensions % d,\
                             expect to be 3 dimensions" % (len(input_shape)))

        super(FM, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):

        if K.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions"
                % (K.ndim(inputs)))

        concated_embeds_value = inputs

        square_of_sum = tf.square(tf.reduce_sum(
            concated_embeds_value, axis=1, keep_dims=True))
        sum_of_square = tf.reduce_sum(
            concated_embeds_value * concated_embeds_value, axis=1, keep_dims=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * tf.reduce_sum(cross_term, axis=2, keep_dims=False)

        return cross_term

    def compute_output_shape(self, input_shape):
        return (None, 1)


class CrossNet(Layer):

    def __init__(self, num_layers=3, **kwargs):
        self.num_layers = num_layers
        super(CrossNet, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        dim = input_shape[-1]
        self.kernels = [self.add_weight(name='kernel' + str(i),
                                        shape=(dim, 1),
                                        initializer=glorot_normal(),
                                        trainable=True) for i in range(self.num_layers)]
        self.bias = [self.add_weight(name='bias' + str(i),
                                     shape=(dim, 1),
                                     initializer=Zeros(),
                                     trainable=True) for i in range(self.num_layers)]
        super(CrossNet, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        if K.ndim(inputs) != 2:
            raise ValueError("Unexpected inputs dimensions %d, expect to be 2 dimensions" % (K.ndim(inputs)))
        x_0 = tf.expand_dims(inputs, axis=-1)  # None * k * 1
        x_l = x_0
        for i in range(self.num_layers):
            temp = tf.tensordot(x_l, self.kernels[i], axes=(1, 0))  # xl * w, None * 1 * 1
            temp = tf.matmul(x_0, self.kernels[i])  # x0 * xl * w, None * k * 1
            x_l = temp + x_l + self.bias[i]
        x_l = tf.squeeze(x_l)
        return x_l

    def compute_output_shape(self, input_shape):
        return input_shape
