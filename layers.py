import tensorflow as tf


def lastd(layer):
    return layer.get_shape().as_list()[-1]


def reshape2d(layer):
    shape = layer.get_shape()
    shape = [-1, shape[1:].num_elements()]
    return tf.reshape(layer, shape)


def Weights(*shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def Bias(b, n):
    return tf.Variable(tf.constant(b, shape=[n]))


def Connected(layer, n, b=0.0, activation=None):
    m = lastd(layer)
    layer = tf.matmul(layer, Weights(m, n)) + Bias(b, n)
    return layer if activation == None else activation(layer)


def Convolved(layer, side, depth, stride=1, padding='SAME', b=None, activation=None):
    strides = [1, stride, stride, 1]
    W = Weights(side, side, lastd(layer), depth)
    C = tf.nn.conv2d(layer, W, strides, padding)
    B = Weights(depth) if b == None else Bias(b, depth)
    layer = C + B

    return layer if activation == None else activation(layer)


def MaxPooled(layer, ksize, stride=None, padding='SAME', activation=None):
    if stride == None:
        stride = ksize

    layer = tf.nn.max_pool(layer, [1, ksize, ksize, 1], [1, stride, stride, 1], padding)
    return layer if activation == None else activation(layer)


def mini_inception_architecture(input_shape, depth, stride, n_hidden, n_classes, nonlinear=tf.nn.relu):
    inputs = tf.placeholder(tf.float32, (None,) + input_shape)

    def mini_inception_module(layer):
        conv_1x1 = Convolved(layer, 1, depth, stride)
        conv_3x3 = Convolved(layer, 3, depth, stride)
        conv_5x5 = Convolved(layer, 5, depth, stride)
        max_pool = MaxPooled(layer, 3, stride)

        inception = nonlinear(tf.concat(3, [conv_1x1, conv_3x3, conv_5x5, max_pool]))
        return nonlinear(Convolved(inception, 1, depth))

    outputs = mini_inception_module(inputs)
    outputs = mini_inception_module(outputs)

    outputs = Connected(reshape2d(outputs), n_hidden, 1.0, nonlinear)
    outputs = Connected(outputs, n_classes, 1.0)

    return (inputs, outputs)
