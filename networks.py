from math import ceil, floor
from sys import maxsize

import tensorflow as tf
from tqdm import tqdm


class Network(object):
    def __init__(self, architecture, *args, **kwargs):
        with tf.Graph().as_default():
            (inputs, outputs) = architecture(*args, **kwargs)
            self.inputs = inputs
            self.outputs = outputs
            self.session = tf.Session()

    def init_variables(self):
        with self.session.graph.as_default():
            init = tf.initialize_all_variables()
            self.session.run(init)


def batches(dataset, l, counter=range):
    X = dataset.X
    y = dataset.y
    n = int(ceil(len(X) / l)) # Ensures a final "rest" batch will be issued as appropriate
    for k in counter(n):
        a = k * l
        b = a + l
        yield (k, X[a:b], y[a:b])


class Accuracy(object):
    def __init__(self, network, batch_size, *datasets):
        self.inputs = network.inputs
        self.session = network.session
        with network.session.graph.as_default():
            self.outputs = tf.placeholder(tf.float32)
            predicted = tf.equal(tf.argmax(network.outputs, 1), tf.argmax(self.outputs, 1))
            self.accuracy = tf.reduce_sum(tf.cast(predicted, tf.float32))

        self.datasets = datasets
        self.batch_size = batch_size

    def __call__(self):
        def accuracy(dataset):
            total = 0.0
            count = 0.0
            for (k, X_k, y_k) in batches(dataset, self.batch_size):
                data = {self.inputs: X_k, self.outputs: y_k}
                total += self.session.run(self.accuracy, feed_dict=data)
                count += len(X_k)

            return total / count

        return tuple(accuracy(dataset) for dataset in self.datasets)


class Optimizer(object):
    def __init__(self, network, learning_rate):
        self.inputs = network.inputs
        self.session = network.session
        with self.session.graph.as_default():
            self.outputs = tf.placeholder(tf.float32)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(network.outputs, self.outputs)
            self.loss = tf.reduce_mean(cross_entropy)
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)

    def __call__(self, X, y):
        data = {self.inputs: X, self.outputs: y}
        (_, l) = self.session.run([self.optimizer, self.loss], feed_dict=data)
        return l


def load(network, path):
    session = network.session
    with session.graph.as_default():
        saver = tf.train.Saver()
        saver.restore(session, path)


def save(network, path):
    session = network.session
    with session.graph.as_default():
        saver = tf.train.Saver()
        saver.save(session, path)


def print_report(x_batch, y_train, y_val, y_loss):
    print('Batch #%d: loss %.3f, train accuracy %.3f, validation accuracy %.3f' % (x_batch[-1], y_loss[-1], y_train[-1], y_val[-1]))


def train(network, D_train, D_val, batch_size=50, batch_step=10, learning_rate=0.1, epochs=5, path='network.chk', report=print_report):
    accuracy = Accuracy(network, batch_size, D_train, D_val)
    optimizer = Optimizer(network, learning_rate)
    session = network.session

    x_batch = []
    y_train = []
    y_val = []
    y_loss = []

    network.init_variables()

    for i in range(epochs):
        counter = lambda n: tqdm(range(n), desc='Epoch {:>2}/{}'.format(i + 1, epochs), unit='batches')
        for (k, X_k, y_k) in batches(D_train, batch_size, counter):
            loss = optimizer(X_k, y_k)

            if k % batch_step == 0:
                (a_train, a_val) = accuracy()
                x_batch.append(len(x_batch) * batch_step)
                y_train.append(a_train)
                y_val.append(a_val)
                y_loss.append(loss)

        save(network, path)

        report(x_batch, y_train, y_val, y_loss)
