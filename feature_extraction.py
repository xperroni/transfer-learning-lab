import pickle
import tensorflow as tf
# TODO: import Keras layers you need here

from datasets import Dataset, Data, Likelihoods
from layers import reshape2d, Connected
from networks import train, Network

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_file', '', "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', '', "Bottleneck features validation file (.p)")
flags.DEFINE_integer('breadth', 10, "Number of output classes")


def load_bottleneck_data(training_file, validation_file, breadth):
    """
    Utility function to load bottleneck features.

    Arguments:
        training_file - String
        validation_file - String
    """
    print("Training file", training_file)
    print("Validation file", validation_file)
    print("Output breadth", breadth)

    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']

    D_train = Dataset('Training', Data(X_train), Likelihoods(y_train, breadth))
    D_val = Dataset('Validation', Data(X_val), Likelihoods(y_val, breadth))

    return (D_train, D_val)


def bottleneck_architecture(input_shape, n_hidden, n_classes, nonlinear=tf.nn.relu):
    inputs = tf.placeholder(tf.float32, (None,) + input_shape)
    outputs = Connected(reshape2d(inputs), n_hidden, 1.0, nonlinear)
    outputs = Connected(outputs, n_classes, 1.0)
    return (inputs, outputs)


def main(_):
    # load bottleneck data
    (D_train, D_val) = load_bottleneck_data(FLAGS.training_file, FLAGS.validation_file, FLAGS.breadth)

    print(D_train.X.shape, D_train.y.shape)
    print(D_val.X.shape, D_val.y.shape)

    # TODO: define your model and hyperparams here
    # make sure to adjust the number of classes based on
    # the dataset
    # 10 for cifar10
    # 43 for traffic
    network = Network(
        bottleneck_architecture,
        input_shape=D_train.X.shape[1:],
        n_hidden=64,
        n_classes=FLAGS.breadth
    )

    # TODO: train your model here
    train(network, D_train, D_val, learning_rate=0.01, epochs=100)


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
