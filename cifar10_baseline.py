import pickle
from os.path import join as joinpath

import numpy as np

from datasets import merge, split, Dataset, Normalized, Vectorized
from layers import mini_inception_architecture
from networks import train, Network
from viewers import enlarge_plot_area, plot_report, Displayer


def load_labels(path):
    with open(path, mode='rb') as data:
        d = pickle.load(data)
        return d['label_names']


def DatasetCIFAR10(title, path):
    def load(name):
        dataset = Dataset('Batch', joinpath(path, name), key_X='data', key_y='labels', encoding='latin1')
        images = np.zeros((len(dataset), 32, 32, 3), dtype=np.uint8)
        for (i, image) in enumerate(dataset.X.data):
            for c in range(3):
                a = c * 1024
                b = a + 1024
                images[i, :, :, c].flat = image[a:b]

        dataset.X.data = images
        dataset.y.data = np.array(dataset.y.data)
        return dataset

    return Normalized(
        Vectorized(
            merge(title,
                load('data_batch_1'),
                load('data_batch_2'),
                load('data_batch_3'),
                load('data_batch_4'),
                load('data_batch_5')
            )
        )
    )


def display_samples():
    enlarge_plot_area()

    display = Displayer(load_labels('datasets/cifar-10-batches-py/batches.meta'))
    D_train = DatasetCIFAR10('Train', 'datasets/cifar-10-batches-py/')
    display(D_train)


def main():
    D_train = DatasetCIFAR10('Train', 'datasets/cifar-10-batches-py/')

    enlarge_plot_area()
    display = Displayer(load_labels('datasets/cifar-10-batches-py/batches.meta'))
    display(D_train)

    (D_train, D_val) = split(D_train)

    network = Network(
        mini_inception_architecture,
        input_shape=(32, 32, 3),
        depth=32,
        stride=1,
        n_hidden=64,
        n_classes=10
    )

    train(network, D_train, D_val, report=plot_report)


if __name__ == '__main__':
    main()
