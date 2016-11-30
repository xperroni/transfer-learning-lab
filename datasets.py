import pickle
from collections import defaultdict
from glob import iglob
from itertools import product
from os.path import isdir
from os.path import join as join_path
from re import search

import numpy as np
from scipy.misc import imread
from sklearn.cross_validation import train_test_split


class Data(object):
    r'''Base class for single-type data collection classes.
    '''
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, value):
        self.data[index] = value

    def __len__(self):
        return len(self.data)

    def extend(self, data):
        self.data = np.concatenate([self.data, data])

    @property
    def shape(self):
        return self.data.shape


class Images(Data):
    r'''Collection of RGB images, represented as 3D arrays of unsigned 8-bit integers.
    '''
    def __init__(self, data):
        Data.__init__(self, data)

    def image(self, index):
        return self[index]


class Tensors(Images):
    r'''Collection of multi-dimensional vectors.
    '''
    def __init__(self, values):
        Images.__init__(self, values.astype(np.float32))

    def image(self, index):
        x = self[index]
        image = np.zeros(x.shape, dtype=np.uint8)
        for d in range(3):
            channel = np.array(x[:, :, d])
            channel -= channel.min()
            channel *= (255.0 / channel.max())
            image[:, :, d] = channel.astype(np.uint8)

        return image


class Labels(Data):
    r'''Collection of class labels, represented as integer values.
    '''
    def __init__(self, data, breadth):
        Data.__init__(self, data)
        self.breadth = breadth
        if breadth == None:
            n = range(len(self))
            classes = set(self.classof(i) for i in n)
            self.breadth = len(classes)

    def classof(self, index):
        return self[index]

    @property
    def classes(self):
        classes = dict((c, []) for c in range(self.breadth))
        for i in range(len(self)):
            k = self.classof(i)
            classes[k].append(i)

        return classes


class Likelihoods(Labels):
    r'''Collection of vectors indicating the likelihoods an input belongs to each of a set of classes.
    '''
    def __init__(self, data, breadth):
        Labels.__init__(self, data, breadth)

        if len(data.shape) == 1:
            data = data[:, None]

        if data.shape[1] == 1:
            self.data = (np.arange(self.breadth) == data).astype(np.float32)

    def classof(self, index):
        return np.argmax(self[index])


class Dataset(object):
    r'''A collection of data cases and associated class identifiers.
    '''
    def __init__(self, *args, **kwargs):
        r'''Create a new dataset instance.

            Datasets can be loaded from files, for example:

                dataset = Dataset('title', 'path/to/pickled_file.p')

            They can also be created from other datasets:

                dataset = Dataset(other_dataset)

            or

                dataset = Dataset(title, X, y)
        '''
        if len(args) == 1:
            self.__assign(args[0])
            return

        self.title = args[0]
        if len(args) == 2:
            (X, y) = self.__load(args[1], kwargs)
            self.X = Images(X)
            self.y = Labels(y, kwargs.get('breadth'))
        elif len(args) == 3:
            self.X = args[1]
            self.y = args[2]
        else:
            raise Exception('Invalid argument list: %s' % str(args))

    def __assign(self, dataset):
        self.title = dataset.title
        self.X = dataset.X
        self.y = dataset.y

    def __load(self, path, kwargs):
        if isdir(path):
            X = []
            y = []
            for filename in iglob(join_path(path, '*')):
                match = search(r'(\d+)_\d+\.', filename)
                if match != None:
                    image = imread(filename)
                    label = int(match.group(1))
                    X.append(image)
                    y.append(label)

            return (np.array(X), np.array(y))

        key_X = kwargs.get('key_X', 'features')
        key_y = kwargs.get('key_y', 'labels')
        with open(path, mode='rb') as data:
            dataset = pickle.load(data, encoding=kwargs.get('encoding', 'ASCII'))
            return (dataset[key_X], dataset[key_y])

    def __len__(self):
        return len(self.X)

    def __str__(self):
        template = (
            '%s dataset\n'
            'Number of entries: %d\n'
            'Input shape: %s\n'
            'Output shape: %s\n'
            'Number of classes: %d\n'
        )

        return template % (
            self.title,
            len(self),
            str(self.X.shape[1:]),
            str(self.y.shape[1:]),
            self.y.breadth
        )

    def extend(self, dataset):
        self.X.extend(dataset.X)
        self.y.extend(dataset.y)


def split(dataset, rate=0.25):
    X = dataset.X
    y = dataset.y

    DataX = X.__class__
    DataY = y.__class__

    (X_train, X_valid, y_train, y_valid) = train_test_split(
        X.data,
        y.data,
        test_size=rate,
        random_state=832289
    )

    breadth = dataset.y.breadth
    return (
        Dataset('Train', DataX(X_train), DataY(y_train, breadth)),
        Dataset('Validate', DataX(X_valid), DataY(y_valid, breadth))
    )


def merge(title, *datasets):
    template = datasets[0]
    X = template.X
    y = template.y

    shape_X = (sum(dataset.X.shape[0] for dataset in datasets),) + X.shape[1:]
    shape_y = (sum(dataset.y.shape[0] for dataset in datasets),) + y.shape[1:]
    breadth = y.breadth

    Merged = template.__class__
    DataX = X.__class__
    DataY = y.__class__

    X = DataX(np.zeros(shape_X))
    y = DataY(np.zeros(shape_y), breadth)

    a = 0
    for dataset in datasets:
        b = a + len(dataset)
        X[a:b] = dataset.X.data
        y[a:b] = dataset.y.data
        a = b

    return Merged(title, X, y)


class Vectorized(Dataset):
    r'''A dataset where both inputs and outputs are represented as floating-point arrays.
    '''
    def __init__(self, *args, **kwargs):
        Dataset.__init__(self, *args, **kwargs)
        self.X = Tensors(self.X.data)
        self.y = Likelihoods(self.y.data, self.y.breadth)


class Normalized(Dataset):
    def __init__(self, *args, **kwargs):
        Dataset.__init__(self, *args, **kwargs)

        X = self.X
        n = X.shape[0]
        d = X.shape[-1]
        for (i, j) in product(range(n), range(d)):
            channel = X[i, :, :, j]
            channel -= channel.mean()
            channel /= channel.std()
