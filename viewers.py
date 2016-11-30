from random import sample

import numpy as np
from matplotlib import pyplot as plt


def enlarge_plot_area():
    plt.rcParams['figure.figsize'] = (9.0, 4.0) # Otiginal: (6.0, 4.0)


class Displayer(object):
    def __init__(self, labels):
        self.labels = labels

    def __call__(self, dataset, width=5):
        r'''Display `k` samples of each class from the given dataset.
        '''
        n = len(dataset)

        print('\n\n  %s' % ('-' * 80))
        print('  %s dataset (total %d entries)' % (dataset.title, n))
        print('  %s\n\n' % ('-' * 80))

        labels = self.labels
        classes = dataset.y.classes

        for c in range(len(classes)):
            indexes = classes[c]
            n = len(indexes)
            if n == 0:
                continue

            print('  Class %d ("%s", total %d entries) samples:' % (c, labels[c], len(indexes)))
            k = min(width, n)
            s = sample(indexes, k)
            self.display_signs(dataset, s, width)

    def display_signs(self, dataset, indexes, width):
        r'''Display the indexed sign images and corresponding labels side by side.
        '''
        n = len(indexes)
        for j in range(n):
            plotter = plt.subplot2grid((1, width), (0, j))
            self.display_sign(plotter, dataset, indexes[j])

        plt.tight_layout()
        plt.show()

    def display_sign(self, plotter, dataset, i):
        r'''Display a sign image and corresponding numeric label.
        '''
        plotter.imshow(dataset.X.image(i))
        plotter.xaxis.set_visible(False)
        plotter.yaxis.set_visible(False)
        plotter.title.set_text(str(i))


def plot_lines(title, x, *ys, **kwargs):
    y_min = kwargs.get('y_min', maxsize)
    y_max = kwargs.get('y_max', -maxsize)

    plotter = plt.subplot(111)
    for (y, c, l) in ys:
        label = '%s (last: %.3f)' % (l, round(y[-1], 3))
        plotter.plot(x, y, c, label=label)
        y_min = min(floor(np.min(y)), y_min)
        y_max = max(ceil(np.max(y)), y_max)

    plotter.set_title(title)
    plotter.set_xlim([x[0], x[-1]])
    plotter.set_ylim([y_min, y_max])
    plotter.legend(loc=kwargs.get('loc', 1))
    plt.tight_layout()
    plt.show()


def plot_report(x_batch, y_train, y_val, y_loss):
    plot_lines('Loss', x_batch,
        (y_loss, 'g', 'Loss')
    )

    plot_lines('Accuracy', x_batch,
        (y_train, 'r', 'Training Accuracy'),
        (y_val, 'b', 'Validation Accuracy'),
        loc=4
    )
