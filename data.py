import numpy as np


def _consecutive_index_generator(length, offset=0):
    """Generate pair of ids of consecutive images.

    Offset is the distance between the images.
    """
    offset += 1
    for i in range(length - offset):
        yield (i, i + offset)


def generate_learning_set(array, random_permutation=True, offset=0):
    """Generate learning set of consecutive scans

    Parameters
    ----------
    array: numpy array of shape n_scans x n_voxels
        Array of masked scans

    random_permutation: boolean
        If True, consecutive scans are switched with a probability of .5

    offset: int
        Distance between two consecutive scans

    Returns
    -------
    learning_set: list of (img_a, img_b, label)
        Two consecutive images, label is 1 if images are ordered, 0 otherwise
    """
    np.random.seed()

    learning_set = []
    for (ia, ib) in _consecutive_index_generator(array.shape[0], offset=offset):
        label = np.random.randint(0, 2)
        if label == 0:
            ia, ib = ib, ia
        learning_set.append((ia, ib, label))

    return learning_set


if __name__ == '__main__':
    array = np.arange(5)
    res = generate_learning_set(array)
    assert(len(res) == 4)
    for ia, ib, label in res:
        if label == 0:
            ia, ib = ib, ia

    res = generate_learning_set(array, random_permutation=False, offset=1)
    assert(len(res) == 3)
    for ia, ib, label in res:
        assert(ia < ib)
        assert(label == 1)
    print('Basic testing is OK')