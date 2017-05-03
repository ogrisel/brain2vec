import numpy as np


def _consecutive_index_generator(length, offset=1):
    """Generate pair of ids of consecutive images.

    Offset is the distance between the images.
    """
    for i in range(length - offset):
        yield (i, i + offset)


def generate_learning_set(array, scans_to_average=1, random_permutation=True, offset=1, seed=0):
    """Generate learning set of consecutive scans

    Parameters
    ----------
    array: numpy array of shape n_scans x n_voxels
        Array of masked scans

    scans_to_average: integer
        Number of scan preceding the prediction scan to average

    random_permutation: boolean
        If True, consecutive scans are switched with a probability of .5

    offset: int
        Distance between two consecutive scans

    Returns
    -------
    learning_set: (list of img, list of img, list of label)
        Lists of consecutive indices and list of labels. If label is 0, indices
        are ordered decreasingly
    """
    rng = np.random.RandomState(seed)
    n_samples = array.shape[0]
    ia_list = []
    ib_list = []
    label_list = []
    for (ia, ib) in _consecutive_index_generator(n_samples, offset=offset + scans_to_average - 1):
        label = 1
        scan_a = np.mean(array[ia:ia + scans_to_average], axis=0)
        scan_b = array[ib]
        if random_permutation:
            label = rng.randint(0, 2)
            if label == 0:
                scan_a, scan_b = scan_b, scan_a
        ia_list.append(scan_a)
        ib_list.append(scan_b)
        label_list.append(label)

    return np.asarray(ia_list), np.asarray(ib_list), np.asarray(label_list)


if __name__ == '__main__':
    array = np.arange(5)
    res = generate_learning_set(array)
    assert(len(res[0]) == 4)
    for ia, ib, label in zip(*res):
        if label == 0:
            assert(ia > ib)
        else:
            assert(ia < ib)
    res = generate_learning_set(array, random_permutation=False, offset=2)
    assert(len(res[0]) == 3)
    for ia, ib, label in zip(*res):
        assert(label == 1)
        assert(ia < ib)

    res = generate_learning_set(array, scans_to_average=2, random_permutation=False, offset=1)    
    assert(len(res[0]) == 3)
    for ia, ib, label in zip(*res):
        assert(label == 1)
        assert(ia < ib)

    print('Basic testing is OK')
