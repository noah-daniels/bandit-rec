import numpy as np


def my_argmax(array, k):
    """
    Returns the indices of the k largest elements.

    >>> a = my_argmax([1, 5, 2, 3, 4], 2)
    >>> 1 in a and 4 in a
    True
    """
    return np.argpartition(array, -k)[-k:]


def my_argmax_dict(dict_, k):
    sorted_ = sorted(dict_.items(), key=lambda x: x[1], reverse=True)

    if len(dict_) <= k:
        return [x for x, _ in sorted_]

    result = []
    while len(result) < k:
        batch = []
        x, y = sorted_.pop(0)
        batch.append(x)
        while len(sorted_) > 0:
            x2, y2 = sorted_.pop(0)
            if y2 == y:
                batch.append(x2)
            else:
                sorted_.insert(0, (x2, y2))
                break

        if len(result) + len(batch) <= k:
            result.extend(batch)
        else:
            result.extend(np.random.choice(batch, k - len(result), replace=False))

    return result


def bin_linear(data, bins):
    """
    Split data into bins and sum the contents of each bin.
    If the data does not divide equally into the given amount of bins, pad with zeros.

    >>> bin_linear(np.array([1, 2, 3, 4]), 2)
    array([3, 7])
    >>> bin_linear(np.array([1, 2, 3, 4, 5]), 2)
    array([6, 9])
    """
    if not len(data) % bins:
        padded = data
    else:
        padded = np.pad(data, (0, bins - len(data) % bins))
    return np.vstack(np.split(padded, bins)).sum(axis=1)


def smooth_fast(y, box_pts=5):
    n = len(y)
    ws = 2 * box_pts + 1
    y_smooth = []
    for i, yi in enumerate(y):
        if i < box_pts:
            y_smooth.append(None)
        elif i == box_pts:
            y_smooth.append(np.mean(y[i - box_pts : i + box_pts + 1]))
        elif i < n - box_pts:
            y_smooth.append(
                y_smooth[i - 1] + (y[i + box_pts] - y[i - box_pts - 1]) / ws
            )
        elif i < n:
            y_smooth.append(None)

    return y_smooth
