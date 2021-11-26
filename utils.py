import numpy as np
from PIL import Image

def openImage(path):
    '''
    Open image from given ``path`` and split into RGB channels.

    Parameters
    ----------
    path : str
        Path to file as a string, ``pathlib.Path`` or file object.

    Returns
    -------
    numpy.ndarray
        3-dimensional image array of shape ``(H, W, 3)`` where ``H`` and
        ``W`` are the height and width of the image respectively.
    '''

    # open image file
    im = Image.open(path)
    # convert image into RGB channels array
    img = np.array(im.convert('RGB'))

    return img

def saveImage(img, path):
    '''
    Open image from given ``path`` and split into RGB channels.

    Parameters
    ----------
    img : numpy.ndarray
        3-dimensional image array.
    path : str
        Path to file as a string, ``pathlib.Path`` or file object.
    '''

    im = Image.fromarray(
        np.array(
            img,
            dtype=np.uint8
        ),
        'RGB',
    )
    im.save(path)

def splitChannels(img):
    '''
    Split image array into R, G, B arrays.

    Parameters
    ----------
    img : numpy.ndarray
        3-dimensional image array.

    Returns
    -------
    R : numpy.ndarray
        2-dimensional red channel array.
    G : numpy.ndarray
        2-dimensional green channel array.
    B : numpy.ndarray
        2-dimensional blue channel array.
    '''

    # unpack channels into three arrays
    R, G, B = np.transpose(img)
    R, G, B = (
        np.transpose(R),
        np.transpose(G),
        np.transpose(B),
    )

    return R, G, B

def combineChannels(R, G, B):
    '''
    Combine R, G, B arrays into image array.

    Parameters
    ----------
    R : numpy.ndarray
        2-dimensional red channel array.
    G : numpy.ndarray
        2-dimensional green channel array.
    B : numpy.ndarray
        2-dimensional blue channel array.

    Returns
    -------
    numpy.ndarray
        3-dimensional image array.
    '''

    # stack channels into one array
    img = np.stack((R, G, B), axis=-1)

    return img

def addNoise(channel, p=0.5, is_noisy=None):
    '''
    Add randomized noise to channel.

    Parameters
    ----------
    channel : numpy.ndarray
        2-dimensional channel array.
    p : float, optional
        Probability of a pixel being converted to noise, should be in the
        range ``[0, 1)``.
    is_noisy : numpy.ndarray, optional
        2-dimensional boolean array indicating which pixels should be
        converted to noise. If not provided, this is randomized.

    Returns
    -------
    noisy_channel : numpy.ndarray
        Channel with randomized noise added.
    is_noisy : numpy.ndarray
        Boolean array of same shape as ``channel``, indicating which pixels
        have been converted to noise.
    '''

    noisy_channel = channel.copy()
    channel_shape = np.shape(noisy_channel)

    # if is_noisy is not specified, randomize it
    if is_noisy is None:
        is_noisy = np.array(
            np.random.rand(*channel_shape) < p,
            dtype=bool,
        )
    else:
        if channel_shape != np.shape(is_noisy):
            raise ValueError('channel and is_noisy must be of same shape')

    # randomize noise array of values 0 or 255
    noise = np.array(
        np.around(np.random.rand(*channel_shape)) * 255,
        dtype=np.uint8,
    )

    # replace appropriate pixels with noisy values
    for i in range(channel_shape[0]):
        for j in range(channel_shape[1]):
            if is_noisy[i][j]:
                noisy_channel[i][j] = noise[i][j]

    return noisy_channel, is_noisy

def slidingWindowOperation(channel, window_shape, op='mean'):
    '''
    Perform operation on sliding window over channel.

    Parameters
    ----------
    channel : numpy.ndarray
        2-dimensional channel array.
    window_shape : int or numpy.ndarray
        Integer or array indicating shape of window. Window is ``(N, N)``
        square if given integer ``N``, ``(N, M)`` rectangular if given
        array ``[N, M]``.
    op : str or callable function, optional
        Operation to perform on each window. It an be set to ``'mean'`` or
        ``'median'`` for mean and median operations respectively.

    Returns
    -------
    numpy.ndarray
        Output array of size ``(H - N + 1, W - M + 1)``, where ``H`` and
        ``W`` are the height and width of the image respectively.
    '''

    # process op as string or callable
    if isinstance(op, str):
        if op.lower().strip() == 'mean':
            op = np.mean
        elif op.lower().strip() == 'median':
            op = np.median
        else:
            raise ValueError(f'Unknown operation {op}')
    elif not callable(op):
        raise ValueError('op must be string or callable function')

    # process window shape
    window_shape_shape = np.shape(window_shape)
    if np.shape(window_shape_shape)[0] == 0:
        N = window_shape
        M = window_shape
    elif window_shape_shape[0] == 2:
        N, M = window_shape
    else:
        raise ValueError('window_shape must be scalar or 2-dimensional')

    channel_shape = np.shape(channel)

    # get output shape
    output_shape = (
        channel_shape[0] - N + 1,
        channel_shape[1] - M + 1,
    )
    output = np.zeros(output_shape, dtype=np.float64)

    # perform operation for every window
    for i in range(output_shape[0]):
        for j in range(output_shape[1]):
            output[i][j] = op(channel[i : i + N, j : j + M])

    return output

def detectNoise(channel, N, E=53):
    '''
    Identify noisy pixels in channel array.

    Parameters
    ----------
    channel : numpy.ndarray
        2-dimensional channel array.
    N : int
        Window length. Must be odd.
    E : int
        Noise threshold value.

    Returns
    -------
    numpy.ndarray
        Boolean array of same shape as ``channel``, indicating which pixels
        are noisy.
    '''

    # N must be odd since we use centre pixel in each window
    if N % 2 != 1:
        raise ValueError('N must be an odd integer')

    # compute median by window
    median = slidingWindowOperation(channel, N, 'median')
    median_shape = np.shape(median)

    channel_shape = np.shape(channel)
    is_noisy = np.zeros(channel_shape, dtype=bool)

    # create noisy pixels array based on threshold values
    for i in range(median_shape[0]):
        for j in range(median_shape[1]):
            offset_i = i + (N // 2)
            offset_j = j + (N // 2)
            if abs(channel[offset_i, offset_j] - median[i, j]) > E:
                is_noisy[offset_i, offset_j] = True

    return is_noisy

def getCandidate(C, A, idx):
    '''
    Calculate candidate for interpolated pixel using given indices.

    Parameters
    ----------
    C : numpy.ndarray
        Primary channel array window.
    A : numpy.ndarray
        Secondary channel array window.
    idx : (int, int)
        Indices to use for computation.

    Returns
    -------
    candidate_C : int
        Candidate for interpolated pixel using primary channel window.
    candidate_CA : int
        Candidate for interpolated pixel using primary and secondary
        channel windows.

    Notes
    -----
    Candidates are calculated using the following computations:

    .. math::
        \\hat{C}[4] &= \\frac{|C[L] + C[R]|}{2} \\\\
        \\hat{C}[4] &= \\frac{|C[L] + C[R]|}{4} +
        \\frac{|-A[L] + 2A[4] - A[R]|}{4}
    '''

    L, R = idx

    # gradient term
    grad = (int(C[L]) + int(C[R])) / 2
    # laplacian term
    lap = abs(-int(A[L]) + (2 * int(A[4])) - int(A[R])) / 2

    candidate_C = np.uint8(grad)
    candidate_CA = np.uint8((grad + lap) / 2)

    return candidate_C, candidate_CA

def getValidity(is_noisy_C, is_noisy_A, idx):
    '''
    Calculate validity of candidate for interpolated pixel using given
    indices. Validity is based on the channel's noisy pixels.

    Parameters
    ----------
    is_noisy_C : numpy.ndarray
        Boolean array indicating which pixels of primary channel window
        array are noisy.
    is_noisy_A : numpy.ndarray
        Boolean array indicating which pixels of secondary channel window
        array are noisy.
    idx : (int, int)
        Indices to use for computation.

    Returns
    -------
    validity_C : bool
        Validity of interpolated pixel using primary channel window.
    validity_CA : bool
        Validity of interpolated pixel using primary and secondary channel
        windows.

    Notes
    -----
    Validity is evaluated by looking at whether the pixels at the given
    indices are noisy. If any one of the pixels are noisy, the candidate is
    considered invalid.
    '''

    L, R = idx

    # validity using only C channel
    validity_C = (
        (not is_noisy_C[L]) and
        (not is_noisy_C[R])
    )

    # validity using C and A channels
    validity_CA = (
        validity_C and
        (not is_noisy_A[L]) and
        (not is_noisy_A[4]) and
        (not is_noisy_A[R])
    )

    return validity_C, validity_CA

def getDeference(C, A, idx):
    '''
    Calculate deference of candidate for interpolated pixel using given
    indices.

    Parameters
    ----------
    C : numpy.ndarray
        Primary channel array window.
    A : numpy.ndarray
        Secondary channel array window.
    idx : (int, int)
        Indices to use for computation.

    Returns
    -------
    deference_C : int
        Deference of interpolated pixel using primary channel window.
    deference_CA : int
        Deference of interpolated pixel using primary and secondary channel
        windows.

    Notes
    -----
    Candidates are calculated using the following computations:

    .. math::
        \\hat{C}_{def}[4] &= |C[L] - C[R]| \\\\
        \\hat{C}_{def}[4] &= ||C[L] - C[R]| - |A[L] - A[R]||
    '''

    L, R = idx

    # deference using only C channel
    deference_C = abs(int(C[L]) - int(C[R]))
    # deference using C and A channels
    deference_CA = abs(
        deference_C - abs(int(A[L]) - int(A[R]))
    )

    return deference_C, deference_CA

def interpolatePixel(C, A1, A2, is_noisy_C, is_noisy_A1, is_noisy_A2):
    '''
    Calculate interpolated pixel for centre pixel of given ``3x3`` window
    of channels.

    Parameters
    ----------
    C : numpy.ndarray
        Primary channel array window.
    A1 : numpy.ndarray
        Secondary channel array 1 window.
    A2 : numpy.ndarray
        Secondary channel array 2 window.
    is_noisy_C : numpy.ndarray
        Boolean array indicating which pixels of primary channel window
        array are noisy.
    is_noisy_A1 : numpy.ndarray
        Boolean array indicating which pixels of secondary channel window
        array 1 are noisy.
    is_noisy_A2 : numpy.ndarray
        Boolean array indicating which pixels of secondary channel window
        array 2 are noisy.

    Returns
    -------
    int
        Interpolated pixel.
    '''

    # raise error if shape isn't 3x3
    expected_shape = (3, 3)
    if (
        np.shape(C) != expected_shape or
        np.shape(A1) != expected_shape or
        np.shape(A2) != expected_shape or
        np.shape(is_noisy_C) != expected_shape or
        np.shape(is_noisy_A1) != expected_shape or
        np.shape(is_noisy_A2) != expected_shape
    ):
        err_msg = 'C, A1, A2, is_noisy_C, is_noisy_A1, is_noisy_A2 '
        err_msg += 'must all be of shape (3, 3)'
        raise ValueError(err_msg)

    # get flattened copies of all input arrays
    _C = np.reshape(C, 9)
    _A1 = np.reshape(A1, 9)
    _A2 = np.reshape(A2, 9)
    _is_noisy_C = np.reshape(is_noisy_C, 9)
    _is_noisy_A1 = np.reshape(is_noisy_A1, 9)
    _is_noisy_A2 = np.reshape(is_noisy_A2, 9)

    # indices to compute candidates
    indices = np.array([
        [3, 5],
        [1, 7],
        [0, 8],
        [2, 6],
    ])

    # candidates array
    candidates_C = np.zeros(8, dtype=np.uint8)
    candidates_CA = np.zeros(8, dtype=np.uint8)
    # validities array
    validity_C = np.zeros(8, dtype=bool)
    validity_CA = np.zeros(8, dtype=bool)
    # deferences array
    deference_C = np.zeros(8, dtype=np.uint8)
    deference_CA = np.zeros(8, dtype=np.uint8)

    # compute candidates, validities and deferences
    for i, idx in enumerate(indices):
        candidates_C[i], candidates_CA[i] = getCandidate(
            _C,
            _A1,
            idx,
        )
        validity_C[i], validity_CA[i] = getValidity(
            _is_noisy_C,
            _is_noisy_A1,
            idx,
        )
        deference_C[i], deference_CA[i] = getDeference(
            _C,
            _A1,
            idx,
        )

        candidates_C[i + 4], candidates_CA[i + 4] = getCandidate(
            _C,
            _A2,
            idx,
        )
        validity_C[i + 4], validity_CA[i + 4] = getValidity(
            _is_noisy_C,
            _is_noisy_A2,
            idx,
        )
        deference_C[i + 4], deference_CA[i + 4] = getDeference(
            _C,
            _A2,
            idx,
        )

    # use computed arrays to determine candidate
    validity_CA_sum = np.sum(validity_CA)
    if validity_CA_sum == 1:
        # if only one is valid, return it
        return candidates_CA[np.argmax(validity_CA)]
    elif validity_CA_sum > 1:
        # if more than one valid, return one with minimum deference
        return candidates_CA[np.argmin(deference_CA)]
    else:
        # if none are valid, repeat but neglect A channel
        validity_C_sum = np.sum(validity_C)
        if validity_C_sum == 1:
            # if only one is valid, return it
            return candidates_C[np.argmax(validity_C)]
        elif validity_C_sum > 1:
            # if more than one valid, return one with minimum deference
            return candidates_C[np.argmin(deference_C)]
        else:
            # if none are valid, return median pixel
            return np.median(_C)

def interpolateChannel(C, A1, A2, is_noisy_C, is_noisy_A1, is_noisy_A2):
    '''
    Interpolate pixels for channel where noisy pixels are present.

    Parameters
    ----------
    C : numpy.ndarray
        Primary channel array.
    A1 : numpy.ndarray
        Secondary channel array 1.
    A2 : numpy.ndarray
        Secondary channel array 2.
    is_noisy_C : numpy.ndarray
        Boolean array indicating which pixels of primary channel array are
        noisy.
    is_noisy_A1 : numpy.ndarray
        Boolean array indicating which pixels of secondary channel array 1
        are noisy.
    is_noisy_A2 : numpy.ndarray
        Boolean array indicating which pixels of secondary channel array 2
        are noisy.

    Returns
    -------
    int
        Interpolated primary channel.
    '''

    # raise error if shapes of all input arrays are not equal
    C_shape = np.shape(C)
    if (
        C_shape != np.shape(A1) or
        C_shape != np.shape(A2) or
        C_shape != np.shape(is_noisy_C) or
        C_shape != np.shape(is_noisy_A1) or
        C_shape != np.shape(is_noisy_A2)
    ):
        err_msg = 'C, A1, A2, is_noisy_C, is_noisy_A1, is_noisy_A2 '
        err_msg += 'must all be of same shape'
        raise ValueError(err_msg)

    N = 3
    C_interpolated = C.copy()
    for i in range(N // 2, C_shape[0] - (N // 2)):
        for j in range(N // 2, C_shape[1] - (N // 2)):
            if is_noisy_C[i][j]:
                i_win = [i - (N // 2), i + (N // 2) + 1]
                j_win = [j - (N // 2), j + (N // 2) + 1]
                C_interpolated[i][j] = interpolatePixel(
                    C[i_win[0] : i_win[1], j_win[0] : j_win[1]],
                    A1[i_win[0] : i_win[1], j_win[0] : j_win[1]],
                    A2[i_win[0] : i_win[1], j_win[0] : j_win[1]],
                    is_noisy_C[i_win[0] : i_win[1], j_win[0] : j_win[1]],
                    is_noisy_A1[i_win[0] : i_win[1], j_win[0] : j_win[1]],
                    is_noisy_A2[i_win[0] : i_win[1], j_win[0] : j_win[1]],
                )

    return C_interpolated
