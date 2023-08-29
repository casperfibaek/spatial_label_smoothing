from typing import Tuple
import numpy as np


def _distance_2D(p1: np.ndarray, p2: np.ndarray) -> float:
    """ Returns the distance between two points. (2D) """
    d1 = (p1[0] - p2[0]) ** 2
    d2 = (p1[1] - p2[1]) ** 2

    return np.sqrt(d1 + d2)


def kernel_sobel(
    radius=1,
    scale=2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates a 2D Sobel style kernel consisting of a horizontal and vertical component.
    This function returns a kernel that can be used to apply a Sobel filter to an image.

    `kernel_gx, kernel_gy = get_kernel_sobel(radius=1, scale=2)`

    The kernels for radis=2, scale=2 are:
    ```python
    gx = [
        [ 0.56  0.85  0.   -0.85 -0.56],
        [ 0.85  1.5   0.   -1.5  -0.85],
        [ 1.    2.    0.   -2.   -1.  ],
        [ 0.85  1.5   0.   -1.5  -0.85],
        [ 0.56  0.85  0.   -0.85 -0.56],
    ]

    gy = [
        [ 0.56  0.85  1.    0.85  0.56],
        [ 0.85  1.5   2.    1.5   0.85],
        [ 0.    0.    0.    0.    0.  ],
        [-0.85 -1.5  -2.   -1.5  -0.85],
        [-0.56 -0.85 -1.   -0.85 -0.56],
    ]
    ```

    Parameters
    ----------
    radius : float, optional
        The radius of the kernel. Default: 1.0.

    scale : float, optional
        The scale of the kernel. Default: 2.0.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The kernels 
    
    """
    size = np.int64(np.ceil(radius) * 2 + 1)
    kernel = np.zeros((size, size), dtype=np.float32)
    center = np.array([0.0, 0.0], dtype=np.float32)

    step = size // 2
    for idx_i, col in enumerate(range(-step, step + 1)):
        for idx_j, row in enumerate(range(-step, step + 1)):
            point = np.array([col, row], dtype=np.float32)
            distance = _distance_2D(center, point)
            if col == 0 and row == 0:
                kernel[idx_i, idx_j] = 0
            else:
                weight = np.power((1 - 0.5), distance) * 2
                kernel[idx_i, idx_j] = weight * scale

    # vertical
    kernel_gx = kernel.copy()
    kernel_gx[:, size // 2:] *= -1
    kernel_gx[:, size // 2] = 0

    # horisontal
    kernel_gy = kernel.copy()
    kernel_gy[size // 2:, :] *= -1
    kernel_gy[size // 2, :] = 0

    return kernel_gx, kernel_gy


def _area_covered(square, radius):
    """
    Calculates the area covered by a circle within a square.
    Monte-carlo(ish) method. Can be parallelized.
    """
    n_points = 100
    min_y = square[:, 0].min()
    max_y = square[:, 0].max()
    min_x = square[:, 1].min()
    max_x = square[:, 1].max()

    steps = int(np.rint(np.sqrt(n_points)))
    range_y = np.linspace(min_x, max_x, steps)
    range_x = np.linspace(min_y, max_y, steps)

    center = np.array([0.0, 0.0], dtype=np.float32)
    adjusted_radius = radius + 0.5

    points_within = 0
    for y in range_y:
        for x in range_x:
            point = np.array([y, x], dtype=np.float32)
            if _distance_2D(center, point) <= adjusted_radius:
                points_within += 1

    area = points_within / (steps ** 2)

    return area


def _circular_kernel_2D(radius):
    """ Creates a circular 2D kernel. Supports fractional radii. """
    size = np.int64(np.ceil(radius) * 2 + 1)
    kernel = np.zeros((size, size), dtype=np.float32)

    center = np.array([0.0, 0.0], dtype=np.float32)

    step = size // 2
    for idx_i, col in enumerate(range(-step, step + 1)):
        for idx_j, row in enumerate(range(-step, step + 1)):
            square = np.zeros((4, 2), dtype=np.float32)
            square[0] = np.array([col - 0.5, row - 0.5], dtype=np.float32)
            square[1] = np.array([col + 0.5, row - 0.5], dtype=np.float32)
            square[2] = np.array([col + 0.5, row + 0.5], dtype=np.float32)
            square[3] = np.array([col - 0.5, row + 0.5], dtype=np.float32)

            within = np.zeros(4, dtype=np.uint8)
            for i in range(4):
                within[i] = _distance_2D(center, square[i]) <= radius + 0.5

            # Case 1: completely inside
            if within.sum() == 4:
                kernel[idx_i][idx_j] = 1.0

            # Case 2: completely outside
            elif within.sum() == 0:
                kernel[idx_i][idx_j] = 0.0

            # Case 3: partially inside
            else:
                kernel[idx_i][idx_j] = _area_covered(square, radius)

    return kernel


def _distance_weighted_kernel_2D(radius, method, decay=0.2, sigma=2.0):
    """
    Creates a distance weighted kernel.
    
    Parameters
    ----------

    radius : float
        Radius of the kernel.
    
    method : int
        Method to use for weighting.
        0. linear
        1. sqrt
        2. power
        3. log
        4. gaussian
        5. constant
    """
    size = np.int64(np.ceil(radius) * 2 + 1)
    kernel = np.zeros((size, size), dtype=np.float32)

    center = np.array([0.0, 0.0], dtype=np.float32)

    step = size // 2
    for idx_i, col in enumerate(range(-step, step + 1)):
        for idx_j, row in enumerate(range(-step, step + 1)):
            point = np.array([col, row], dtype=np.float32)
            distance = _distance_2D(center, point)

            # Linear
            if method == 0:
                kernel[idx_i, idx_j] = np.power((1 - decay), distance)

            # Sqrt
            elif method == 1:
                kernel[idx_i, idx_j] = np.power(np.sqrt((1 - decay)), distance)

            # Power
            elif method == 2:
                kernel[idx_i, idx_j] = np.power(np.power((1 - decay), 2), distance)

            # Gaussian
            elif method == 3:
                kernel[idx_i, idx_j] = np.exp(-(np.power(distance, 2)) / (2 * np.power(sigma, 2)))

            # Constant
            else:
                kernel[idx_i, idx_j] = 1.0

    return kernel


def create_kernel(
    radius: float,
    circular: bool = False,
    distance_weighted: bool = False,
    normalised: bool = True,
    hole: bool = False,
    method: int = 0,
    decay: float = 0.2,
    sigma: float = 2.0,
) -> np.ndarray:
    """
    Creates a 2D kernel for convolution.

    Parameters
    ----------
    radius : float
        Radius of the kernel.
    
    circular : bool
        Whether to use a circular kernel.
    
    distance_weighted : bool
        Whether to use a distance weighted kernel.
    
    normalised : bool
        Whether to normalise the kernel.
    
    hole : bool
        Whether to create a hole in the center of the kernel.

    method : int
        Method to use for weighting.
        0. linear
        1. sqrt
        2. power
        3. gaussian
        4. constant
    
    decay : float
        Decay rate for distance weighted kernels. Only used if `distance_weighted` is True.

    sigma : float
        Sigma for gaussian distance weighted kernels. Only used if `distance_weighted` is True and `method` is 3.

    Returns
    -------
    kernel : np.ndarray
        The kernel.
    """
    size = np.int64(np.ceil(radius) * 2 + 1)
    kernel = np.ones((size, size), dtype=np.float32)

    if hole:
        kernel[size // 2, size // 2] = 0.0

    if circular:
        circular_kernel = _circular_kernel_2D(radius)
        kernel *= circular_kernel

    if distance_weighted:
        distance_weighted_kernel = _distance_weighted_kernel_2D(
            radius,
            method,
            decay,
            sigma,
        )
        kernel *= distance_weighted_kernel

    if normalised:
        kernel /= np.sum(kernel)

    return kernel
