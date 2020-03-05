import cv2
import numpy as np
from PIL import Image
from scipy import misc
from scipy.ndimage import affine_transform as scipy_affine_transform
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from skimage.transform import resize


def calcAffineMatrix(sourcePoints, targetPoints):
    # For three or more source and target points, find the affine transformation
    a = []
    b = []
    for sp, trg in zip(sourcePoints, targetPoints):
        a.append([sp[0], 0, sp[1], 0, 1, 0])
        a.append([0, sp[0], 0, sp[1], 0, 1])
        b.append(trg[0])
        b.append(trg[1])
    result, _, _, _ = np.linalg.lstsq(np.array(a), np.array(b), rcond=None)

    a0, a1, a2, a3, a4, a5 = result
    affineTrafo = np.float32([[a0, a2, a4], [a1, a3, a5]])
    return affineTrafo


def pil_to_arr(pil_image):
    rgb_image = pil_image.convert('RGB')
    img_arr = np.array(rgb_image)
    return img_arr


def arr_to_pil(img_arr):
    return Image.fromarray(np.uint8(img_arr)).convert('L')


def affine_transform(image, affine_value=0.003):  # 0.005):
    shape = image.shape
    alpha_affine = min(shape[0], shape[1]) * affine_value
    random_state = np.random.RandomState(None)
    # Random affine
    shape_size = shape[:2]
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32(
        [center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
         center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine,
                                       alpha_affine, size=pts1.shape).astype(np.float32)
    M = calcAffineMatrix(pts1, pts2)
    R = M[0:2, 0:2]
    Off = M[:, 2]
    for aD in range(shape[2]):
        image[:, :, aD] = scipy_affine_transform(
            image[:, :, aD], R, offset=Off, mode='constant', cval=255.0)
    return image


def elastic_transform(image, elastic_value_x=0.0004, elastic_value_y=0.0004):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications JUST in Y-DIRECTION).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    shape = image.shape
    random_state = np.random.RandomState(None)
    nY = shape[0] // 25
    nX = shape[1] // 25
    sigma = min(shape[1], shape[0]) * 0.0025
    alpha_X = elastic_value_x * min(shape[0], shape[1])
    alpha_Y = elastic_value_y * min(shape[0], shape[1])
    dx = gaussian_filter((random_state.rand(nY, nX) * 2 - 1), sigma)
    dy = gaussian_filter((random_state.rand(nY, nX) * 2 - 1), sigma)
    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(
        shape[0]), np.arange(shape[2]))
    dx = resize(dx, (shape[0], shape[1]), order=3)
    dy = resize(dy, (shape[0], shape[1]), order=3)
    # plt.imshow(dx, cmap=plt.cm.gray)
    # plt.show()
    dxT = []
    dyT = []
    for dummy in range(shape[2]):
        dxT.append(dx)
        dyT.append(dy)
    dx = np.dstack(dxT)
    dy = np.dstack(dyT)
    dx = dx * alpha_X
    dy = dy * alpha_Y
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x +
                                                      dx, (-1, 1)), np.reshape(z, (-1, 1))
    image = map_coordinates(image, indices, order=1, cval=255.0).reshape(shape)
    return image


def _create_matrices(shape, range):

    h, w = shape[0:2]

    points = np.random.uniform(range[0], range[1], size=(4, 2))
    points = -np.mod(np.abs(points), 1)

    # top left
    points[0, 1] = 1.0 - points[0, 1]  # h = 1.0 - jitter

    # top right
    points[1, 0] = 1.0 - points[1, 0]  # w = 1.0 - jitter
    points[1, 1] = 1.0 - points[1, 1]  # h = 1.0 - jitter

    # bottom right
    points[2, 0] = 1.0 - points[2, 0]  # h = 1.0 - jitter

    # bottom left
    # nothing

    points[:, 0] = points[:, 0] * w
    points[:, 1] = points[:, 1] * h

    # obtain a consistent order of the points and unpack them
    # individually
    points = _order_points(points)
    (tl, tr, br, bl) = points

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = int(max(int(widthA), int(widthB)) * 0.9)

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(points, dst)

    return M, maxHeight, maxWidth


def _order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    pts_ordered = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    pts_ordered[0] = pts[np.argmin(s)]
    pts_ordered[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    pts_ordered[1] = pts[np.argmin(diff)]
    pts_ordered[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return pts_ordered


def perspective_transform(im, range=(0.01, 0.07)):
    M, max_height, max_width = _create_matrices(im.shape, range)
    #max_width = int(max_width * 2.5)
    warped = cv2.warpPerspective(
        im, M, (max_width, max_height), borderValue=(255, 255, 255))
    if np.min(warped) > 250:
        warped = im
    return warped


class ElasticDistortionGenerator(object):
    @classmethod
    def afffine_transform(cls, pil_im):
        img_arr = pil_to_arr(pil_im)
        img_arr = affine_transform(img_arr)
        return arr_to_pil(img_arr)

    @classmethod
    def elastic_transform(cls, pil_im):
        img_arr = pil_to_arr(pil_im)
        img_arr = elastic_transform(img_arr)
        return arr_to_pil(img_arr)

    @classmethod
    def perspective_transform(cls, pil_im):
        img_arr = pil_to_arr(pil_im)
        img_arr = perspective_transform(img_arr)
        return arr_to_pil(img_arr)
