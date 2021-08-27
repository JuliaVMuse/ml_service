from typing import List

from PIL import Image
import numpy as np
import cv2


def preprocess(input_image, beta):
    alpha = 2.0
    new_image = alpha * input_image - beta
    return np.clip(new_image, 0, 255).astype(np.uint8)


def preprocess_document(gray_orig: np.ndarray) -> np.array:
    """
    Normalize an input image

    :param gray_orig: np.array
    :return: doc's image, np.array
    """

    means, right_mins, right_means = [], [], []
    THRESHOLD = 0.84
    shape_orig = gray_orig.shape

    if shape_orig[1] * THRESHOLD - 10 > (shape_orig[1] // 3) * 2:
        for i in range((shape_orig[1] // 3) * 2, int(shape_orig[1] * THRESHOLD), 10):
            cropped = gray_orig[(shape_orig[0] // 16) * 2: (shape_orig[0] // 16) * 3, i: i + 20]

            min_value = cropped.min()
            mean_value = cropped.mean()
            right_mins.append(min_value)
            right_means.append(mean_value)

    for i in range(0, shape_orig[1] - 0, 10):
        cropped = gray_orig[(shape_orig[0] // 16) * 2: (shape_orig[0] // 16) * 3, i: i + 20]
        mean_value = cropped.mean()
        means.append(mean_value)
    min_value_mean = min(means)

    THRESHOLD = 235

    if min(right_mins) < 137 and min(right_means) < THRESHOLD:
        im = preprocess(gray_orig, min_value_mean)
        return np.array(Image.fromarray(im).convert('RGB'))
    else:
        return np.array(Image.fromarray(gray_orig).convert('RGB'))


def preprocessing_image(array_images: List,
                        type_doc: str) -> np.array:
    """
    Preprocessing

    :param array_images: doc's images, list
    :param type_doc: type of document, str
    :return: doc's image, np.array
    """

    im_gray = cv2.cvtColor(array_images[0], cv2.COLOR_RGB2GRAY)  # перевод из цветного в серое

    if type_doc == 'cert':
        return preprocess_document(im_gray)  # препроцессинг  return RGB

    else:
        return np.array(Image.fromarray(im_gray).convert('RGB'))
