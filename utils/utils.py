
import cv2
import numpy as np
from tensorflow import cast, float32

def get_image(path: str = None):
    """ Read particular image and converts it to a tensor ready for
    prediction.

    :params path: Path to the image to be predicted
    :return x: image converted to a tensor format
    """

    dim = (256, 256)

    x = 'data/movie_frames/test/Y/Y5.jpg' if path == None else path

    try:
        x = cv2.imread(x)
    except:
        raise ValueError('There are no .jpg movie frames. Run mp4_to_jpg and pass the test tag.')

    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)


    x = cv2.resize(x, dim, interpolation = cv2.INTER_AREA) / 255

    x = x.reshape((1,256,256,3))

    x = cast(x, float32)

    return x
