import tensorflow.keras as keras
import numpy as np
import PIL

def load(image_file):
    """Loads an image from a file.

       Arguments:
       image_file -- path to image file.
    """
    image = keras.preprocessing.image.load_img(str(image_file))
    image = keras.preprocessing.image.img_to_array(image, data_format='channels_last', dtype=np.float32)
    image /= 255.0
    return image


def resize(image_array, new_w, new_h):
    """Resizes an image.

       Arguments:
       image_array -- Numpy array image array, as returned by load()
       new_w -- new width
       new_h -- new height
    """
    im_h, im_w, im_c = image_array.shape

    assert new_w > 1 and new_h > 1  # darknet can't handle these, either

    # PIL's resizing yields enough of a difference in the yolov3 network output
    # that I thought it worthwhile to implement resizing the way Darknet does.

    sx = np.arange(new_w-1, dtype=np.float32) * (np.float32(im_w-1) / np.float32(new_w-1))
    ix = sx.astype(np.int)
    dx = np.reshape(sx - ix.astype(np.float32), (1, new_w-1, 1))

    part = np.concatenate([(1-dx)*image_array[:,ix,:] + dx*image_array[:,ix+1,:],
                           np.expand_dims(image_array[:,-1,:], 1)],
                          axis=1)

    sy = np.arange(new_h, dtype=np.float32) * (np.float32(im_h-1) / np.float32(new_h-1))
    iy = sy.astype(np.int)
    dy = np.reshape(sy - iy.astype(np.float32), (new_h, 1, 1))

    resized = (1-dy)*part[iy,...] + np.concatenate([dy[:-1,...]*part[iy[:-1]+1,...], np.zeros((1,new_w,im_c))])
    return resized


def letterbox(image_array, new_w, new_h, resize_with_pil=False):
    """Fits an image into the given dimensions while maintaining aspect ratio.
       This results in an image that has the (resized) original image in the center, padded
       with .5 either above and below or to its left and right, depending on the original
       message's and desired output's aspect ratios.

       Arguments:
       image_array -- image to letterbox
       new_w -- new width
       new_h -- new height
       resize_with_pil -- use PIL to resize the original message, to facilitate experiments
    """
    im_h, im_w, im_c = image_array.shape

    boxed = np.full((new_h, new_w, im_c), .5, dtype=np.float32)

    if new_w/im_w < new_h/im_h:
        new_h = (im_h * new_w) // im_w
    else:
        new_w = (im_w * new_h) // im_h

    if resize_with_pil:
        image = keras.preprocessing.image.array_to_img(image_array, data_format='channels_last', dtype=np.float32)
        image = image.resize((new_w, new_h), resample=PIL.Image.BILINEAR)
        image_array = keras.preprocessing.image.img_to_array(image, data_format='channels_last', dtype=np.float32)
        image_array /= 255.                                                                                               
    else:
        image_array = resize(image_array, new_w, new_h)

    y = (boxed.shape[0]-new_h)//2
    x = (boxed.shape[1]-new_w)//2

    boxed[y:y+new_h,x:x+new_w,:] = image_array

    return boxed
