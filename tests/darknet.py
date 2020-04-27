import numpy as np
import ctypes
from ctypes import POINTER, c_void_p, c_int, c_char_p, c_float
import tensorflow.keras as keras

DARKNET_DLL = "../darknet/libdarknet.so"
HELPER_DLL = "./helper.so"


class _LAYER_OUTPUT(ctypes.Structure):
    _fields_ = [
        ('h', c_int),
        ('w', c_int),
        ('c', c_int),
        ('output', POINTER(c_float))
    ]


class _NET_OUTPUTS(ctypes.Structure):
    _fields_ = [
        ('count', c_int),
        ('output', POINTER(_LAYER_OUTPUT))
    ]


class _DETECTION(ctypes.Structure):
    _fields_ = [
        ('x', c_float),
        ('y', c_float),
        ('w', c_float),
        ('h', c_float),
        ('classes', c_int),
        ('prob', POINTER(c_float)),
        ('mask', POINTER(c_float)),
        ('objectness', c_float),
        ('sort_class', c_int)
    ]


class _IMAGE(ctypes.Structure):
    _fields_ = [
        ('w', c_int),
        ('h', c_int),
        ('c', c_int),
        ('data', ctypes.POINTER(c_float))
    ]


class darknet:
    """ctypes wrapper for original Darknet code"""

    helper_dll = ctypes.CDLL(HELPER_DLL, ctypes.RTLD_GLOBAL)

    helper_dll.get_net_outputs.argtypes = [c_void_p]
    helper_dll.get_net_outputs.restype = _NET_OUTPUTS

    helper_dll.free_net_outputs.argtypes = [_NET_OUTPUTS]

    def __init__(self, config_file, weights_file):
        self.dll = ctypes.CDLL(DARKNET_DLL, ctypes.RTLD_GLOBAL)

        self.dll.load_network.argtypes = [c_char_p, c_char_p, c_int]
        self.dll.load_network.restype = c_void_p

        self.dll.free_network.argtypes = [c_void_p]

        self.dll.network_inputs.argtypes = [c_void_p]
        self.dll.network_inputs.restype = c_int

        self.dll.network_outputs.argtypes = [c_void_p]
        self.dll.network_outputs.restype = c_int

        self.dll.network_height.argtypes = [c_void_p]
        self.dll.network_height.restype = c_int

        self.dll.network_width.argtypes = [c_void_p]
        self.dll.network_width.restype = c_int

        self.dll.get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
        self.dll.get_network_boxes.restype = POINTER(_DETECTION)

        self.dll.do_nms_sort.argtypes = [POINTER(_DETECTION), c_int, c_int, c_float]

        self.dll.free_detections.argtypes = [POINTER(_DETECTION), c_int]

        self.net = None     # in case loading fails... although, darknet tends to calls exit() in such cases
        self.net = self.dll.load_network(str(config_file).encode(), str(weights_file).encode(), 1)

        self.dll.network_predict_image.argtypes = [c_void_p, _IMAGE]

        self.dll.network_predict.argtypes = [c_void_p, np.ctypeslib.ndpointer(dtype=c_float, shape=self.input_shape(), flags='C')]

    def __del__(self):
        if self.net != None:
            self.dll.free_network(self.net)

    def input_size(self):
        return self.dll.network_inputs(self.net)

    def output_size(self):
        return self.dll.network_outputs(self.net)

    def input_shape(self):
        height = self.dll.network_height(self.net)
        width = self.dll.network_width(self.net)

        return (self.input_size()//height//width, height, width)

    class image:
        dll = ctypes.CDLL(DARKNET_DLL, ctypes.RTLD_GLOBAL)

        dll.load_image_color.argtypes = [c_char_p, c_int, c_int]
        dll.load_image_color.restype = _IMAGE

        dll.free_image.argtypes = [_IMAGE]

        dll.letterbox_image.argtypes = [_IMAGE, c_int, c_int]
        dll.letterbox_image.restype = _IMAGE

        @classmethod
        def load(cls, image_file, new_w=0, new_h=0):
            return darknet.image(cls.dll.load_image_color(str(image_file).encode(), new_w, new_h))

        def __init__(self, img):
            assert isinstance(img, _IMAGE)
            self.image = img

        def __del__(self):
            if self.image != None:
                self.__class__.dll.free_image(self.image)

        def __repr__(self):
            return f'<darknet.image w={self.image.w} h={self.image.h} c={self.image.c}>'

        def letterbox(self, width, height):
            return darknet.image(self.__class__.dll.letterbox_image(self.image, width, height))

        def to_array(self):
            # note that as_array doesn't copy the data
            return np.ctypeslib.as_array(self.image.data, shape=(self.image.c, self.image.h, self.image.w))\
                                    .copy().transpose([1,2,0])


    def predict(self, net_input):
        if isinstance(net_input, darknet.image):
            self.dll.network_predict_image(self.net, net_input.image)
        else:
            assert keras.backend.image_data_format() == 'channels_last'
            net_input = np.ascontiguousarray(net_input.transpose([2,0,1])) # channels last to first
            self.dll.network_predict(self.net, net_input)

        net_outputs = darknet.helper_dll.get_net_outputs(self.net)

        predictions = []
        for i in range(net_outputs.count):
            layer = net_outputs.output[i]
            predictions.append(
                np.ctypeslib.as_array(layer.output, shape=(layer.c, layer.h, layer.w)).copy().transpose([1,2,0]))

        darknet.helper_dll.free_net_outputs(net_outputs)

        return predictions if len(predictions) != 1 else predictions[0]


    def get_network_boxes(self, image, thresh=.5, hier_thresh=.5, do_nms=False):
        num_boxes = c_int()
        boxes = self.dll.get_network_boxes(self.net, image.image.w, image.image.h, thresh, hier_thresh,
                                           None, 0, ctypes.byref(num_boxes))

        if do_nms and num_boxes.value > 0:
            self.dll.do_nms_sort(boxes, num_boxes, boxes[0].classes, thresh)

        results = []
        for i in range(num_boxes.value):
            d = boxes[i]
            array = np.array([d.x, d.y, d.w, d.h, d.objectness] + [d.prob[i] for i in range(d.classes)], dtype=np.float32)
            results.append(np.expand_dims(array, 0))

        self.dll.free_detections(boxes, num_boxes)

        return np.concatenate(results, axis=0)
