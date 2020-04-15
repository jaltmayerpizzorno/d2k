import tensorflow.keras as keras
import numpy as np


class Yolo(keras.layers.Layer):
    """Implements a Keras YOLO layer.  Training is not yet supported."""
    def __init__(self, classes=None, anchors=None, net_dims=None, just_activate=False, **kwargs):
        self.classes = classes
        self.anchors = anchors
        self.just_activate = just_activate
        self.net_dims = net_dims
        super(Yolo, self).__init__(**kwargs)

        self.anchors_w = np.array([a[0] for a in anchors], dtype=np.float32).reshape(1,1,1,len(anchors),1)
        self.anchors_h = np.array([a[1] for a in anchors], dtype=np.float32).reshape(1,1,1,len(anchors),1)

        self.net_height = np.int32(net_dims[0])
        self.net_width = np.int32(net_dims[1])

    def get_config(self):
        config = super(Yolo, self).get_config().copy()
        config.update({
            'classes': self.classes,
            'anchors': self.anchors,
            'net_dims': self.net_dims,
            'just_activate': self.just_activate,
        })
        return config

    def build(self, input_shape):
        super(Yolo, self).build(input_shape)

        self.l_h, self.l_w = input_shape[1], input_shape[2]
        self.l_a = len(self.anchors)

        self.range_w = np.arange(self.l_w, dtype=np.float32).reshape(1, 1, self.l_w, 1, 1)
        self.range_h = np.arange(self.l_h, dtype=np.float32).reshape(1, self.l_h, 1, 1, 1)

    def call(self, input):
        input = keras.backend.reshape(input, shape=(-1, self.l_h, self.l_w, self.l_a, 4+1+self.classes))
        if self.just_activate:
            # that's how far Darknet takes it within the network (when not training)
            return keras.backend.concatenate((
                keras.activations.sigmoid(input[...,:2]),   # x, y
                input[...,2:4],
                keras.activations.sigmoid(input[...,4:])    # objectness, array of class detections
            ))

        objectness = keras.activations.sigmoid(input[...,4:5])
        x = (keras.activations.sigmoid(input[...,0:1]) + self.range_w) / self.l_w
        y = (keras.activations.sigmoid(input[...,1:2]) + self.range_h) / self.l_h
        w = (keras.activations.exponential(input[...,2:3]) * self.anchors_w) / self.net_width
        h = (keras.activations.exponential(input[...,3:4]) * self.anchors_h) / self.net_height
        classes = keras.activations.sigmoid(input[...,5:]) * objectness

        return keras.backend.concatenate((x, y, w, h, objectness, classes))
