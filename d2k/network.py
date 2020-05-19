import tensorflow as tf
import numpy as np
import struct
import d2k.box


class ConfigurationError(Exception):
    pass


class ConversionError(Exception):
    pass


class Network:
    @staticmethod
    def parse(config_text):
        """Parses a Darknet network configuration file, returning a list of (section, options)
           tuples where 'section' is the section name, as in '[convolutional]', and 'options'
           is a dictionary with the various options.

           Raises ConfigurationError in case of errors.

           Arguments:
           config_text -- configuration file as a multiline string
        """
        parsed = []
        section = None
        options = None

        def typed(x):
            try:
                return int(x)
            except(ValueError):
                try:
                    return float(x)
                except(ValueError):
                    pass
            return x.strip()


        for line in config_text.splitlines():
            if len(line) == 0 or line[0] in ['#',';']: continue

            if line[0] == '[':
                options = {}
                section = (line, options)
                parsed.append(section)
            else:
                if (section is None):
                    raise ConfigurationError('Option assignment outside section in ' + line)

                key, value = line.partition('=')[::2]
                options[key.rstrip()] = typed(value) if ',' not in value else [typed(x) for x in value.split(',')]

        return parsed


    @staticmethod
    def _check_config(parsed):
        """Checks a parsed Darknet configuration for various errors and also provides defaults,
           so as to simplify working with it.

           Raises ConfigurationError in case of errors.

           Arguments:
           parsed -- result of Network.parse()
        """
        if len(parsed) == 0 or parsed[0][0] not in ['[net]', '[network]']:
            raise ConfigurationError('Initial section "[net]" (or "[network]") missing')

        if 'height' not in parsed[0][1]: # 0 in darknet (unusable)
            raise ConfigurationError('Network height undefined')

        if 'width' not in parsed[0][1]: # 0 in darknet (unusable)
            raise ConfigurationError('Network width undefined')

        if 'channels' not in parsed[0][1]: # 0 in darknet (unusable)
            raise ConfigurationError('Network channels undefined')

        defaults = {
            '[convolutional]': {
                'filters': 1,
                'size': 1,
                'stride': 1,
                'pad': 0,
                'activation': 'logistic',
                'batch_normalize': 0
            },
            '[shortcut]': {
                'activation': 'linear'
            },
            '[upsample]': {
                'stride': 2
            },
            '[maxpool]': {
                'stride': 1
            },
            '[yolo]': {
                'classes': 20,
                'num': 1,
                'nms_kind': 'default',
                'scale_x_y': 1.0
            }
        }

        for (i, (section, options)) in enumerate(parsed[1:]):
            if section in defaults:
                missing = defaults[section].keys() - options.keys()
                for m in missing:
                    options[m] = defaults[section][m]

                if section == '[yolo]' and 'mask' not in options:
                    options['mask'] = list(range(options['num']))

        for (i, (section, options)) in enumerate(parsed[1:]):
            if section == '[route]':
                if not 'layers' in options: raise ConfigurationError('[route] must specify source layers')

                layers = options['layers']
                if not isinstance(layers, list):
                    layers = [layers]
                options['layers'] = [x+i if x < 0 else x for x in layers]
                for l in options['layers']:
                    if l < 0 or l >= i:
                        raise ConfigurationError('[route] requests layer out of range')

            elif section == '[shortcut]' and 'from' in options:
                if options['from'] < 0:
                    options['from'] += i
                if options['from'] < 0 or options['from'] >= i:
                    raise ConfigurationError('[route] requests layer out of range')

            elif section == '[maxpool]':
                if not 'size' in options: options['size'] = options['stride']
                if not 'padding' in options: options['padding'] = options['size']-1

            elif section == '[yolo]':
                if 'mask' in options and not isinstance(options['mask'], list):
                    options['mask'] = [options['mask']]

                # Darknet's parser allows 'anchors' to be missing, but that just leads to the anchors
                # (which are stored in 'biases') to be all .5...  it seems more useful to error out
                if 'anchors' not in options:
                    raise ConfigurationError('[yolo] layer without anchors')

                anchors = options['anchors']

                if not isinstance(anchors, list) or len(anchors) % 2 == 1:
                    raise ConfigurationError('[yolo] with uneven number of anchors')

                options['anchors'] = [(anchors[2*i], anchors[2*i+1]) for i in range(len(anchors)//2)]

                for m in options['mask']:
                    if m < 0 or m >= len(options['anchors']):
                        raise ConfigurationError('[yolo] mask out of range')


    def __init__(self, config):
        __class__._check_config(config)
        self.config = config


    @staticmethod
    def load(config):
        """Loads a Darknet network configuration file, returning a list of (section, options)
           tuples where 'section' is the section name, as in '[convolutional]', and 'options'
           is a dictionary with the various options.

           Raises ConfigurationError in case of errors.

           Arguments:
           config -- configuration file, a multiline string
        """
        return Network(Network.parse(config))


    def input_shape(self):
        """Returns this Network's input shape."""
        net_options = self.config[0][1]
        return (net_options['height'], net_options['width'], net_options['channels'])


    def convert(self, decode_grid=True):
        """Returns a list of Python statements defining a Keras network equivalent to this Network.

           Raises ConversionError in case or errors.

           Arguments:
           decode_grid -- decode box coordinates from YOLO layers' grids.  Turn off to facilitate
                          comparing with Darknet (default: True)
        """
        def _checkSupported(options, supportedSet):
             assert isinstance(supportedSet, set)
             unsupported = options.keys() - supportedSet
             if len(unsupported) > 0: raise ConversionError('Unsupported option(s) ' + ' '.join(unsupported))

        net_options = self.config[0][1]

        assert tf.keras.backend.image_data_format() == 'channels_last'

        net = [f'layer_in = keras.Input(shape=({net_options["height"]}, {net_options["width"]}, {net_options["channels"]}))']
        prev_layer = 'layer_in'
        output_layers = set()

        for (i, (section, options)) in enumerate(self.config[1:]):

            try:
                if section == '[convolutional]':
                    filters = int(options['filters'])
                    size = int(options['size'])
                    stride = int(options['stride'])
                    pad = bool(options['pad'])
                    activation = options['activation']
                    batch_normalize = bool(options['batch_normalize'])

                    _checkSupported(options, {'filters', 'size', 'stride', 'pad', 'activation', 'batch_normalize'})

                    # Darknet's padding is always symmetrical, but not TensorFlow's, which favors right and bottom padding.
                    # See https://stackoverflow.com/questions/42924324/tensorflows-asymmetric-padding-assumptions
                    # and https://github.com/Microsoft/MMdnn/issues/153#issuecomment-387982630
                    if pad and size//2 > 0:
                        net.append(f'layer_{i} = keras.layers.ZeroPadding2D((({size//2},{size//2}),({size//2},{size//2})))' +
                                                                         f'({prev_layer})')
                        prev_layer = f'layer_{i}'

                    net.append(f'layer_{i} = keras.layers.Conv2D({filters}, {size}, strides={stride}, ' +
                               f'use_bias={not batch_normalize}, name=\'conv_{i}\')({prev_layer})')

                    if batch_normalize:
                        net.append(f'layer_{i} = keras.layers.BatchNormalization(epsilon=.00001, name=\'bn_{i}\')(layer_{i})')

                    if activation == 'leaky':
                        net.append(f'layer_{i} = keras.layers.LeakyReLU(alpha=.1)(layer_{i})')
                    elif activation == 'mish':
                        net.append(f'layer_{i} = layer_{i} * K.tanh(K.softplus(layer_{i}))')
                    elif activation != 'linear':
                        raise ConversionError(f'Unsupported activation "{activation}"')


                elif section == '[route]':
                    layers = list(options['layers'])

                    _checkSupported(options, {'layers'})

                    assert i>0, "[route] in layer 0 can't refer to any layers, so this shouldn't be loadable"
                    output_layers.add(i-1)
                    output_layers.difference_update(set(layers))

                    layers_names = [f'layer_{x}' for x in layers]

                    if len(layers_names) == 1:
                        net.append(f'layer_{i} = {layers_names[0]}')
                    else:
                        net.append(f'layer_{i} = keras.layers.Concatenate()([' + ', '.join(layers_names) + '])')

                elif section == '[shortcut]':
                    index = options['from']
                    activation = options['activation']

                    _checkSupported(options, {'from', 'activation'})

                    net.append(f'layer_{i} = keras.layers.Add()([layer_{index}, {prev_layer}])')

                    if activation != 'linear': raise ConversionError(f'Unsupported activation "{activation}"')

                elif section == '[upsample]':
                    stride = options['stride']
                    _checkSupported(options, {'stride'})

                    net.append(f'layer_{i} = keras.layers.UpSampling2D({stride})({prev_layer})')

                elif section == '[maxpool]':
                    stride = options['stride']
                    size = options['size']
                    padding = options['padding']
                    _checkSupported(options, {'stride', 'size', 'padding'})

                    if padding > 0:
                        # XXX is there an easy keras.backend equivalent of tf.pad?
                        net.append(f'layer_{i} = tf.pad({prev_layer}, tf.constant([[0,0],[{padding//2},{padding-padding//2}],' +
                                                                                 f'[{padding//2},{padding-padding//2}],[0,0]]), ' +
                                                                                                      f'constant_values=-np.inf)')
                        prev_layer = f'layer_{i}'

                    net.append(f'layer_{i} = keras.layers.MaxPool2D(pool_size={size}, strides={stride})({prev_layer})')

                elif section == '[yolo]':
                    classes = options['classes']
                    mask = options['mask']
                    anchors = [options['anchors'][m] for m in mask]
                    scale_x_y = options['scale_x_y']
                    nms_kind = options['nms_kind']
                    _checkSupported(options, {'num', 'classes', 'mask', 'anchors', 'scale_x_y', 'nms_kind',
                                              # those below are ignored (so far)
                                               'jitter', 'ignore_thresh', 'truth_thresh', 'random', 'beta_nms',
                                               'iou_loss', 'cls_normalizer', 'iou_normalizer', 'iou_thresh'}) # training only

                    if nms_kind not in ['default', 'greedynms']:
                        raise ConversionError(f'Unsupported nms_kind {nms_kind}')

                    L = f'layer_{i}'

                    net.append(f'{L} = K.reshape({prev_layer}, (-1, *K.int_shape({prev_layer})[1:3], {len(anchors)}, {4+1+classes}))')

                    if scale_x_y != 1.0:
                        net.append(f'{L}_scale_x_y = K.constant({scale_x_y}, dtype="float32")')

                    net.append(f'{L}_xy = keras.activations.sigmoid({L}[...,0:2])' +
                               (f' * {L}_scale_x_y - .5*({L}_scale_x_y - 1)' if scale_x_y != 1.0 else ''))
                    net.append(f'{L}_wh = {L}[...,2:4]')
                    net.append(f'{L}_obj_classes = keras.activations.sigmoid({L}[...,4:])')
                    net.append(f'{L} = K.concatenate(({L}_xy, {L}_wh, {L}_obj_classes))')

                    if decode_grid:
                        net.append(f'{L}_l_w, {L}_l_h = K.int_shape({L})[1:3]')
                        net.append(f'{L}_range_w = K.reshape(K.arange(0, {L}_l_w, dtype="float32"), (1, 1, {L}_l_w, 1, 1))')
                        net.append(f'{L}_range_h = K.reshape(K.arange(0, {L}_l_h, dtype="float32"), (1, {L}_l_h, 1, 1, 1))')

                        net.append(f'{L}_anchors_w = K.constant({[a[0] for a in anchors]}, dtype="float32", ' +
                                                                                         f'shape=(1,1,1,{len(anchors)},1))')
                        net.append(f'{L}_anchors_h = K.constant({[a[1] for a in anchors]}, dtype="float32", ' +
                                                                                         f'shape=(1,1,1,{len(anchors)},1))')

                        net.append(f'{L}_x = ({L}[...,0:1] + {L}_range_w) / {L}_l_w')
                        net.append(f'{L}_y = ({L}[...,1:2] + {L}_range_h) / {L}_l_h')
                        net.append(f'{L}_w = (K.exp({L}[...,2:3]) * {L}_anchors_w) / {net_options["width"]}')
                        net.append(f'{L}_h = (K.exp({L}[...,3:4]) * {L}_anchors_h) / {net_options["height"]}')
                        net.append(f'{L} = K.concatenate(({L}_x, {L}_y, {L}_w, {L}_h, {L}_obj_classes))')

                else:
                    raise ConversionError(f'Unsupported section {section}')

            except Exception as e:
                raise ConversionError(f'Error converting {section} layer {i}') from e

            prev_layer = f'layer_{i}'

        if prev_layer != 'layer_in':
            layer_out = [f'layer_{x}' for x in sorted(output_layers)]
            layer_out.append(prev_layer)
            net.append('layer_out = ' + ('[' if len(layer_out)>1 else '')
                                      + ', '.join(layer_out) +
                                        (']' if len(layer_out)>1 else ''))

        return net


    def _read_weights(self, model, weights):
        """Reads Darknet weights into a Keras model generated by this class."""

        class BinaryReader:
            def __init__(self, buffer):
                self.buffer = buffer
                self.offset = 0

            def read_int32(self, count=1):
                self.offset = self.offset + count*4;
                return struct.unpack(f'{count}i', self.buffer[self.offset - count*4:self.offset])

            def read_float32(self, shape):
                size = np.prod(shape) * 4
                self.offset = self.offset + size
                return np.ndarray(shape, dtype=np.float32, buffer=self.buffer[self.offset - size:self.offset])

            def at_eof(self):
                return len(self.buffer) == self.offset


        assert tf.keras.backend.backend() == 'tensorflow' # weight orders are TF-specific

        r = BinaryReader(weights)

        version = r.read_int32(3)
        if not version in [(0,2,0), (0,2,5)]:
            raise ConversionError(f'Unsupported weights format version {version}')

        r.read_int32(2) # "seen" (int64)

        def _order(items, order):
            return [items[x] for x in order]

        for (i, (section, options)) in enumerate(self.config[1:]):
            if section == '[convolutional]':
                batch_normalize = bool(options['batch_normalize'])

                conv = model.get_layer(f'conv_{i}')

                if batch_normalize:
                    bn = model.get_layer(f'bn_{i}')

                    # Darknet saves beta("bias"), gamma, mean, var
                    # TF needs it   gamma,        beta,  mean, var

                    shapes = _order([x.shape for x in bn.get_weights()], [1, 0, 2, 3])
                    weights = _order([r.read_float32(s) for s in shapes], [1, 0, 2, 3])

                    #print(f'bn_{i} from ', [x.shape for x in bn.get_weights()], ' to ', [x.shape for x in weights])
                    bn.set_weights(weights)
                else:
                    bias = r.read_float32(conv.get_weights()[1].shape)

                # Darknet saves (bias) (out_dim, in_dim, height, width)
                # TF needs it          (height,  width,  in_dim, out_dim) (bias)

                shapes = _order(conv.get_weights()[0].shape, [3, 2, 0, 1])
                weights = r.read_float32(shapes)
                weights = [weights.transpose([2, 3, 1, 0])]

                if not batch_normalize:
                    weights.append(bias)

                #print(f'conv_{i} from ', [x.shape for x in conv.get_weights()], ' to ', [x.shape for x in weights])

                conv.set_weights(weights)
            else:
                assert section in ['[route]', '[shortcut]', '[upsample]', '[maxpool]', '[yolo]'] # none of these have weights

        assert r.at_eof()


    def make_model(self, weights=None, decode_grid=True):
        """Returns a Keras model equivalent to this Network.

           Raises ConversionError in case of errors.

           Arguments:
           weights -- darknet weights to load into model, as a binary string
           decode_grid -- decode box coordinates from YOLO layers' grids.  Turn off to facilitate
                          comparing with Darknet (default: True)
        """

        import tensorflow.keras as keras

        g = globals().copy()
        l = locals().copy()
        exec('import tensorflow.keras.backend as K', g, l)
        exec('\n'.join(self.convert(decode_grid=decode_grid)), g, l)
        model = keras.Model(l['layer_in'], l['layer_out'])
        if weights != None: self._read_weights(model, weights)
        return model


    def decode_yolo_grid(self, output):
        """Decodes the box coordinates output by the YOLO grids on this network.
           Our models already performs this decoding (unless 'decode_grid=False' is passed),
           but Darknet's network doesn't.
        """
        yolo_config = [layer[1] for layer in self.config if layer[0] == '[yolo]']
        assert len(yolo_config) > 0, "No YOLO layers on this network"

        # We need to try to keep computations using float32/int32 to stay
        # as close to darknet as possible.

        net_width = np.int32(self.config[0][1]['width'])
        net_height = np.int32(self.config[0][1]['height'])

        result = []
        for l_out, l in zip(output, yolo_config):
            l_h, l_w = l_out.shape[0:2]
            l_m = len(l['mask'])
            l_out = l_out.reshape((l_h, l_w, l_m, l['classes']+4+1))

            l_anchors_w = np.array([l['anchors'][x][0] for x in l['mask']], dtype=np.float32).reshape(1,1,l_m,1)
            l_anchors_h = np.array([l['anchors'][x][1] for x in l['mask']], dtype=np.float32).reshape(1,1,l_m,1)

            # x = (col + l_out[row, col, m, 0]) / l_w
            x = (l_out[...,0:1] + np.arange(l_w, dtype=np.float32).reshape(1,l_w,1,1)) / l_w

            # y = (row + l_out[row, col, m, 1]) / l_h
            y = (l_out[...,1:2] + np.arange(l_h, dtype=np.float32).reshape(l_h,1,1,1)) / l_h

            # w = np.exp(l_out[row, col, m, 2]) * l_anchors[m][0] / net_width
            w = (np.exp(l_out[...,2:3], dtype=np.float32) * l_anchors_w) / net_width

            # h = np.exp(l_out[row, col, m, 3]) * l_anchors[m][1] / net_height
            h = (np.exp(l_out[...,3:4], dtype=np.float32) * l_anchors_h) / net_height

            obj_classes = l_out[...,4:]

            result.append(np.concatenate([x, y, w, h, obj_classes], axis=3))

        return result


def load(cfg):
    return Network.load(cfg)


def boxes_from_output(output, net_dim, img_dim, thresh=.5):
    """Extracts bounding boxes for detections from the model output (prediction).

       Arguments:
       output -- network/model output
       net_dim -- network/model input dimensions as (width, height)
       img_dim -- original image input dimensions as (width, height)
       thresh -- class detection threshold (default: .5)
    """

    boxes = []

    net_dim = np.array([*net_dim], dtype=np.int32)
    img_dim = np.array([*img_dim], dtype=np.int32)

    for l_out in output:
        objectness = l_out[...,4]
        detections = l_out[objectness >= thresh]

        objectness = detections[...,4:5]
        classes = detections[..., 5:]

        classes *= objectness
        classes[classes < thresh] = 0

        d2k.box.letterbox_transform(detections, net_dim, img_dim, reverse=True)

        xy = detections[...,0:2]
        wh = detections[...,2:4]

        xy *= img_dim
        wh *= img_dim

        boxes.extend(d2k.box.boxes_from_array(detections))

    return boxes


def detect_image(model, image, thresh=.5, iou_thresh=.5):
    """Performs object detection on image, returning a list of boxes with any detections.

       Arguments:
       model -- Keras model obtained from Network.make_model()
       image -- image to run detection on.
       thresh -- class detection threshold (default: .5)
       iou_thresh -- IOU threshold for non-max suppression (default: .5)
    """
    net_h, net_w = model.layers[0].input_shape[0][1:3]
    img_h, img_w = image.shape[0:2]

    image = d2k.image.letterbox(image, net_w, net_h)

    output = model.predict(np.expand_dims(image, axis=0))
    output = [x.squeeze(axis=0) for x in output]

    boxes = d2k.network.boxes_from_output(output, (net_w, net_h), (img_w, img_h), thresh=thresh)
    return d2k.box.nms_boxes(boxes, iou_thresh=iou_thresh)
