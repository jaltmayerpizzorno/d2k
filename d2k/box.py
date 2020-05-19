import numpy as np
from PIL import ImageDraw, ImageFont

class Box:
    """Implements a bounding box resulting from YOLO object detection."""

    def __init__(self, x, y, w, h, objectness=0., classes=[]):
        """Instantiates a bounding box.

        Arguments:
        x, y -- coordinates for center of the box
        w, h -- width and height
        objectness -- likelihood object detection
        classes -- likelihoods for detection of the various classes
        """
        self.x = float(x)
        self.y = float(y)
        self.w = float(w)
        self.h = float(h)
        self.objectness = float(objectness)
        self.classes = [float(x) for x in classes]

    @staticmethod
    def from_array(array):
        return Box(*array[0:4], objectness=array[4], classes=array[5:])

    def intersection(self, other):
        """Returns by how much this and another Box intersect."""
        assert isinstance(other, Box)

        def overlap(x1, w1, x2, w2):
            left  = max(x1 - w1/2, x2 - w2/2)
            right = min(x1 + w1/2, x2 + w2/2)
            return right - left

        w_overlap = overlap(self.x, self.w, other.x, other.w)
        h_overlap = overlap(self.y, self.h, other.y, other.h)

        return w_overlap*h_overlap if w_overlap > 0 and h_overlap > 0 else 0

    def iou(self, other):
        """Returns this' and another Box's Intersection Over Union."""
        assert isinstance(other, Box)

        isect = self.intersection(other)
        union = self.w*self.h + other.w*other.h - isect
        return isect/union

    def corners(self):
        """Returns this Box's corners as (left, top, right, bottom)."""
        left   = int(self.x - self.w/2.)
        right  = int(self.x + self.w/2.)
        top    = int(self.y - self.h/2.)
        bottom = int(self.y + self.h/2.)
        return (left, top, right, bottom)

    def to_list(self):
        """Returns a list of this Box's contents to facilitate comparisons."""
        return [self.x, self.y, self.w, self.h, self.objectness, *self.classes]

    def __lt__(self, other):
        """Less-than operator to facilitate using sorted(), etc."""
        assert isinstance(other, Box)
        return self.to_list() < other.to_list()

    def __eq__(self, other):
        """Equality operator to facilitate using sorted(), etc."""
        assert isinstance(other, Box)
        # if we didn't need __lt__ we could use vars(self) == vars(other)
        return self.to_list() == other.to_list()

    def __repr__(self):
        return f'd2k.box.Box(({self.x},{self.y},{self.w},{self.h}),{self.objectness},{self.classes})'


def nms_boxes(boxes, iou_thresh=.5, remove_all_zeros=True):
    boxes = [b for b in boxes if b.objectness > 0]

    n_classes = len(boxes[0].classes) if len(boxes) > 0 else 0

    for k in range(n_classes):
        boxes = sorted(boxes, key=lambda box: box.classes[k], reverse=True)
        for i, box in enumerate(boxes):
            if box.classes[k] == .0: continue   # optimization

            for box2 in boxes[i+1:]:
                if box.iou(box2) > iou_thresh:
                    box2.classes[k] = .0

    if remove_all_zeros:
        boxes = [b for b in boxes if sum(b.classes) > 0]

    return boxes


def boxes_from_array(boxes_array):
    return [Box.from_array(b) for b in boxes_array]


def letterbox_transform(boxes, img_dim, net_dim, reverse=False):
    """Transforms the (relative, i.e., 0..1) box coordinates corresponding to "letterboxing" an image,
       i.e., resizing and padding it so that it fits the network's dimensions while retaining the aspect
       ratio.  See d2k.image.letterbox

       Arguments:
       boxes -- NumPy array of boxes starting with [x,y,w,h]
       img_dim -- 1x2 NumPy array with the image's width and height
       net_dim -- 1x2 NumPy array with the network's width and height
       reverse -- whether to reverse the transformation (default: False)
    """

    xy = boxes[...,0:2]
    wh = boxes[...,2:4]
       
    img_w, img_h = img_dim
    net_w, net_h = net_dim

    if (net_w/img_w) < (net_h/img_h):
        # landscape: emb_w = net_w
        emb_h = (img_h * net_w) // img_w

        shift = np.array([0, (net_h - emb_h)/np.float32(2.)/net_h])
        scale = np.array([1, emb_h/net_h])
    else:
        # portrait: emb_h = net_h
        emb_w = (img_w * net_h) // img_h

        shift = np.array([(net_w - emb_w)/np.float32(2.)/net_w, 0])
        scale = np.array([emb_w/net_w, 1])

    if reverse:
        xy -= shift
        xy /= scale
        wh /= scale
    else:
        xy *= scale
        xy += shift
        wh *= scale


def draw_boxes(im, boxes, names=None):
    if len(boxes) == 0: return

    n_classes = len(boxes[0].classes)
    classes_found = list({i for i in range(n_classes) for b in boxes if b.classes[i] > 0})
    hue = {classes_found[i]: int(360*i/len(classes_found)) for i in range(len(classes_found))}

    if names == None:
        names = [f'c={i}' for i in range(n_classes)]
    else:
        assert n_classes == len(names), f'{n_classes} != {len(names)}'

    # try to avoid really thin lines on big images and thick lines on little ones
    line_width = round(1+max(im.size[0], im.size[1])/500)

    try:
        font = ImageFont.truetype('Helvetica.ttc', size=round(6+4*line_width))
    except OSError:
        font = None # use default

    draw = ImageDraw.Draw(im)

    b_texts = []
    for b in boxes:
        b_classes = [i for i in range(n_classes) if b.classes[i] > 0]
        if b_classes == []: continue

        b_color = f'hsl({hue[b_classes[0]]},100%,70%)'

        draw.rectangle(b.corners(), outline=b_color, width=line_width)
        b_left, b_top = b.corners()[:2]

        b_text = ' '.join([f'{names[i]} {b.classes[i]:.2}' for i in b_classes])
        b_textsize = draw.textsize(b_text, font=font)

        draw.rectangle((b_left, b_top-b_textsize[1]-line_width, b_left+b_textsize[0]+2*line_width, b_top), fill=b_color)
        b_texts.append(((b_left+line_width, b_top-b_textsize[1]), b_text))

    # draw label texts last to try to avoid overwriting by other boxes
    for t in b_texts:
        draw.text(t[0], t[1], fill=(0,0,0), font=font)
