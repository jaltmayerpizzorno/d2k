import sys
import os
import d2k
import tensorflow.keras as keras
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # only error messages, please

if len(sys.argv) < 2:
    print('Use: ' + sys.argv[0] + ' filename')
    sys.exit(2)

image_file = sys.argv[1]

if (os.path.exists('yolov3.h5')):
    model = keras.models.load_model('yolov3.h5', custom_objects={'Yolo': d2k.layers.Yolo})
else:
    with open('darknet/yolov3.cfg', 'r') as f:
        network = d2k.network.load(f.read())

    with open('darknet/yolov3.weights', 'rb') as f:
        weights = f.read()

    model = network.make_model(weights)
    model.save('yolov3.h5')

image = d2k.image.load(image_file)
boxes = d2k.network.detect_image(model, image)

with open('darknet/coco.names', 'r') as f:
    names = f.read().splitlines()

for b in boxes:
    print(f'{str(b.corners()):<25}', ' '.join([f'{names[i]} {c:>5.2f}' for i, c in enumerate(b.classes) if c > 0.]))

im = Image.open(image_file)

d2k.box.draw_boxes(im, boxes, names=names)

im.show()