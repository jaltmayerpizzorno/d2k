import sys
import os
import d2k
import tensorflow.keras as keras
from PIL import Image
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # only error messages, please

if len(sys.argv) < 2:
    print('Use:', sys.argv[0], 'filename')
    sys.exit(2)

image_file = sys.argv[1]

saved_model = Path('yolov3.h5')

if (saved_model.exists()):
    model = keras.models.load_model(saved_model, custom_objects={'Yolo': d2k.layers.Yolo})
else:
    network = d2k.network.load(Path('darknet/yolov3.cfg').read_text())
    model = network.make_model(Path('darknet/yolov3.weights').read_bytes())
    model.save(saved_model)

image = d2k.image.load(image_file)
boxes = d2k.network.detect_image(model, image)

names = Path('darknet/coco.names').read_text().splitlines()

for b in boxes:
    print(f'{str(b.corners()):<25}', ' '.join([f'{names[i]} {c:>5.2f}' for i, c in enumerate(b.classes) if c > 0.]))

im = Image.open(image_file)

d2k.box.draw_boxes(im, boxes, names=names)

im.show()
