import os
import argparse
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # only error messages, please

argparser = argparse.ArgumentParser()
argparser.add_argument('--version', type=int, help='select the YOLO version',
                       choices=[3, 4], default=3)
argparser.add_argument('--save', help='save the model as an .h5 file', action='store_true')
argparser.add_argument('--print', help='print generated model', action='store_true')
argparser.add_argument('image_file', type=str, nargs='?', help='image to process')
args = argparser.parse_args()

if not (args.print or args.save or args.image_file != None):
    argparser.print_usage()
    quit()

saved_model = Path(f'yolo{args.version}.h5')

import d2k
from PIL import Image

if (args.image_file != None and not args.print and saved_model.exists()):
    import tensorflow.keras as keras
    model = keras.models.load_model(saved_model)
else:
    network = d2k.network.load(Path(f'darknet-files/yolov{args.version}.cfg').read_text())
    model = network.make_model(Path(f'darknet-files/yolov{args.version}.weights').read_bytes())

    if (args.print):
        print('\n'.join(network.convert()), '\n')

    if (args.save):
        model.save(str(saved_model))

if args.image_file != None:
    image = d2k.image.load(args.image_file)
    boxes = d2k.network.detect_image(model, image)

    names = Path('darknet-files/coco.names').read_text().splitlines()

    for b in sorted(boxes):
        print(f'{str(b.corners()):<25}', ' '.join([f'{names[i]} {c:>5.2f}' for i, c in enumerate(b.classes) if c > 0.]))

    im = Image.open(args.image_file)

    d2k.box.draw_boxes(im, boxes, names=names)

    im.show()
