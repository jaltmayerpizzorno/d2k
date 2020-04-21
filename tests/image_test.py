import pytest
import numpy as np
from darknet import darknet
import d2k.image
from PIL import Image
from pathlib import Path


def setup_function():
    np.random.seed(0)   # to make tests reproducible


def test_load(tmp_path):
    image = Image.fromarray(np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8), 'RGB')
    image_file = tmp_path / "image.bmp"
    with open(image_file, "wb") as f:
        image.save(f, format='BMP')

    dn_image = darknet.image.load(image_file)
    dn_image = dn_image.to_array()
    print("dn_image=", dn_image, "shape:", dn_image.shape, "flags:", dn_image.flags)

    k_image = d2k.image.load(image_file)
    print("k_image=", k_image, "shape:", k_image.shape)

    np.testing.assert_equal(dn_image, k_image)


@pytest.mark.parametrize("width, height", [(100,300), (300,100), (100,200), (200,100), (400, 600)])
def test_resize(tmp_path, width, height):
    image = Image.fromarray(np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8), 'RGB')
    image_file = tmp_path / "image.bmp"
    with open(image_file, "wb") as f:
        image.save(f, format='BMP')

    dn_image = darknet.image.load(image_file, width, height)
    dn_image = dn_image.to_array()

    k_image = d2k.image.load(image_file)
    k_image = d2k.image.resize(k_image, width, height)

    assert np.max(k_image) <= 1.0   # normalized?

    np.testing.assert_almost_equal(dn_image, k_image, decimal=7)


@pytest.mark.parametrize("width, height", [(100,200), (200,100)])
def test_letterbox(tmp_path, width, height):
    image = Image.fromarray(np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8), 'RGB')
    image_file = tmp_path / "image.bmp"
    with open(image_file, "wb") as f:
        image.save(f, format='BMP')

    dn_image = darknet.image.load(image_file)
    dn_image = dn_image.letterbox(width, height)
    dn_image = dn_image.to_array()

    k_image = d2k.image.load(image_file)
    k_image = d2k.image.letterbox(k_image, width, height)

    assert np.max(k_image) <= 1.0   # normalized?

    np.testing.assert_almost_equal(dn_image, k_image, decimal=7)
