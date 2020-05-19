import pytest
import d2k.box
from d2k.box import Box
import numpy as np


@pytest.mark.parametrize("b2_args, isect", [
                         [[100, 50, 60, 20], 60*20],
                         [[ 65, 70, 20, 30], 5*5],
                         [[ 65, 30, 20, 30], 5*5],
                         [[135, 70, 20, 30], 5*5],
                         [[135, 30, 20, 30], 5*5],
                         [[100, 20, 10, 10], 0],
                         [[ 20, 50, 10, 10], 0],
                        ])
def test_intersection(b2_args, isect):
    b1 = Box(100, 50, 60, 20)
    b2 = Box(*b2_args)

    assert isect == b1.intersection(b2)
    assert isect == b2.intersection(b1)


def test_intersection_wrong_parameter():
    b = Box(10, 10, 10, 10)

    with pytest.raises(AssertionError):
        b.intersection(10)


@pytest.mark.parametrize("b2_args, iou", [
                         [[100, 50, 60, 20], 1.],
                         [[ 65, 70, 20, 30], 5*5/(60*20+20*30-5*5)],
                         [[ 65, 30, 20, 30], 5*5/(60*20+20*30-5*5)],
                         [[135, 70, 20, 30], 5*5/(60*20+20*30-5*5)],
                         [[135, 30, 20, 30], 5*5/(60*20+20*30-5*5)],
                         [[100, 20, 10, 10], 0],
                         [[ 20, 50, 10, 10], 0],
                        ])
def test_iou(b2_args, iou):
    b1 = Box(100, 50, 60, 20)
    b2 = Box(*b2_args)

    assert iou == b1.iou(b2)
    assert iou == b2.iou(b1)


def test_iou_wrong_parameter():
    b = Box(10, 10, 10, 10)

    with pytest.raises(AssertionError):
        b.iou(10)


def test_corners():
    assert Box(100, 50, 60, 20).corners() == (70, 40, 130, 60)
    assert Box(100, 50, 60, 15).corners() == (70, 42, 130, 57)


def test_from_array():
    array = np.array([ 1,  2,  3,  4,  5,  6,  7,  8], dtype=np.float32)

    b = Box.from_array(array)

    assert 1 == b.x
    assert isinstance(b.x, float)
    assert 2 == b.y
    assert isinstance(b.y, float)
    assert 3 == b.w
    assert isinstance(b.w, float)
    assert 4 == b.h
    assert isinstance(b.h, float)
    assert 5 == b.objectness
    assert isinstance(b.objectness, float)
    assert [float(x) for x in range(6,9)] == b.classes
    for c in b.classes: assert isinstance(c, float)


def test_boxes_from_array():
    array = np.array([[ 1,  2,  3,  4,  5,  6,  7,  8],
                      [11, 12, 13, 14, 15, 16, 17, 18],
                      [21, 22, 23, 24, 25, 26, 27, 28]], dtype=np.float32)

    boxes = d2k.box.boxes_from_array(array)

    assert 3 == len(boxes)

    for i, b in enumerate(boxes):
        assert float(f'{i}1') == b.x
        assert isinstance(b.x, float)
        assert float(f'{i}2') == b.y
        assert isinstance(b.y, float)
        assert float(f'{i}3') == b.w
        assert isinstance(b.w, float)
        assert float(f'{i}4') == b.h
        assert isinstance(b.h, float)
        assert float(f'{i}5') == b.objectness
        assert isinstance(b.objectness, float)
        assert [float(f'{i}{x}') for x in range(6,9)] == b.classes
        for c in b.classes: assert isinstance(c, float)


def test_to_list():
    array = np.array([ 1,  2,  3,  4,  5,  6,  7,  8], dtype=np.float32)

    b = Box.from_array(array)
    assert [ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.] == b.to_list()


def test_nms_boxes():
    array = np.array([[105, 55, 60, 20, .5, .6, .8],    # top box for class 1
                      [100, 50, 60, 20, .5, .9, .7],    # top box for class 0
                      [ 95, 55, 60, 20, .5, .3, .5],    # high IOU, not top, should go away
                      [100, 20, 10, 10, .5, .8, .4],    # 0 IOU to those above
                      [100, 20, 10, 10, .0, .9, .0],    # unrealistic (0 objectness), just to check optimization
                     ], dtype=np.float32)

    boxes = d2k.box.boxes_from_array(array)
    nms = d2k.box.nms_boxes(boxes, iou_thresh=.5)

    nms = sorted(nms)   # because sort within nms may not be stable

    nms = np.array([b.to_list() for b in nms])

    np.testing.assert_almost_equal(np.array([
                      [100, 20, 10, 10, .5, .8, .4],
                      [100, 50, 60, 20, .5, .9, .0],
                      [105, 55, 60, 20, .5, .0, .8],
                     ]), nms, decimal=7)


def test_nms_boxes_dont_remove_all_zeros():
    array = np.array([[105, 55, 60, 20, .5, .6, .8],
                      [100, 50, 60, 20, .5, .9, .7],
                      [ 95, 55, 60, 20, .5, .3, .5],
                      [100, 20, 10, 10, .5, .8, .4],
                      [100, 20, 10, 10, .0, .9, .0],
                     ], dtype=np.float32)

    boxes = d2k.box.boxes_from_array(array)
    nms = d2k.box.nms_boxes(boxes, iou_thresh=.5, remove_all_zeros=False)

    nms = sorted(nms)   # because sort within nms may not be stable

    nms = np.array([b.to_list() for b in nms])

    np.testing.assert_almost_equal(np.array([
                      [ 95, 55, 60, 20, .5, .0, .0],
                      [100, 20, 10, 10, .5, .8, .4],
                      [100, 50, 60, 20, .5, .9, .0],
                      [105, 55, 60, 20, .5, .0, .8],
                     ]), nms, decimal=7)


def test_nms_boxes_thresh():
    array = np.array([[105, 55, 60, 20, .5, .6, .8],    # top box for class 1
                      [100, 50, 60, 20, .5, .9, .7],    # top box for class 0
                      [100, 50, 60, 20, .5, .8, .6],    # 1.0 IOU, not top
                      [ 95, 55, 60, 20, .5, .3, .5],    # high but not 1.0 IOU
                      [100, 20, 10, 10, .5, .8, .4],    # 0 IOU to those above
                      [100, 20, 10, 10, .0, .9, .0],    # unrealistic (0 objectness), just to check optimization
                     ], dtype=np.float32)

    boxes = d2k.box.boxes_from_array(array)
    nms = d2k.box.nms_boxes(boxes, iou_thresh=.9)

    nms = sorted(nms)   # because sort within nms may not be stable

    nms = np.array([b.to_list() for b in nms])

    np.testing.assert_almost_equal(np.array([
                      [ 95, 55, 60, 20, .5, .3, .5],
                      [100, 20, 10, 10, .5, .8, .4],
                      [100, 50, 60, 20, .5, .9, .7],
                      [105, 55, 60, 20, .5, .6, .8],
                     ]), nms, decimal=7)


def test_letterbox_transform_landscape():
    img_dim = np.array([200, 100])
    net_dim = np.array([50, 50])

    boxes = np.array([[100., 10., 100., 50., 1., 1.],
                      [100., 20.,  10., 10., 1., 1.],
                      [20.,  50.,  10., 10., 1., 1.]])

    boxes[...,0:2] /= img_dim
    boxes[...,2:4] /= img_dim

    d2k.box.letterbox_transform(boxes, img_dim, net_dim)

    boxes[...,0:2] *= net_dim
    boxes[...,2:4] *= net_dim

    np.testing.assert_almost_equal(boxes, np.array(
                    [[25., 15. ,  25., 12.5,  1.,  1.],
                     [25., 17.5,  2.5,  2.5,  1.,  1.],
                     [ 5., 25. ,  2.5,  2.5,  1.,  1.]]))

    boxes[...,0:2] /= net_dim
    boxes[...,2:4] /= net_dim

    d2k.box.letterbox_transform(boxes, img_dim, net_dim, reverse=True)

    boxes[...,0:2] *= img_dim
    boxes[...,2:4] *= img_dim

    np.testing.assert_almost_equal(boxes, np.array(
                     [[100., 10., 100., 50., 1., 1.],
                      [100., 20.,  10., 10., 1., 1.],
                      [20.,  50.,  10., 10., 1., 1.]]))


def test_letterbox_transform_portrait():
    img_dim = np.array([200, 400])
    net_dim = np.array([50, 50])

    boxes = np.array([[100., 10., 100., 50., 1., 1.],
                      [100., 20.,  10., 10., 1., 1.],
                      [20.,  50.,  10., 10., 1., 1.]])

    boxes[...,0:2] /= img_dim
    boxes[...,2:4] /= img_dim

    d2k.box.letterbox_transform(boxes, img_dim, net_dim)

    boxes[...,0:2] *= net_dim
    boxes[...,2:4] *= net_dim

    np.testing.assert_almost_equal(boxes, np.array(
                    [[25.,  1.25, 12.5 ,  6.25,  1.,  1.],
                     [25.,  2.5 ,  1.25,  1.25,  1.,  1.],
                     [15.,  6.25,  1.25,  1.25,  1.,  1.]]))

    boxes[...,0:2] /= net_dim
    boxes[...,2:4] /= net_dim

    d2k.box.letterbox_transform(boxes, img_dim, net_dim, reverse=True)

    boxes[...,0:2] *= img_dim
    boxes[...,2:4] *= img_dim

    np.testing.assert_almost_equal(boxes, np.array(
                     [[100., 10., 100., 50., 1., 1.],
                      [100., 20.,  10., 10., 1., 1.],
                      [20.,  50.,  10., 10., 1., 1.]]))
