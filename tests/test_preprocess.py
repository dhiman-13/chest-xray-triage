import cv2
import numpy as np

from cxr.preprocess import (
    STAT_COLUMNS,
    STORE_SIZE,
    decode_image,
    image_stats,
    pad_to_square,
    resize_for_storage,
    to_model_input,
)


def test_decode_png_to_gray_uint8(png_bytes):
    img = decode_image(png_bytes)
    assert img.dtype == np.uint8 and img.ndim == 2 and img.shape == (300, 240)


def test_decode_16bit_and_rgb():
    img16 = (np.linspace(0, 65535, 100 * 100).reshape(100, 100)).astype(np.uint16)
    out = decode_image(cv2.imencode(".png", img16)[1].tobytes())
    assert out.dtype == np.uint8 and out.max() == 255 and out.min() == 0
    rgb = np.zeros((50, 60, 3), np.uint8)
    assert decode_image(cv2.imencode(".png", rgb)[1].tobytes()).shape == (50, 60)


def test_pad_to_square_keeps_content_centered():
    img = np.full((100, 50), 200, np.uint8)
    sq = pad_to_square(img)
    assert sq.shape == (100, 100)
    assert sq[:, 25:75].min() == 200 and sq[:, :25].max() == 0


def test_model_input_shape_and_normalisation():
    img = np.full((500, 400), 128, np.uint8)
    x = to_model_input(img, 224)
    assert x.shape == (3, 224, 224) and x.dtype == np.float32
    # same result whether we start from the raw image or the cached 256px version
    np.testing.assert_allclose(x, to_model_input(resize_for_storage(img), 224), atol=1e-5)
    assert resize_for_storage(img).shape == (STORE_SIZE, STORE_SIZE)


def test_image_stats_keys():
    stats = image_stats(np.random.default_rng(0).integers(0, 255, (256, 256), dtype=np.uint8))
    assert list(stats) == STAT_COLUMNS
