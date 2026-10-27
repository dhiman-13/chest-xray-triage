import cv2
import numpy as np
import pytest


@pytest.fixture(scope="session")
def tiny_bundle(tmp_path_factory):
    """A real ONNX bundle exported from a small random network (no training needed)."""
    from cxr.export import build_bundle
    from cxr.models.classifier import create_backbone

    out = tmp_path_factory.mktemp("bundle")
    net = create_backbone("resnet18", pretrained=False)
    build_bundle(net, out, img_size=64, backbone="resnet18", version="test")
    return out


@pytest.fixture
def png_bytes():
    rng = np.random.default_rng(0)
    img = (rng.random((300, 240)) * 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()
