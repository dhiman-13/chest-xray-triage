import numpy as np
import torch

from cxr.explain import gradcam
from cxr.labels import NUM_CLASSES
from cxr.models.classifier import CXRClassifier
from cxr.models.losses import FocalLoss, build_loss


def test_forward_and_training_step():
    model = CXRClassifier(backbone="resnet18", pretrained=False, pos_weight=[2.0] * NUM_CLASSES)
    x, y = torch.randn(4, 3, 64, 64), torch.randint(0, 2, (4, NUM_CLASSES)).float()
    assert model(x).shape == (4, NUM_CLASSES)
    loss = model.criterion(model(x), y)
    loss.backward()
    assert torch.isfinite(loss)


def test_focal_loss_downweights_easy_examples():
    y = torch.ones(1, 1)
    easy, hard = torch.tensor([[4.0]]), torch.tensor([[-1.0]])
    fl, bce = FocalLoss(gamma=2.0), build_loss("bce", None)
    assert fl(easy, y) / bce(easy, y) < fl(hard, y) / bce(hard, y)


def test_gradcam_shape_and_range():
    net = CXRClassifier(backbone="resnet18", pretrained=False).net
    cam = gradcam(net, torch.randn(1, 3, 64, 64), class_idx=3)
    assert cam.shape == (64, 64) and 0 <= cam.min() and cam.max() <= 1 + 1e-6
    assert np.isfinite(cam).all()
