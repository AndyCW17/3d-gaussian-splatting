import torch
from splatproj.training.loss import l1_loss, ssim, training_loss


def test_identical_images_give_zero_l1():
    img = torch.rand(32, 32, 3)
    assert l1_loss(img, img.clone()).item() == 0.0


def test_identical_images_give_ssim_one():
    img = torch.rand(32, 32, 3)
    result = ssim(img, img.clone())
    assert abs(result.item() - 1.0) < 1e-5


def test_identical_images_give_zero_training_loss():
    img = torch.rand(32, 32, 3)
    assert training_loss(img, img.clone()).item() < 1e-5


def test_different_images_give_positive_loss():
    torch.manual_seed(0)
    a = torch.rand(32, 32, 3)
    b = torch.rand(32, 32, 3)
    assert training_loss(a, b).item() > 0.0


def test_loss_is_differentiable():
    # This is the property that actually matters for training: the loss
    # must produce gradients we can backpropagate through.
    torch.manual_seed(0)
    rendered = torch.rand(16, 16, 3, requires_grad=True)
    target = torch.rand(16, 16, 3)

    loss = training_loss(rendered, target)
    loss.backward()

    assert rendered.grad is not None
    assert not torch.isnan(rendered.grad).any()