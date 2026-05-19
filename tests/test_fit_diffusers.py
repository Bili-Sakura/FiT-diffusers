import pytest
torch = pytest.importorskip("torch")
from diffusers_fit import FiTTransformer2DModel, create_transport
from diffusers_fit.schedulers.fit_transport import Sampler

def test_forward_sit():
    m = FiTTransformer2DModel(context_size=256, patch_size=2, in_channels=4, hidden_size=32, depth=2, num_heads=4, num_classes=10, learn_sigma=False, use_sit=True, online_rope=True)
    x = torch.randn(1, 16, 16)
    out = m(x, torch.tensor([0.5]), torch.tensor([0]), torch.zeros(1, 2, 16).long(), torch.ones(1, 16))
    assert out.shape == x.shape

def test_create_transport():
    t = create_transport(path_type="Linear", prediction="velocity", snr_type="lognorm")
    Sampler(t)
