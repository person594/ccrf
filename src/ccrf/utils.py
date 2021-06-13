import torch

def logmmexp(a, b):
    """
    Multiplies two matricies in logarithmic space.
    Returns log(exp(a) @ exp(b)), where a and b are matrices,
    and the logarithms and exponents are calculated component-wise.
    This calculation should be numerically stable.
    """
    m, k = a.shape
    k2, n = b.shape
    assert k == k2
    c = torch.zeros(m, k, n, device=a.device)
    c = c + a.view(m, k, 1)
    c = c + b.view(1, k, n)
    return torch.logsumexp(c, 1)
