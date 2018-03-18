import torch
import torch.distributions
import time
import math

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since):
    now = time.time()
    s = now - since
    # es = s / (percent)
    # rs = es - s
    return '%s' % (asMinutes(s))

# only for normal-normal!
# dude i literally copied this out of the pytorch source code
def kl_divergence(p, q):
    var_ratio = (p.scale / q.scale).pow(2)
    t1 = ((p.loc - q.loc) / q.scale).pow(2)
    return 0.5 * (var_ratio + t1 - 1 - var_ratio.log())