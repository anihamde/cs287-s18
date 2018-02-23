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

def escape(l):
    return l.replace("\"", "<quote>").replace(",", "<comma>")