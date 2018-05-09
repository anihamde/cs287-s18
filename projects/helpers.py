import torch
import time
import math
import torch
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import roc_curve, auc, precision_recall_curve

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

# source: https://github.com/pytorch/pytorch/issues/229
def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, 
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)

def lstm_hidden(laydir,bs,hiddensz):
    return (Variable(torch.zeros(laydir,bs,hiddensz).cuda()), Variable(torch.zeros(laydir,bs,hiddensz).cuda()))

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

def unpackage_hidden(h):
    """Unwraps hidden states into Tensors."""
    if type(h) == Variable:
        return h.data
    else:
        return tuple(unpackage_hidden(v) for v in h)

def freeze_model(model):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model

#def calc_auc(model, y_test, y_score):
#    n_classes = Y_test.shape[1]
#    fpr = dict()
#    tpr = dict()
#    roc_auc = dict()
#    for i in range(n_classes):
#        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
#        roc_auc[i] = auc(fpr[i], tpr[i])
#    # Compute micro-average ROC curve and ROC area
#    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
#    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#    return roc_auc["micro"]

def calc_auc(model, y_test, y_score, auctype = "ROC"):
    y_score = 1 / ( 1 + np.exp(-y_score) ) # sigmoid it!
    n_classes = y_test.shape[1] # 164
    if auctype == "ROC":
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        return roc_auc["micro"]
    elif auctype == "PR":
        prec = dict()
        rec = dict()
        pr_auc = dict()
        for i in range(n_classes):
            prec[i], rec[i], _ = precision_recall_curve(y_test[:,i], y_score[:,i])
            pr_auc[i] = auc(rec[i], prec[i])
        # Compute micro-average prec-rec curve and prec-rec AUC
        prec["micro"], rec["micro"], _ = precision_recall_curve(y_test.ravel(), y_score.ravel())
        pr_auc["micro"] = auc(rec["micro"], prec["micro"])
        return pr_auc["micro"]
