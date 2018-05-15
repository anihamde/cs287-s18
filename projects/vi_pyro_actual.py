import argparse
import numpy as np
import torch
import torch.nn as nn
import visdom
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
# from utils.mnist_cached import MNISTCached as MNIST
# from utils.mnist_cached import setup_data_loaders
from helpers_pyro import mnist_test_tsne, plot_llk, plot_vae_samples
from vi_model import *

parser = argparse.ArgumentParser(description="parse args")
parser.add_argument('-n', '--num-epochs', default=101, type=int, help='number of training epochs')
parser.add_argument('-tf', '--test-frequency', default=5, type=int, help='how often we evaluate the test set')
parser.add_argument('-lr', '--learning-rate', default=1.0e-3, type=float, help='learning rate')
parser.add_argument('--cuda', action='store_true', default=False, help='whether to use cuda')
parser.add_argument('-visdom', '--visdom_flag', action="store_true", help='Whether plotting in visdom is desired')
parser.add_argument('-i-tsne', '--tsne_iter', default=100, type=int, help='epoch when tsne visualization runs')
args = parser.parse_args()

expn_pth = '~/57epigenomes.RPKM.pc' # change for GCP
# expn_pth = '/n/data_02/Basset/data/expn/roadmap/57epigenomes.RPKM.pc'
print("Reading gene expression data from:\n{}".format(expn_pth))
# Gene expression dataset
expn = pd.read_table(expn_pth,header=0)
col_names = expn.columns.values[1:]
expn = expn.drop(col_names[-1],axis=1) # 19795*57 right now # TODO: is this all right?
expn.columns = col_names
pinned_lookup = torch.nn.Embedding.from_pretrained(torch.FloatTensor(expn.as_matrix().T),freeze=True)
pinned_lookup.cuda()

torch.manual_seed(3435)
imgs = torch.poisson(pinned_lookup.weight) # discretize data
# imgs = pinned_lookup.weight.round()
# imgs = pinned_lookup.weight
dat = torch.utils.data.TensorDataset(imgs, torch.zeros(57,1)) # placeholder arg required pytorch <0.4.0...
loader = torch.utils.data.DataLoader(dat, batch_size=args.batch_size, shuffle=True)
print(next(iter(loader))[0].size())

# setup the VAE
vae = PyroVAE()
adam_args = {"lr": args.learning_rate}
optimizer = Adam(adam_args)
svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())

# setup visdom for visualization
if args.visdom_flag:
    vis = visdom.Visdom()

train_elbo = []
test_elbo = []
# training loop
for epoch in range(args.num_epochs):
    # initialize loss accumulator
    epoch_loss = 0.
    # do a training epoch over each mini-batch x returned
    # by the data loader
    for x,_ in loader:
        # do ELBO gradient and accumulate loss
        epoch_loss += svi.step(x)

    # report training diagnostics
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = epoch_loss / normalizer_train
    train_elbo.append(total_epoch_loss_train)
    print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))

    torch.save(vae.state_dict(), args.model_file)

#     if epoch % args.test_frequency == 0:
#         # initialize loss accumulator
#         test_loss = 0.
#         # compute the loss over the entire test set
#         for i, (x, _) in enumerate(test_loader):
#             # if on GPU put mini-batch into CUDA memory
#             if args.cuda:
#                 x = x.cuda()
#             # compute ELBO estimate and accumulate loss
#             test_loss += svi.evaluate_loss(x)

#             # pick three random test images from the first mini-batch and
#             # visualize how well we're reconstructing them
#             if i == 0:
#                 if args.visdom_flag:
#                     plot_vae_samples(vae, vis)
#                     reco_indices = np.random.randint(0, x.size(0), 3)
#                     for index in reco_indices:
#                         test_img = x[index, :]
#                         reco_img = vae.reconstruct_img(test_img)
#                         vis.image(test_img.reshape(28, 28).detach().cpu().numpy(),
#                                   opts={'caption': 'test image'})
#                         vis.image(reco_img.reshape(28, 28).detach().cpu().numpy(),
#                                   opts={'caption': 'reconstructed image'})

#         # report test diagnostics
#         normalizer_test = len(test_loader.dataset)
#         total_epoch_loss_test = test_loss / normalizer_test
#         test_elbo.append(total_epoch_loss_test)
#         print("[epoch %03d]  average test loss: %.4f" % (epoch, total_epoch_loss_test))

#     if epoch == args.tsne_iter:
#         mnist_test_tsne(vae=vae, test_loader=test_loader)
#         plot_llk(np.array(train_elbo), np.array(test_elbo))

# return vae
