import torch
from tqdm import tqdm

import numpy as np
from torch.utils import data
from torchvision.datasets import MNIST
import time

from scipy.stats import norm, multivariate_normal

import warnings
warnings.filterwarnings('ignore')


def confidence(logits, y):
    probs = torch.nn.Softmax(dim=1)(logits)

    # Attempt 1 (not enough precision)
    # p = probs[torch.arange(len(y)),y]

    # Attempt 2 (too slow)
    # p = []
    # for prob, label in zip(probs, y):
    #     p.append(torch.cat([prob[:label], prob[label+1:]]).sum())
    # p = torch.tensor(p)

    # Attempt 3
    mask = torch.ones_like(probs)
    mask[[torch.arange(len(y)), y]] = 0
    # p = torch.clamp(probs[mask.bool()].view((-1,9)).sum(dim=1), max=1)
    p = probs[mask.bool()].view((-1,9)).sum(dim=1)
    
    # return torch.clamp(torch.log((1-p) / p), min=-50, max=50).cpu()
    return (torch.log(probs[torch.arange(len(y)),y]) - torch.log(p)).cpu()


# def get_params(trained_models, subsets, X, Y, logits=None):
#     if logits is None:
#         with torch.no_grad():
#             logits = [model(X) for model in tqdm(trained_models)]
#     losses = {ind:confidence(logit, Y) for ind, logit in tqdm(logits.items())}
    
#     in_losses = [[] for _ in range(len(X))]
#     out_losses = [[] for _ in range(len(X))]
    
#     for ind,subset in tqdm(subsets.items(), total=len(subsets)):
#         subset_set = set(subset)
#         for j in range(len(X)):
#             if j in subset_set:
#                 in_losses[j].append(losses[ind][j])
#             else:
#                 out_losses[j].append(losses[ind][j])

#     in_mu = [np.mean(g) for g in tqdm(in_losses)]
#     in_std = [np.std(g) for g in tqdm(in_losses)]

#     out_mu = [np.mean(g) for g in tqdm(out_losses)]
#     out_std = [np.std(g) for g in tqdm(out_losses)]

#     return in_mu, in_std, out_mu, out_std

def get_params(trained_models, subsets, X, Y, logits=None, flipped_logits=None):
    if logits is None:
        with torch.no_grad():
            logits = [model(X) for model in tqdm(trained_models)]
            flipped_logits = [model(torch.flip(X, [3])) for model in tqdm(trained_models)]
    losses = {ind:confidence(logit, Y) for ind, logit in tqdm(logits.items())}
    flipped_losses = {ind:confidence(logit, Y) for ind, logit in tqdm(flipped_logits.items())}
    
    in_losses = [[] for _ in range(len(X))]
    out_losses = [[] for _ in range(len(X))]
    
    for ind,subset in tqdm(subsets.items(), total=len(subsets)):
        subset_set = set(subset)
        for j in range(len(X)):
            if j in subset_set:
                in_losses[j].append((losses[ind][j], flipped_losses[ind][j]))
            else:
                out_losses[j].append((losses[ind][j], flipped_losses[ind][j]))

    in_mu = [np.mean(g, axis=0) for g in tqdm(in_losses)]
    in_cov = [np.cov(np.transpose(g)) for g in tqdm(in_losses)]

    out_mu = [np.mean(g, axis=0) for g in tqdm(out_losses)]
    out_cov = [np.cov(np.transpose(g)) for g in tqdm(out_losses)]

    return in_mu, in_cov, out_mu, out_cov

# def lira(test_model, test_model_subset, in_mu, in_std, out_mu, out_std, trainloader, test_logits=None):
#     # print('Compute test logit gap')
#     if test_logits is None:
#         test_logits = []
#         Y = []
#         for x, y in trainloader:
#             with torch.no_grad():
#                 x = x.cuda()
#                 y = y.cuda()
#                 test_logits.append(test_model(x))
#                 Y.append(y)
#         test_logits = torch.cat(test_logits)
#         Y = torch.cat(Y)
#     else:
#         Y = []
#         for x, y in trainloader:
#             y = y.cuda()
#             Y.append(y)
#         Y = torch.cat(Y)
        
#     test_logit_gap = confidence(test_logits, Y)
#     # print('Get ground truth')
#     test_point_in = [ind in test_model_subset for ind in range(len(Y))]

#     # print('Compute Likelihoods')
#     in_likelihood = [norm.pdf(test_logit_gap[i], loc=in_mu[i], scale=in_std[i]) for i in range(len(Y))]
#     out_likelihood = [norm.pdf(test_logit_gap[i], loc=out_mu[i], scale=out_std[i]) for i in range(len(Y))]

#     test_point_in = np.array(test_point_in)
#     pred_in = np.array([i > o for i,o in zip(in_likelihood, out_likelihood)])
#     scores = np.array([i/o for i,o in zip(in_likelihood, out_likelihood)])

#     return test_point_in, pred_in, scores

def lira(test_model, test_model_subset, in_mu, in_std, out_mu, out_std, trainloader):
    # print('Compute test logit gap')
    test_logits = []
    flipped_test_logits = []
    Y = []
    for x, y in tqdm(trainloader):
        with torch.no_grad():
            x = x.cuda()
            y = y.cuda()
            test_logits.append(test_model(x))
            flipped_test_logits.append(test_model(torch.flip(x, [3])))
            Y.append(y)
    test_logits = torch.cat(test_logits)
    flipped_test_logits = torch.cat(flipped_test_logits)
    Y = torch.cat(Y)

    test_logit_gap = confidence(test_logits, Y)
    flipped_logit_gap = confidence(flipped_test_logits, Y)
    test_point_in = [ind in test_model_subset for ind in range(len(Y))]

    in_likelihood = [multivariate_normal.pdf([test_logit_gap[i],flipped_logit_gap[i]], mean=in_mu[i], cov=in_std[i]) for i in range(len(Y))]
    out_likelihood = [multivariate_normal.pdf([test_logit_gap[i],flipped_logit_gap[i]], mean=out_mu[i], cov=out_std[i]) for i in range(len(Y))]

    test_point_in = np.array(test_point_in)
    pred_in = np.array([i > o for i,o in zip(in_likelihood, out_likelihood)])
    scores = np.array([i/o for i,o in zip(in_likelihood, out_likelihood)])

    return test_point_in, pred_in, scores
