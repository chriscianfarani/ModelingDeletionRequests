import torch
import torch.nn as nn

from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np
import sklearn.metrics as metrics

from argparse import ArgumentParser
from tqdm import tqdm
from pathlib import Path
import tarfile
import io

import lira_attack
from resnet import resnet18
from unlearn import get_unlearn_method


BATCH_SIZE = 128
LR = 0.1
WEIGHT_DECAY=0.0001
MOMENTUM=0.9
NUM_WORKERS=2
NUM_EPOCHS=50

IN_MU = None
OUT_MU = None
IN_STD = None
OUT_STD = None

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_cifar10_datasets(root_path):
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), 
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    train = datasets.CIFAR10(root_path, train=True, transform=transform, download=True)
    test = datasets.CIFAR10(root_path, train=False, transform=test_transform, download=True)

    return train, test

def train_model(m, train_subset=range(50000), num_epochs=NUM_EPOCHS, data_dir='./data/cifar10'):
    train, _ = load_cifar10_datasets(data_dir)
    trainset = torch.utils.data.Subset(train, train_subset)
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    
    criterion = nn.CrossEntropyLoss()
    opt = SGD(m.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    for epoch in (pbar := tqdm(range(num_epochs))):
        for i, (x,y) in enumerate(train_loader):
            x,y = x.to(device), y.to(device)
            for param in m.parameters():
                param.grad = None

            preds = m(x)
            loss = criterion(preds, y)
            loss.backward()
            opt.step()

def get_cifar10_images(data_dir='./data/cifar10'):
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    train = datasets.CIFAR10(data_dir, train=True, transform=transform, download=True)
    test = datasets.CIFAR10(data_dir, train=False, transform=transform, download=True)

    train_loader = DataLoader(train, batch_size=50000, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test, batch_size=10000, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    for i,l in train_loader:
        X = i.to(device)
        Y = l.to(device)

    for i,l in test_loader:
        test_X = i.to(device)
        test_Y = l.to(device)

    return X, Y, test_X, test_Y


def load_cifar10(root_path, shuffle=True, batch_size=BATCH_SIZE, half_train=True):
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    train = datasets.CIFAR10(root_path, train=True, transform=transform, download=True)
    test = datasets.CIFAR10(root_path, train=False, transform=transform, download=True)
    if half_train:
        subset = np.random.choice(range(50000), size=25000, replace=False)
        train = torch.utils.data.Subset(train, subset)
    else:
        subset = list(range(50000))

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=shuffle, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    return train_loader, test_loader, subset

def get_model_ind(save_dir):
    inds = [int(x.stem.split('_')[-2]) for x in save_dir.glob('*.pt')]
    if len(inds) == 0:
        return 1
    return max(inds) + 1

def get_logits(m, loader):
    logits = []
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        with torch.no_grad():
            logits.append(m(X))
    return torch.cat(logits)

def fill_lira_params(X, Y):
    logits = torch.load('./data/logits_50epochs_updated.pt', map_location=device)
    flipped_logits = torch.load('./data/flipped_logits_50epochs_updated.pt', map_location=device)
    subsets = {}
    tar: tarfile.TarFile = tarfile.open('./data/subsets.tar.gz', 'r:gz')
    for ind in logits:
        f = tar.extractfile(f'resnet18_50epochs_{ind}_updated.npy')
        if f is not None:
            subset = np.load(io.BytesIO(f.read()))
        else:
            raise FileNotFoundError('Missing subsets data')
        subsets[ind] = subset

    global IN_MU, IN_STD, OUT_MU, OUT_STD
    IN_MU, IN_STD, OUT_MU, OUT_STD = lira_attack.get_params([], subsets, X, Y, logits=logits, flipped_logits=flipped_logits)

def pairwise_cosine_sim(a, b, eps=1e-8):
    a_norm = a / torch.clamp(a.norm(dim=1)[:,None], eps)
    b_norm = b / torch.clamp(b.norm(dim=1)[:,None], eps)
    return a_norm @ b_norm.T

def knn_sv(train_logits, train_labels, test_logits, test_labels, k=10):
    N, N_test = len(train_labels), len(test_labels)
    vals = torch.zeros((N_test, N)).to(device)
    sims_matrix = pairwise_cosine_sim(train_logits, test_logits)
    sorted_matrix = torch.sort(sims_matrix.T, descending=True).indices.to(device)
    vals[torch.arange(len(vals)),sorted_matrix[:,N-1].cpu()] = (test_labels == train_labels[sorted_matrix[:,N-1]])/N

    for i in range(N-2,-1,-1):
        prev_vals = vals[torch.arange(len(vals)),sorted_matrix[:,i+1]]
        d1 = (test_labels == train_labels[sorted_matrix[:,i]]).float()
        d2 = (test_labels == train_labels[sorted_matrix[:,i+1]]).float()
        diff = (d1 - d2) / k
        vals[torch.arange(len(vals)), sorted_matrix[:,i].cpu()] = prev_vals + (diff * min(k,(i+1)) / (i+1))

    return torch.sum(vals, dim=0) / N_test

def masked_knn_sv(train_logits, train_labels, test_logits, test_labels, subset, k=10):
    N, N_test = len(train_labels), len(test_labels)
    vals = torch.zeros((N_test, N)).to(device)
    sims_matrix = pairwise_cosine_sim(train_logits, test_logits)
    sorted_matrix = torch.sort(sims_matrix.T, descending=True).indices.to(device)

    mask = ~torch.isin(sorted_matrix, test_elements=torch.tensor(subset).to(device))
    next_nonzero_indices = torch.zeros(mask.shape)
    for i in tqdm(range(len(mask))):
        nonzero_indices = mask[i].nonzero()[:,0]
        positions = torch.searchsorted(nonzero_indices, torch.arange(len(mask[0])).to(device))
        nonzero_indices_padded =  torch.cat([nonzero_indices, torch.tensor([-1]).to(device)])
        next_nonzero_indices[i,:] = nonzero_indices_padded[positions]
    next_nonzero_indices = next_nonzero_indices.type(torch.int)

    cum_sums = torch.cumsum(mask, dim=1) + 1

    vals[torch.arange(len(vals)),sorted_matrix[:,N-1].cpu()] = (test_labels == train_labels[sorted_matrix[:,N-1]])/cum_sums[:,N-1]

    for i in tqdm(range(N-2,-1,-1)):
        prev_vals = vals[torch.arange(len(vals)), sorted_matrix[torch.arange(len(sorted_matrix)),next_nonzero_indices[:,i+1]]]
        
        d1 = (test_labels == train_labels[sorted_matrix[:,i]]).float()
        d2 = (test_labels == train_labels[sorted_matrix[torch.arange(len(sorted_matrix)),next_nonzero_indices[:,i+1]]]).float()
        d2 = d2 * (next_nonzero_indices[:,i+1] != -1).float().to(device) # TODO: is this right?
        diff = (d1 - d2) / k

        # TODO: Handle i+1
        vals[torch.arange(len(vals)), sorted_matrix[:,i].cpu()] = prev_vals + (diff * torch.clamp(cum_sums[:,i], max=k) / cum_sums[:,i])

    print(vals.shape)
    return torch.sum(vals, dim=0) / N_test

def get_unlearn_inds(args, num_unlearned, train_inds, loss, m, X, Y, test_X, test_Y, trainloader, testloader, full_trainloader=None, full_testloader=None):
    if args.unlearn_type == 'random':
        # Random requester
        # - Randomly permutes training data, selects first num_unlearned to unlearn
        ind_list = list(train_inds)
        perm = torch.randperm(len(ind_list))
        unlearn_inds = [ind_list[i] for i in perm][:num_unlearned]
    if args.unlearn_type == 'loss':
        # Minimum loss requester
        # - Selects num_unlearned training points with lowest loss to unlearn
        ordering = torch.argsort(loss)
        unlearn_inds = [x.item() for x in ordering if x.item() in train_inds][:num_unlearned]
    if args.unlearn_type == 'lira':
        # LiRA requester
        # - Selects num_unlearned training points with highest LiRA scores to unlearn
        y, pred, score = lira_attack.lira(m, train_inds, IN_MU, IN_STD, OUT_MU, OUT_STD, full_trainloader)
        in_point_scores = {ind:score[ind] for ind in train_inds}
        unlearn_inds = sorted(in_point_scores, key=in_point_scores.get)[-num_unlearned:][::-1]
    if args.unlearn_type == 'shapley' or args.unlearn_type == 'min_shapley':
        # Shapley requester
        # - Computes knn-shapley value for each training point, removes num_unlearned highest/lowest
        train_ind_list = list(train_inds)
        index_mapping = {i:ind for i,ind in enumerate(train_ind_list)}
        new_Y = Y[train_ind_list]
        with torch.no_grad():
            logits = get_logits(m, full_trainloader)
            logits = logits[train_ind_list]
            test_logits = get_logits(m, testloader)
        scores = knn_sv(logits, new_Y, test_logits, test_Y)
        ordering = torch.argsort(scores, descending=True)
        if args.unlearn_type == 'shapley':
            unlearn_inds = [index_mapping[x.item()] for x in ordering][:num_unlearned]
        else:
            unlearn_inds = [index_mapping[x.item()] for x in ordering][-num_unlearned:][::-1]
    if args.unlearn_type == 'masked_shapley':
        # Shapley requester
        # - Computes knn-shapley value for each training point, removes num_unlearned highest/lowest
        train_ind_list = list(train_inds)
        index_mapping = {i:ind for i,ind in enumerate(train_ind_list)}
        new_Y = Y[train_ind_list]
        with torch.no_grad():
            logits = get_logits(m, full_trainloader)
            test_logits = get_logits(m, full_testloader)
        print('Computing shapley vals')
        scores = masked_knn_sv(logits, Y, test_logits, test_Y, train_ind_list)
        print('Sorting indices')
        ordering = torch.argsort(scores, descending=True)
        ordering = [x for x in ordering if x in train_ind_list]
        unlearn_inds = ordering[:num_unlearned]  

    return unlearn_inds

def retrain(m, retain_set, forget_set, num_epochs=NUM_EPOCHS):
    # Initialize new model, train from scratch
    m = resnet18(num_classes=10).to(device)
    train_model(m, train_subset = list(retain_set), num_epochs=num_epochs)
    return m

def compute_lira(m, train_inds, trainloader, mask=[]):
    y, pred, score = lira_attack.lira(m, train_inds, IN_MU, IN_STD, OUT_MU, OUT_STD, trainloader)
    mask_set = set(mask)
    valid_inds = [x for x in range(len(score)) if x not in mask_set]
    auc = metrics.roc_auc_score(y[valid_inds], score[valid_inds] - 1)
    fpr, tpr, thresholds = metrics.roc_curve(y, score)
    fpr_ind = np.argmin(np.abs(fpr - 0.001))
    return score, auc, tpr[fpr_ind]

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('ind', type=int)
    parser.add_argument('--unlearn_type', choices=['random', 'loss', 'lira', 'shapley', 'min_shapley', 'masked_shapley'], default='loss')
    parser.add_argument('--num_unlearned', type=int, default=1000, help='Number of points to unlearn on each iteration')
    parser.add_argument('--num_iters', type=int, default=10)
    parser.add_argument('--non_adaptive', action='store_true')
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--unlearn_method', type=str, choices=['retrain', 'scrub', 'relabel', 'saliency', 'sparse_unlearn'], default='retrain')
    parser.add_argument('--exp_dir', type=str, help='Name of directory to save models to', default=None)
    parser.add_argument('--no_model_save', action='store_true', help='Do not save trained models')
    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()
    print(args)
    NUM_EPOCHS=args.num_epochs

    # Get unlearning function
    if args.unlearn_method == 'retrain':
        unlearn_method = retrain
    else:
        unlearn_method = get_unlearn_method(args.unlearn_method)

    # trainloader - cifar10 dataset including only training indices in subset
    # full_trainloader - cifar10 dataset with all training indices
    trainloader, testloader, subset = load_cifar10('./data/cifar10', shuffle=False)
    full_trainloader, full_testloader, _ = load_cifar10('./data/cifar10', shuffle=False, half_train=False)
    X, Y, test_X, test_Y = get_cifar10_images()

    # Compute LiRA parameters for training samples
    if args.unlearn_type == 'lira':
        fill_lira_params(X, Y)

    # Create directories to save models parameters and training subsets/evaluations
    exp_dir = args.exp_dir
    if exp_dir is None:
        exp_dir = 'experiment1'
    model_dir = Path(f'./results/{exp_dir}/{args.unlearn_type}/models')
    model_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(f'./results/{exp_dir}/{args.unlearn_type}/data')
    data_dir.mkdir(parents=True, exist_ok=True)

    # Initialize retain and unlearn sets
    train_inds = set(subset)
    unlearn_inds = list(set(range(50000)) - train_inds)
    model_ind = args.ind
    save_data = {}

    init_m = None
    m = None
    for i in range(args.num_iters):
        num_unlearned = args.num_unlearned

        print(f'Iteration {i}')
        if i == 0:
            # Train initial model
            m = resnet18(num_classes=10).to(device)
            train_model(m, train_subset = list(train_inds), num_epochs=args.num_epochs)
        else:
            # Unlearning step
            # - train_inds = retain set, unlearn_inds = forget set
            m = unlearn_method(m, list(train_inds), list(unlearn_inds))

        # Compute test accuracy
        preds = get_logits(m, full_trainloader)
        test_preds = get_logits(m, testloader)
        accuracy = ((torch.argmax(test_preds, axis=1) == test_Y).sum() / len(test_Y)).item()
        print(f'Accuracy: {accuracy:0.3f}')
        print()

        if init_m is None:
            init_m = m
        loss = torch.nn.CrossEntropyLoss(reduce=False)(preds, Y)

        model_data = {
            'acc': accuracy,
        }
        save_data[i] = model_data

        if args.non_adaptive:
            # Non-adaptive setting: reset train_inds to original training set, select num_unlearned points to remove from initial model
            if i == 0:
                # fill the unlearn_inds at the beginning
                train_inds = set(subset)
                full_trainloader2, full_testloader2, _ = load_cifar10('./data/cifar10', shuffle=False, half_train=False)
                full_unlearn_inds = get_unlearn_inds(args, len(train_inds), train_inds, loss, init_m, X, Y, test_X, test_Y, trainloader, testloader, full_trainloader2, full_testloader2)
            unlearn_inds = full_unlearn_inds[num_unlearned*i:num_unlearned*(i+1)]
        else:
            # Adaptive setting: get num_unlearned points to remove from current retain set and model
            full_trainloader2, full_testloader2, _ = load_cifar10('./data/cifar10', shuffle=False, half_train=False)
            unlearn_inds = get_unlearn_inds(args, num_unlearned, train_inds, loss, m, X, Y, test_X, test_Y, full_trainloader, testloader, full_trainloader2, full_testloader2)

        # Update retain set
        train_inds = train_inds - {x for x in unlearn_inds}

        non_adaptive_str = ''
        if args.non_adaptive:
            non_adaptive_str = '_nonadaptive'
        
        suffix = ''
        if len(args.suffix) > 0:
            suffix = f'_{args.suffix}'
        
        if not args.no_model_save:
            torch.save(m, model_dir/f'resnet18_{args.num_unlearned}unlearn_{args.num_epochs}epochs_{args.unlearn_method}_{model_ind}_{i}{non_adaptive_str}{suffix}.pt')
        
    outfile = data_dir/f'resnet18_{args.num_unlearned}unlearn_{args.num_epochs}epochs_{args.unlearn_method}_{model_ind}{non_adaptive_str}{suffix}.pt'
    with open(outfile, 'w') as f:
        json.dump(save_data, f)
