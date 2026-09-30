import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW, SGD, Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

import copy
from tqdm import tqdm

BATCH_SIZE = 128
LR = 0.1
WEIGHT_DECAY=0.0001
MOMENTUM=0.9
NUM_WORKERS=2
NUM_EPOCHS=50

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_cifar10_datasets(root_path):
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(10),
        # transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), 
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

def finetune(m, retain_set, forget_set, num_epochs=20):
    # finetune 
    # - Fine-tune m on the retain set, with the forget set excluded
    # - Warnecke, Alexander, et al. "Machine unlearning of features and labels." arXiv preprint arXiv:2108.11577 (2021).
    train, _ = load_cifar10_datasets('./data/cifar10')
    train_dataset = torch.utils.data.Subset(train, retain_set)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    m.train()
    criterion = nn.CrossEntropyLoss()
    opt = SGD(m.parameters(), lr=0.01, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    sched = optim.lr_scheduler.MultiStepLR(opt, milestones=[num_epochs * 0.5, num_epochs * 0.75], gamma=0.1)

    for epoch in (pbar := tqdm(range(num_epochs))):
        for i, (x,y) in enumerate(train_loader):
            x,y = x.to(device), y.to(device)
            for param in m.parameters():
                param.grad = None

            preds = m(x)
            loss = criterion(preds, y)
            loss.backward()
            opt.step()
        sched.step()
    m.eval()

    return m

def eu_k(m, retain_set, forget_set, num_epochs=50):
    # Exact-Unlearning the last k layers (EU-k)
    # - Retrain the last k layers from scratch on just the retain set, freeze earlier layers
    # - Here, we're fixing k to 1
    # - Goel, Shashwat, et al. "Towards adversarial evaluations for inexact machine unlearning." arXiv preprint arXiv:2201.06640 (2022).
    for param in m.parameters():
        param.requires_grad = False
    m.fc = torch.nn.Linear(512, 10).to(device)
    
    train, _ = load_cifar10_datasets('./data/cifar10')
    train_dataset = torch.utils.data.Subset(train, retain_set)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    m.train()
    criterion = nn.CrossEntropyLoss()
    opt = SGD(m.fc.parameters(), lr=0.01, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    for epoch in (pbar := tqdm(range(num_epochs))):
        for i, (x,y) in enumerate(train_loader):
            x,y = x.to(device), y.to(device)
            for param in m.parameters():
                param.grad = None

            preds = m(x)
            loss = criterion(preds, y)
            loss.backward()
            opt.step()
    m.eval()

    return m

def cf_k(m, retain_set, forget_set, num_epochs=50):
    # Catastrophically forgetting the last k layers (CF-k)
    # - Fine-tune the last k layers on just the retain set, freeze earlier layers
    # - Here, we're fixing k to 1
    # - Goel, Shashwat, et al. "Towards adversarial evaluations for inexact machine unlearning." arXiv preprint arXiv:2201.06640 (2022).
    for param in m.parameters():
        param.requires_grad = False
    for param in m.fc.parameters():
        param.requires_grad = True
    
    train, _ = load_cifar10_datasets('./data/cifar10')
    train_dataset = torch.utils.data.Subset(train, retain_set)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    m.train()
    criterion = nn.CrossEntropyLoss()
    opt = SGD(m.fc.parameters(), lr=0.01, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    for epoch in (pbar := tqdm(range(num_epochs))):
        for i, (x,y) in enumerate(train_loader):
            x,y = x.to(device), y.to(device)
            for param in m.parameters():
                param.grad = None

            preds = m(x)
            loss = criterion(preds, y)
            loss.backward()
            opt.step()
    m.eval()

    return m


def do_max_epoch(student, teacher, forget_loader, opt):
    # SCRUB helper
    # - Maximize KL divergence between output distributions on the forget set
    for x, y in forget_loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        pred_student = nn.Softmax()(student(x))
        with torch.no_grad():
            pred_teacher = nn.Softmax()(teacher(x))
        l = -nn.KLDivLoss()(pred_student, pred_teacher)
        l.backward()
        opt.step()

def do_min_epoch(student, teacher, retain_loader, opt, alpha=5e-4):
    # SCRUB helper
    # - Minimize KL divergence between output distributions on the retain set
    # - Minimize cross-entropy loss on the retain set
    criterion = nn.CrossEntropyLoss()
    for x, y in retain_loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        pred_student = nn.Softmax()(student(x))
        with torch.no_grad():
            pred_teacher = nn.Softmax()(teacher(x))
        l = nn.KLDivLoss()(pred_student, pred_teacher) + alpha * criterion(pred_student, y)
        l.backward()
        opt.step()

def scrub(m, retain_set, forget_set, max_steps=24, steps=16):
    # SCalable Remembering and Unlearning unBound (SCRUB)
    # - Teacher-student method: 
    #   - Maximize KL divergence between output distributions on the forget set
    #   - Minimize KL divergence between output distributions on the retain set
    #   - Minimize cross-entropy loss on the retain set
    # - Kurmanji, Meghdad, et al. "Towards unbounded machine unlearning." Advances in neural information processing systems 36 (2023): 1957-1987.
    train, _ = load_cifar10_datasets('./data/cifar10')
    retain_dataset = torch.utils.data.Subset(train, retain_set)
    forget_dataset = torch.utils.data.Subset(copy.deepcopy(train), forget_set)
    retain_loader = DataLoader(retain_dataset, batch_size=64, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    forget_loader = DataLoader(forget_dataset, batch_size=16, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    m.train()
    student = m
    teacher = copy.deepcopy(m)
    opt = AdamW(student.parameters(), 5e-5, weight_decay=0.1)
    sched = optim.lr_scheduler.MultiStepLR(opt, milestones=[max_steps * 0.5, max_steps * 0.75], gamma=0.1)
    for step in tqdm(range(steps)):
        if step < max_steps:
            do_max_epoch(student, teacher, forget_loader, opt)
        do_min_epoch(student, teacher, retain_loader, opt)
        sched.step()
    m.eval()

    return m


def l1_regularization(model):
    params_vec = []
    for param in model.parameters():
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)

def sparse_unlearn(m, retain_set, forget_set, num_epochs=20, alpha=5e-4):
    # $\ell_1$-Sparse Unlearning
    # - Fine-tune on just the retain set using a regularization term which encourages $\ell_1$ sparsity of the model weights
    # - Jia, Jinghan, et al. "Model sparsity can simplify machine unlearning." Advances in Neural Information Processing Systems 36 (2023): 51584-51605.
    train, _ = load_cifar10_datasets('./data/cifar10')
    retain_dataset = torch.utils.data.Subset(train, retain_set)
    retain_loader = DataLoader(retain_dataset, batch_size=64, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    m.train()
    criterion = nn.CrossEntropyLoss()
    opt = SGD(m.parameters(), lr=0.001, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    sched = optim.lr_scheduler.MultiStepLR(opt, milestones=[num_epochs * 0.5, num_epochs * 0.75], gamma=0.1)

    for epoch in (pbar := tqdm(range(num_epochs))):
        for i, (x,y) in enumerate(retain_loader):
            x,y = x.to(device), y.to(device)
            opt.zero_grad()

            current_alpha = alpha * (1 - (epoch / num_epochs))
            preds = m(x)
            loss = criterion(preds, y) + current_alpha * l1_regularization(m)
            loss.backward()
            opt.step()
        sched.step()
    m.eval()

    return m


def relabel(m, retain_set, forget_set, num_epochs=10):
    # Random Labeling
    # - Relabel the forget set with randomly sampled labels
    # - Fine-tune the model on the retain + relabeled forget dataset
    # - Golatkar, Aditya, Alessandro Achille, and Stefano Soatto. "Eternal sunshine of the spotless net: Selective forgetting in deep networks." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020.
    train, _ = load_cifar10_datasets('./data/cifar10')
    retain_dataset = torch.utils.data.Subset(train, retain_set)
    forget_dataset = torch.utils.data.Subset(copy.deepcopy(train), forget_set)
    forget_dataset.dataset.targets = np.random.randint(0, 10, len(forget_dataset.dataset.targets))
    train_dataset = torch.utils.data.ConcatDataset([forget_dataset,retain_dataset])
    # train_dataset = retain_dataset
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    m.train()
    criterion = nn.CrossEntropyLoss()
    opt = SGD(m.parameters(), lr=0.01, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    for epoch in (pbar := tqdm(range(num_epochs))):
        for i, (x,y) in enumerate(train_loader):
            x,y = x.to(device), y.to(device)
            for param in m.parameters():
                param.grad = None

            preds = m(x)
            loss = criterion(preds, y)
            loss.backward()
            opt.step()
    m.eval()

    return m


def salun(m, retain_set, forget_set, num_epochs=20, gamma=0):
    # Saliency Unlearning
    # - Like random labeling, but only update weights with gradients larger than gamma threshold
    # - Fan, Chongyu, et al. "Salun: Empowering machine unlearning via gradient-based weight saliency in both image classification and generation." arXiv preprint arXiv:2310.12508 (2023).
    train, _ = load_cifar10_datasets('./data/cifar10')
    retain_dataset = torch.utils.data.Subset(train, retain_set)
    forget_dataset = torch.utils.data.Subset(copy.deepcopy(train), forget_set)
    forget_dataset.dataset.targets = np.random.randint(0, 10, len(forget_dataset.dataset.targets))
    train_dataset = torch.utils.data.ConcatDataset([forget_dataset,retain_dataset])
    # train_dataset = retain_dataset
    forget_loader = DataLoader(forget_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    m.train()
    criterion = nn.CrossEntropyLoss()
    mask_criterion = nn.CrossEntropyLoss(reduction='sum')
    opt = SGD(m.parameters(), lr=0.01, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)


    grads = {}
    for name, param in m.named_parameters():
        grads[name] = torch.zeros(param.shape).to(device)
    for x, y in forget_loader:
        x,y = x.to(device), y.to(device)
        preds = m(x)
        loss = mask_criterion(preds, y)
        loss.backward()

        for name, param in m.named_parameters():
            if param.grad is not None:
                grads[name] += param.grad

    mask = {}
    for name in grads:
        # print(grads[name].median())
        grads[name] = grads[name] / len(forget_dataset)
        mask[name] = (grads[name] > gamma).float()
    
    for epoch in (pbar := tqdm(range(num_epochs))):
        for i, (x,y) in enumerate(train_loader):
            x,y = x.to(device), y.to(device)
            opt.zero_grad()

            preds = m(x)
            loss = criterion(preds, y)
            loss.backward()
            for name, param in m.named_parameters():
                if param.grad is not None:
                    param.grad *= mask[name]
            opt.step()
    m.eval()

    return m