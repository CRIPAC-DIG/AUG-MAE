import torch
import torch.nn.functional as F


def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    loss = loss.mean()
    return loss


def neg_sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    loss = loss.mean()
    return loss


def sig_loss(x, y):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (x * y).sum(1)
    loss = torch.sigmoid(-loss)
    loss = loss.mean()
    return loss

def uniformity_loss(features,t,max_size=30000,batch=10000):
    # calculate loss
    n = features.size(0)
    features = torch.nn.functional.normalize(features)
    if n < max_size:
        loss = torch.log(torch.exp(2.*t*((features@features.T)-1.)).mean())
    else:
        total_loss = 0.
        permutation = torch.randperm(n)
        features = features[permutation]
        for i in range(0, n, batch):
            batch_features = features[i:i + batch]
            batch_loss = torch.log(torch.exp(2.*t*((batch_features@batch_features.T)-1.)).mean())
            total_loss += batch_loss
        loss = total_loss / (n // batch)
    return loss
