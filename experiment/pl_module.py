import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.models.segmentation import fcn_resnet50


# Get pretrained FCN architecture.
def get_fcn(pt=True, nc=21):
    # If nc != 21, last layers will used subset of pretrained weights
    assert 0 < nc <= 21, f'Num_classes must be between 1 and 21, got {nc}'

    fcn = fcn_resnet50(pt)
    if nc != 21:
        out_conv = nn.Conv2d(512, nc, kernel_size=(1, 1), stride=(1, 1))
        aux_conv = nn.Conv2d(256, nc, kernel_size=(1, 1), stride=(1, 1))

        out_conv.weight = nn.Parameter(fcn.classifier[4].weight[:nc])
        out_conv.bias = nn.Parameter(fcn.classifier[4].bias[:nc])
        aux_conv.weight = nn.Parameter(fcn.aux_classifier[4].weight[:nc])
        aux_conv.bias = nn.Parameter(fcn.aux_classifier[4].bias[:nc])

        fcn.classifier[4] = out_conv
        fcn.aux_classifier[4] = aux_conv

    return fcn


# Loss function accounting for both FCN outputs (aux at 1/2 resolution of out)
def loss_func(pred, targ):
    return F.binary_cross_entropy_with_logits(pred['out'], targ) + \
           F.binary_cross_entropy_with_logits(pred['aux'], targ)


class FCNSegmentation(pl.LightningModule):

    def __init__(self,
                 trn_ds, val_ds, tst_ds=None,
                 pretrained=True, num_classes=21,
                 sampler=None,
                 metrics=[],
                 bs=8, nw=0):
        super().__init__()
        self.datasets = {
            'train': {'ds': trn_ds, 'samp': sampler, 'drop': True},
            'valid': {'ds': val_ds, 'samp': None, 'drop': False},
            'test': {'ds': tst_ds, 'samp': None, 'drop': False},
        }
        self.metrics = nn.ModuleList(metrics)
        self.bs = bs
        self.nw = nw
        self.fcn = get_fcn(pt=pretrained, nc=num_classes)

    def train_dataloader(self):
        return self.get_dl('train')

    def val_dataloader(self):
        return self.get_dl('valid')

    def test_dataloader(self):
        return self.get_dl('test')

    def get_dl(self, phase):
        return DataLoader(self.datasets[phase]['ds'],
                          batch_size=self.bs, num_workers=self.nw,
                          drop_last=self.datasets[phase]['drop'],
                          sampler=self.datasets[phase]['samp'])

    def training_step(self, batch, batch_idx):
        x, y = batch
        p = self.fcn(x)
        loss = loss_func(p, y)
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        p = self.fcn(x)
        loss = loss_func(p, y)
        self.log('val_loss', loss)

        for metric in self.metrics:
            self.log(f'val_{type(metric).__name__}',
                     metric(torch.sigmoid(p['out']), y.to(torch.int8)))

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        p = self.fcn(x)
        loss = loss_func(p, y)
        self.log('test_loss', loss)

        for metric in self.metrics:
            self.log(f'test_{type(metric).__name__}',
                     metric(torch.sigmoid(p['out']), y.to(torch.int8)))

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-2)
        return optimizer
