import argparse
import logging
import os
import sys

import numpy as np
import torch
import torchvision
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import madgrad

from PIL import Image

from eval import eval_net
from unet import UNet

import pytorch_lightning as pl
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.models.self_supervised.simclr.simclr_module import SimCLR

from pytorch_lightning.loggers import TensorBoardLogger

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split
from pl_bolts.models.self_supervised.simclr.transforms import (SimCLREvalDataTransform, SimCLRTrainDataTransform)

import lightly.models as models
import lightly.loss as loss
import lightly.data as data
import lightly.embedding as embedding

class MEcho(torchvision.datasets.VisionDataset):
  
  def __init__(self, root=None, notransform=False, twoclass=False, split='train'):
      self.root = root
      self.notransform = notransform
      self.twoclass = twoclass
      self.split = split

      if root is None:
            self.root = '/home/student/Echo/dynamic/echonet/a4c-video-dir'
      
      self.images = []      
      for image in os.listdir(os.path.join(self.root, "frame_images")): 
         if self.twoclass:
            if image.split("_")[-1].split(".")[0] == '0' or  image.split("_")[-1].split(".")[0] == '4':
               self.images.append(image) 
         else:
            self.images.append(image)
       
  def __getitem__(self, index):
      image = self.images[index]
      label = int(image.split("_")[-1].split(".")[0])

      if self.twoclass:
         label = 0 if label == 0 else 1
      
      img = Image.open(os.path.join(self.root, "frame_images", image)).convert("RGB")
      
      if self.notransform:
          transform = torchvision.transforms.ToTensor()
      else:
          transform = SimCLRTrainDataTransform(112)
      return transform(img), label
       
  def __len__(self):
      return len(self.images)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = UNet(n_channels=3, n_classes=1, bilinear=True)

#encoder = torchvision.models._utils.IntermediateLayerGetter(net, {"down4":"down4"})
encoder = list(net.children())[:-5]
encoder = nn.Sequential(*encoder)


dataset = MEcho(root=None)
dataloader = torch.utils.data.DataLoader(dataset, batch_size = 32, num_workers=16, shuffle=True, pin_memory=(device.type == "cuda"))

simclr = SimCLR(1, num_samples = len(dataset), batch_size = 32, dataset = 'cifar10', temperature=0.05)

simclr.encoder = encoder

checkpoint_cb = pl.callbacks.ModelCheckpoint(monitor='train_loss',
                                                    dirpath='./ssl_checkpoints/',
                                                    filename='simclr-{epoch:02d}-{train_loss:.2f}',
                                                    mode='min')
lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

logger = TensorBoardLogger('tb_logs', name='simclr')
trainer = pl.Trainer(gpus=1, max_epochs = 50, callbacks = [checkpoint_cb, lr_monitor], auto_lr_find=True, logger=logger)
trainer.fit(simclr, dataloader)
torch.save(simclr.state_dict(), 'simclrunet.pth')
