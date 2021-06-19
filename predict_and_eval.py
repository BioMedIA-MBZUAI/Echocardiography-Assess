import torch
import logging

from utils.dataset import BasicDataset
from torch.utils.data import DataLoader

from eval import eval_net
from unet import UNet

if __name__ == '__main__':

  batch_size = 20
  img_scale = 1
  val_img = "./data/imgs/test/"
  val_mask = "./data/masks/test/"

  val = BasicDataset(val_img, val_mask, img_scale)
  n_val = len(val)
  val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)


  logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  logging.info(f'Using device {device}')

  net = UNet(n_channels=3, n_classes=1, bilinear=True)

  
  net.load_state_dict(
    torch.load("MODEL.pth", map_location=device) 
  )

  logging.info(f'Model loaded')

  net.to(device=device)
  
  val_score = eval_net(net, val_loader, device)
  print(val_score)


  logging.info('Validation Dice Coeff: {}'.format(val_score))
