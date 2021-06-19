import torch
import torch.nn.functional as F
from tqdm import tqdm

from dice_loss import dice_coeff
from torchvision.utils import save_image


def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for i,batch in enumerate(loader):
            imgs, true_masks, names = batch['image'], batch['mask'], batch['name']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)

            if net.n_classes > 1:
                tot += F.cross_entropy(mask_pred, true_masks).item()
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                #for j, p in enumerate(pred):
                #   save_image(p.squeeze(),'evalout/' + names[j])
                diceval = dice_coeff(pred, true_masks).item()
                tot += diceval
            pbar.update()

    net.train()
    return tot / n_val
