import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import pandas

from dice_loss import dice_coeff
from torchvision.utils import save_image


def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0
    dices = []
    pixels = []

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
                for j, p in enumerate(pred):
                   #save_image(p.squeeze(),'evalout/' + names[j])
                   singledice = dice_coeff(p, true_masks[j]).item()
                   dices.append(singledice)
                   pixels.append([names[j].split('_')[0] + '_' + names[j].split('_')[2], torch.count_nonzero(p).item()])

                   #print("-------------------")
                   #print(torch.count_nonzero(true_masks[j]).item())
                   #print(torch.count_nonzero(p).item())
                diceval = dice_coeff(pred, true_masks).item()
                tot += diceval
            pbar.update()

    pixels = sorted(pixels)
    ef = []

    for z in range(0, len(pixels), 2):
       n1 = pixels[z][0]
       n2 = pixels[z+1][0]
       big = pixels[z][1]
       small = pixels[z+1][1]
       #print(n1, big, n2, small, "EF: ", ((big-small)/big)*100)
       ef.append([pixels[z][0].split('_')[0], ((big-small)/big)*100])
    
    #print("Number of samples: ", len(dices))

    dices = np.asarray(dices)
    #print("Standard Deviation: ", np.std(dices))

    ef_error = []

    with open("FileList.csv") as myfile:
      data = pandas.read_csv(myfile, index_col ="FileName")
      for ele in ef:
         #print(data.loc[ele[0]]["EF"])
         ef_error.append(abs(data.loc[ele[0]]["EF"] - ele[1]))
    print("EF Mean Absolute Error: ", np.mean(np.asarray(ef_error)))

    net.train()
    return tot / n_val
