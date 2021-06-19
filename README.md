# UNet: semantic segmentation with PyTorch



Customized implementation of the [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet) by milesial with a self supervised learning task (SimCLR) added before the downstream task.

Both models were trained on the [EchoNet-Dynamic Dataset](https://echonet.github.io/dynamic/index.html#dataset).

## Usage
**Note : Use Python 3.6 or newer**

### Training SimCLR

`python simclr.py`

### Training UNet

```shell script
> python train.py -h
usage: train.py [-h] [-e E] [-b [B]] [-l [LR]] [-f LOAD] [-s SCALE] [-v VAL]

Train the UNet on images and target masks

optional arguments:
  -h, --help            show this help message and exit
  -e E, --epochs E      Number of epochs (default: 5)
  -b [B], --batch-size [B]
                        Batch size (default: 1)
  -l [LR], --learning-rate [LR]
                        Learning rate (default: 0.1)
  -f LOAD, --load LOAD  Load model from a .pth file (default: False)
  -s SCALE, --scale SCALE
                        Downscaling factor of the images (default: 0.5)
  -v VAL, --validation VAL
                        Percent of the data that is used as validation (0-100)
                        (default: 15.0)

```
By default, the `scale` is 0.5, so if you wish to obtain better results (but use more memory), set it to 1.

The input images and target masks should be in the `data/imgs` and `data/masks` folders respectively.


### Prediction and Evaluation

After training your UNet model and saving it to MODEL.pth, you can easily test the output masks on your images and view the dice coefficient via the CLI.

To predict and evaluate the model performance on the test images:

`python predict_and_eval.py`

### SimCLR Pretrained model
A pretrained SimCLR model `./simclrunet.pth` is already uploaded in the repository.

## Tensorboard
You can visualize in real time the train and test losses, the weights and gradients, along with the model predictions with tensorboard:

`tensorboard --logdir=runs`

You can find a reference training run with the Caravana dataset on [TensorBoard.dev](https://tensorboard.dev/experiment/1m1Ql50MSJixCbG1m9EcDQ/#scalars&_smoothingWeight=0.6) (only scalars are shown currently).


---

Original paper by Olaf Ronneberger, Philipp Fischer, Thomas Brox: [https://arxiv.org/abs/1505.04597](https://arxiv.org/abs/1505.04597)

![network architecture](https://i.imgur.com/jeDVpqF.png)
