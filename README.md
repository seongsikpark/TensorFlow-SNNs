# Readme

Spiking Neural Network library based on Tensorflow & Python


# Sequence (?)
1. model
2. DNN inference
3. DNN inference w/ fused-bn
4. Save activation distributions for data-based weight normalization
5. SNN inference
6. 

## model configurations

## DNN inference

## fused bn


## data-based weight normalization
using training dataset
defalut: outlier-robust layer-wise normalization
refs -

## SNN inference
only with fused-bn


## Pretrained models (internal share, include trained conditions)

CIFAR10

|Models|Para.|Acc.|Server|Best Model|Note|
|----|----|----|----|----|----|
|VGG16      | 15.27M | 94.70 | | O 
|ResNet18   | 11.19M | 95.67 | NIPA |   | Bottleneck - old - re train
|ResNet20   |  4.34M | 96.03 | NIPA | O | Basic
|ResNet32   |  7.45M | 96.46 | NIPA | O | Basic
|ResNet34   | 21.31M | 96.23 | NIPA | | Bottleneck - old - re train
|ResNet44   | 10.55M | 96.68 | NIPA | | Basic
|ResNet50   | 23.60M | 95.57 | NIPA | | Bottleneck - old - re train
|ResNet56   | 13.66M | 96.17 | NIPA | | Basic


|ResNet18   | 11.19M | 93.73 | SJ2 |
|ResNet34   | 21.31M | 94.65 | SJ2 |
|ResNet50   | 23.60M | 94.47 | SJ2 |


|ResNet18V2 | 11.19M | 95.24 | NIPA |
|ResNet20V2 |  4.34M | 96.15 | NIPA |
|ResNet32V2 |  7.45M | 96.40 | NIPA |
|ResNet34V2 | 21.30M | 96.10 | NIPA |
|ResNet50V2 | 23.61M | 95.00 | NIPA |



CIFAR-100

|Models|Para.|Acc.|Server|Best Model|Note|
|----|----|----|----|----|----|
|VGG16      | 15.31M | 72.34 | |
|ResNet18   | 11.24M | 71.16 | NIPA | old
|ResNet20   |  4.36M | 75.07 | NIPA |
|ResNet32   |  7.47M | 75.78 | NIPA |
|ResNet34   | 21.36M | 74.46 | NIPA | old
|ResNet44   | 10.58M | 76.02 | NIPA |
|ResNet50   | 23.78M | 72.32 | NIPA | old
|ResNet56   | 13.68M | 77.27 | NIPA |

|ResNet18V2 | 11.24M | 71.40 | NIPA |
|ResNet20V2 |  4.37M | 74.93 | NIPA |
|ResNet32V2 |  7.47M | 76.04 | NIPA |
|ResNet34V2 | 21.36M | 73.77 | NIPA |
|ResNet50V2 | 23.79M | 70.56 | NIPA |




ImageNet

## Pretrained models 
VGG     - ep-1000_bat-100_opt-SGD_lr-STEP-5E-03_lmb-1E-05_sc_cm
ResNet  - ep-300_bat-100

