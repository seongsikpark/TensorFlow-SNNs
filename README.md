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
|VGG16      | 15.27M | 94.70 | |
|----|----|----|----|----|----|
|ResNet20   |  4.34M | 96.16 | NIPA |  | 
|ResNet32   |  7.45M | 96.45 | NIPA |  | 
|ResNet44   | 10.55M | 96.68 | NIPA | | 
|ResNet56   | 13.66M | 96.32 | NIPA | | ep-1000, lr-step-200
|----|----|----|----|----|----|
|ResNet18   | 11.19M | 95.96 | NIPA | |
|ResNet34   | 21.31M | 96.22 | NIPA | | 
|ResNet50   | 23.60M | 95.44 | NIPA | |
|----|----|----|----|----|----|
|ResNet20V2 |  4.34M | 96.26 | NIPA | | 
|ResNet32V2 |  7.45M | 96.36 | NIPA |
|ResNet44V2 | 10.55M | 96.34 | NIPA | | 
|ResNet56V2 | 13.66M | 96.69 | NIPA | | 
|----|----|----|----|----|----|
|ResNet18V2 | 11.19M | 95.49 | NIPA | 
|ResNet34V2 | 21.31M | 96.07 | NIPA | 
|ResNet50V2 | 23.61M | 95.12 | NIPA |



CIFAR-100

|Models|Para.|Acc.|Server|Best Model|Note|
|----|----|----|----|----|----|
|VGG16      | 15.31M | 72.47 |      | ep-300_bat-100_opt-SGD_lr-step-5E-04_lmb-1E-05_sc_cm
|ResNet20   |  4.36M | 75.05 | NIPA |
|ResNet32   |  7.47M | 75.80 | NIPA |
|ResNet44   | 10.58M | 76.03 | NIPA |
|ResNet56   | 13.68M | 77.27 | NIPA |
|----|----|----|----|----|----|
|ResNet18   | 11.24M | 72.33 | NIPA | 
|ResNet34   | 21.36M | 73.01 | NIPA | 
|ResNet50   | 23.78M | 72.37 | NIPA | ep-1000, lr-step-200
|----|----|----|----|----|----|
|ResNet20V2 |  4.37M | 75.02 | NIPA | ep-1000, lr-step-200
|ResNet32V2 |  7.47M | 76.06 | NIPA |
|ResNet44V2 | 10.58M | 76.27 | NIPA | ep-1000, lr-step-200
|ResNet56V2 | 13.68M | 76.82 | NIPA | ep-1000, lr-step-200
|----|----|----|----|----|----|
|ResNet18V2 | 11.24M | 71.60 | NIPA | 
|ResNet34V2 | 21.36M | 73.65 | NIPA | 
|ResNet50V2 | 23.79M | 71.25 | NIPA | ep-1000, lr-step-200



ImageNet (from Keras)

| Models          | Para. | Acc.  | Acc.(top5) |Best Model|Note|
|-----------------|-------|-------|------------|----|----|
| VGG16           |       |       |
| ResNet50        |       |       |
| ResNet101       | 44.7M | 76.31 | 92.76
| ResNet152       | 60.4M | 76.58 | 93.13
| MobileNetV2     | 3.5M  | 72.01 | 90.60
| EfficientNetV2S | 21.6M | 83.92 | 96.73



##
ImageNet

## Pretrained models 
VGG     - ep-1000_bat-100_opt-SGD_lr-STEP-5E-03_lmb-1E-05_sc_cm
ResNet  - ep-300_bat-100

