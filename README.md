## Deep Learning with PyTorch - Image Classifier

To run this project, please use:

```bash
rm -rf logs/
rm -rf saved_models/
rm -rf plots/
mkdir logs
mkdir saved_models
mkdir plots

## resnet 152 comparison
python train.py "./flowers" --gpu --epochs 20 --arch resnet152

## Fine tuning the top layers only
python train.py "./flowers" --gpu --epochs 20

## Fine tuning the whole model
python train.py "./flowers" --gpu --epochs 20 --all True

## Train the whole model from scratch
python train.py "./flowers" --gpu --epochs 20 --scratch True --all True

## Train the own model 1
python train.py "./flowers" --gpu --epochs 20 --scratch True --all True --arch mymodel1

## Train the own model 2
python train.py "./flowers" --gpu --epochs 20 --scratch True --all True --arch mymodel2
```

