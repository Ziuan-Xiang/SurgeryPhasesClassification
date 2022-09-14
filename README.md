# Resnet Training - Validation - Testing

Please contibute your code here so that everyone can learn and review. 

## Cases

- Surg01-V120
- Surg01-V188
- SurgAll-V40-C


## Template Contents

- configs/: label merging configs
- model/: trained models to upload to Google Drive (DO NOT upload to Github)

- data.py: datasets and loaders, phase mapper
- utils: trainers and evaluators, metrics recorder

- main.ipynb: training and evalutation template

## Others

- Dataset/: Cut frames, give the original labels and merge the small amount labels
- GradCam/: Grad cam algorithm for the resnet to check whether the text below the image is important

## Usage

Please find the details in main.ipynb

1. Build training, validation and test sets, specify transforms and label merging
2. Load the model (ResNet50 pretrained on ImageNet)
3. Login to W&B
4. Build trainer and train/validate the model
5. Build evaluator and test the model

## Notes

- Use indicative run names, which will be reflected to your W&B logs and saved models
- Training/validation results are (supposed to be) automatically documented in W&B in [the surgery project](https://wandb.ai/eezklab/surgery-hernia)
- Testing results should be kept locally by yourself (for now)

