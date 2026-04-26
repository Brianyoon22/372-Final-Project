# Model Card: Sports Image Classifier

## Model
EfficientNet-B0 fine-tuned on 100 sports categories.

## Training Data
Sports Classification dataset (gpiosenka) from Kaggle — 13,492 training images across 100 sport classes.

## Performance
| Metric | Score |
|---|---|
| Test Accuracy | 97.40% |
| Precision | 97.86% |
| Recall | 97.40% |
| F1 Score | 97.35% |

## Limitations
Most errors occur between visually similar sports (e.g., sidecar vs motorcycle racing). The model was trained on clean, well-framed sports images and may perform worse on unusual angles or low-quality photos.

## Usage
See README.md and SETUP.md for instructions.
