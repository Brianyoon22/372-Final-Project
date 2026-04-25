# Sports Image Classifier

A deep learning project that classifies images across 100 sports categories using fine-tuned CNNs, achieving 96.4% test accuracy and deployed as an interactive web application.

## What it Does

This project fine-tunes deep learning models to classify images across 100 sports categories, achieving up to 96.4% test accuracy. Given any image of a sporting event or athlete, the model predicts which sport is being depicted — from archery to water polo — and returns the top 5 most likely categories with confidence scores. The project compares two pretrained CNN architectures (EfficientNet-B0 and ResNet50), applies data augmentation and regularization to improve generalization, performs error analysis to understand failure cases, and deploys the best model as an interactive web application where anyone can upload a sports photo and get an instant prediction.

## Quick Start

**Requirements:** A Google account (for Colab + Drive) and the dataset zip from Kaggle.

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/gpiosenka/sports-classification) and upload `sports-classification.zip` to the root of your Google Drive.
2. Open `notebooks/finalproject_clean.ipynb` in [Google Colab](https://colab.research.google.com).
3. Enable GPU: Runtime → Change runtime type → **T4 GPU** → Save.
4. Run all cells top to bottom. Total training time ~40 minutes.
5. The final cell launches a Gradio web app with a shareable public URL.

See [SETUP.md](SETUP.md) for detailed setup and troubleshooting instructions.

## Video Links

- **Demo Video:** [Link here]
- **Technical Walkthrough:** [Link here]

## Evaluation

Both models were evaluated on a held-out test set of 500 images across 100 sports classes:

| Model | Epochs | Test Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|---|
| Random Baseline | — | 1.20% | N/A | N/A | N/A |
| EfficientNet-B0 | 10 | 94.80% | 95.10% | 94.80% | 94.70% |
| ResNet50 | 3 | 96.40% | 96.50% | 96.40% | 96.33% |

**Key findings:**
- ResNet50 reached 96.4% accuracy in just 3 epochs, outperforming EfficientNet-B0 trained for 10 epochs — demonstrating a meaningful convergence speed tradeoff between architectures.
- Only 15 out of 500 test images were misclassified. Nearly all errors involved visually similar sports (e.g., sidecar ↔ motorcycle racing, snow boarding ↔ giant slalom.) — mistakes a human viewer might also make.
- Both models dramatically outperform the 1.2% random baseline, demonstrating the effectiveness of ImageNet transfer learning for sports classification.

## Repository Structure

```
sports-image-classifier/
├── README.md
├── SETUP.md
├── ATTRIBUTION.md
├── requirements.txt
├── notebooks/
│   └── finalproject_clean.ipynb
├── src/
│   └── (no standalone scripts — all code is in the notebook)
├── data/
│   └── (dataset downloaded from Kaggle — see SETUP.md)
├── models/
│   └── (best_model.pth and best_resnet.pth saved to Google Drive during training)
├── videos/
│   ├── demo.mp4
│   └── walkthrough.mp4
└── docs/
    └── (additional documentation)
```
