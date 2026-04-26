# Sports Image Classifier

A deep learning project that classifies images across 100 sports categories using fine-tuned CNNs, achieving 98.0% test accuracy and deployed as an interactive web application.

## What it Does

This project fine-tunes deep learning models to classify images across 100 sports categories, achieving up to 98.0% test accuracy. Given any image of a sporting event or athlete, the model predicts which sport is being depicted — from archery to water polo — and returns the top 5 most likely categories with confidence scores. The project compares two pretrained CNN architectures (EfficientNet-B0 and ResNet50), applies data augmentation and regularization to improve generalization, performs error analysis to understand failure cases, and deploys the best model as an interactive web application where anyone can upload a sports photo and get an instant prediction.

## Quick Start

**Requirements:** A Google account (for Colab + Drive) and the dataset zip from Kaggle.

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/gpiosenka/sports-classification) and upload `sports-classification.zip` to the root of your Google Drive.
2. Open `notebooks/finalproject_clean.ipynb` in [Google Colab](https://colab.research.google.com).
3. Enable GPU: Runtime → Change runtime type → **T4 GPU** → Save.
4. Run all cells top to bottom. Total training time ~40 minutes.
5. The final cell launches a Gradio web app with a shareable public URL.

See [SETUP.md](SETUP.md) for detailed setup and troubleshooting instructions.

## Video Links

- **Demo Video:** [(https://duke.box.com/s/lyp2cnoj8i1820e96wbszl86rofboqg6)]
- **Technical Demo:** [https://duke.box.com/s/le9a0c8cwoj0dgdqgoadjpnq85769qdp]

## Evaluation

Both models were evaluated on a held-out test set of 500 images across 100 sports classes:

| Model | Epochs | Test Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|---|
| Random Baseline | — | 0.80% | N/A | N/A | N/A |
| EfficientNet-B0 | 10 | 98.00% | 98.38% | 98.00% | 97.97% |
| ResNet50 | 3 | 97.00% | 97.64% | 97.00% | 96.93% |

**Key findings:**
- EfficientNet-B0 reached 98.0% accuracy after 10 epochs, outperforming ResNet50 at 97.0% — demonstrating that the lighter EfficientNet architecture generalizes better on this dataset given sufficient training time.
- Only 10 out of 500 test images were misclassified. Nearly all errors involved visually similar sports (e.g., sidecar ↔ motorcycle racing, snow boarding ↔ giant slalom.) — mistakes a human viewer might also make.
- Both models dramatically outperform the 0.80% random baseline, demonstrating the effectiveness of ImageNet transfer learning for sports classification.

## Repository Structure
sports-image-classifier/
├── README.md
├── SETUP.md
├── ATTRIBUTION.md
├── requirements.txt
├── notebooks/
│ └── finalproject_clean.ipynb
├── src/
│ └── (no standalone scripts — all code is in the notebook)
├── data/
│ └── (dataset downloaded from Kaggle — see SETUP.md)
├── models/
│ └── (best_model.pth and best_resnet.pth saved to Google Drive during training)
├── videos/
│ ├── demo.mp4
│ └── technical_demo.mp4
└── docs/
└── (additional documentation)
