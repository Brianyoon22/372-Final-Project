# Setup Instructions

## Prerequisites

- A free [Google account](https://accounts.google.com) (for Google Colab and Google Drive)
- A free [Kaggle account](https://www.kaggle.com) (to download the dataset)
- A modern web browser (Chrome recommended)

No local Python installation is required — all code runs in Google Colab.

## Step 1 — Download the Dataset

1. Log into [Kaggle](https://www.kaggle.com) and go to the [Sports Classification dataset](https://www.kaggle.com/datasets/gpiosenka/sports-classification).
2. Click the **Download** button to download `sports-classification.zip` (~1.3 GB).
3. Go to [Google Drive](https://drive.google.com) and drag `sports-classification.zip` into the root of your Drive (not inside any folder).

## Step 2 — Open the Notebook in Colab

1. Go to [colab.research.google.com](https://colab.research.google.com).
2. Click **File → Upload notebook** and select `finalproject_clean.ipynb`.

## Step 3 — Enable GPU

1. In Colab, click **Runtime → Change runtime type**.
2. Set Hardware Accelerator to **T4 GPU**.
3. Click **Save**.

## Step 4 — Run the Notebook

Run all cells in order from top to bottom. The notebook will:

1. Install all dependencies automatically (`torch`, `torchvision`, `timm`, `gradio`, `scikit-learn`)
2. Mount your Google Drive and unzip the dataset
3. Train EfficientNet-B0 for 10 epochs (~25 min) and ResNet50 for 3 epochs (~10 min)
4. Save model checkpoints to your Google Drive
5. Evaluate both models and display results
6. Launch a Gradio web app with a public URL

**Total runtime: approximately 40–45 minutes on a T4 GPU.**

## Step 5 — Use the Web App

When the final cell runs, Colab will print a public URL like:
```
Running on public URL: https://xxxxxxxx.gradio.live
```
Open that URL in any browser to use the sports classifier. Upload any sports image and the model will return the top 5 predicted sports with confidence scores.

## Troubleshooting

**"FileNotFoundError: sports-classification.zip"** — Make sure the zip file is in the root of your Google Drive (not in a subfolder). The notebook looks for it at `/content/drive/MyDrive/sports-classification.zip`.

**"CUDA not available"** — You forgot to enable the GPU. Go to Runtime → Change runtime type → T4 GPU and re-run all cells.

**Session disconnected mid-training** — Model checkpoints are saved to Google Drive after each epoch. Re-run all cells; the notebook will re-mount Drive and reload the saved checkpoints rather than starting from scratch.

**Gradio URL not working** — The public URL expires after 72 hours. Re-run the final cell to generate a new one.

## Dependencies

All dependencies are installed automatically in Cell 1. For reference:

```
torch
torchvision
timm
gradio
scikit-learn
Pillow
matplotlib
numpy
```

See `requirements.txt` for pinned versions.
