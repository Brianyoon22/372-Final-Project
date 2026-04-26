# Attribution

## Dataset

**Sports Classification Dataset**
- Source: [Kaggle — gpiosenka](https://www.kaggle.com/datasets/gpiosenka/sports-classification)
- License: Public domain / Kaggle dataset terms
- Description: 14,493 images across 100 sports categories, pre-split into train (13,492), validation (500), and test (500) sets. Images are 224×224 JPEGs.

## Pretrained Models

**EfficientNet-B0**
- Source: [`timm` library](https://github.com/huggingface/pytorch-image-models) (Hugging Face / Ross Wightman)
- Original paper: Tan & Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," ICML 2019.
- Pretrained on ImageNet-1K. Fine-tuned on the sports dataset as part of this project.

**ResNet50**
- Source: [`timm` library](https://github.com/huggingface/pytorch-image-models)
- Original paper: He et al., "Deep Residual Learning for Image Recognition," CVPR 2016.
- Pretrained on ImageNet-1K. Fine-tuned on the sports dataset as part of this project.

## Libraries & Frameworks

| Library | Version | Use |
|---|---|---|
| PyTorch | 2.x | Model training, GPU acceleration |
| torchvision | 0.x | Image transforms and DataLoader |
| timm | 1.x | Pretrained EfficientNet and ResNet models |
| Gradio | 4.x | Web application deployment |
| scikit-learn | 1.x | Evaluation metrics (F1, precision, recall) |
| matplotlib | 3.x | Training curves and error analysis plots |
| NumPy | 1.x | Array operations |
| Pillow | 10.x | Image loading |

## AI Development Tools

This project was developed with  assistance from **Claude (Anthropic)** via claude.ai. Claude was used to debug a DataLoader error and a Gradio compatibility issue.
