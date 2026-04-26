"""
Loads the trained EfficientNet-B0 model and runs inference on an image.

Usage:
    python models/load_model.py --model models/best_model.pth --image path/to/image.jpg
"""
import json
import argparse
import torch
import timm
from torchvision import transforms
from PIL import Image

def load_model(model_path, num_classes=100):
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

def predict(model, image_path, class_names):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    img = transform(image).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(img), dim=1)
        top5 = probs.topk(5)
    print("Top 5 predictions:")
    for prob, idx in zip(top5.values[0], top5.indices[0]):
        print(f"  {class_names[idx]}: {prob:.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to best_model.pth")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--classes", default="classes.json", help="Path to classes.json")
    args = parser.parse_args()

    with open(args.classes) as f:
        class_names = json.load(f)

    model = load_model(args.model)
    predict(model, args.image, class_names)
