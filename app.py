import json
import torch
import timm
import gradio as gr
from torchvision import transforms
from huggingface_hub import hf_hub_download

HF_REPO_ID = "YOUR_HF_USERNAME/sports-classifier"  # <-- update this
MODEL_FILE  = "best_model.pth"

model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILE)

with open("classes.json") as f:
    class_names = json.load(f)

device = torch.device("cpu")
model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=100)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def predict(image):
    img = transform(image).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(img), dim=1)
        top5  = probs.topk(5)
    return {class_names[idx]: float(prob)
            for prob, idx in zip(top5.values[0], top5.indices[0])}

gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=5),
    title="Sports Image Classifier",
    description="Upload a sports image and the model will predict which of 100 sports it is!",
).launch()
