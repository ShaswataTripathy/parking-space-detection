import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import os
import opendatasets as od
from tqdm import tqdm
import gradio as gr

# Step 2: Download and Preprocess Dataset
dataset_url = "https://www.kaggle.com/datasets/ljp0726/parking-space-detection"
output_dir = "parking_dataset"
od.download(dataset_url, output_dir)

def load_dataset(dataset_path):
    images = []
    annotations = []
    for img_file in os.listdir(os.path.join(dataset_path, "images")):
        img_path = os.path.join(dataset_path, "images", img_file)
        annotation_path = os.path.join(dataset_path, "annotations", img_file.replace(".jpg", ".txt"))
        if os.path.exists(annotation_path):
            with open(annotation_path, "r") as f:
                bboxes = []
                labels = []
                for line in f.readlines():
                    data = list(map(float, line.strip().split()))
                    x, y, w, h = data[1:5]
                    x1 = x - w / 2
                    y1 = y - h / 2
                    x2 = x + w / 2
                    y2 = y + h / 2
                    bboxes.append([x1, y1, x2, y2])
                    labels.append(1 if data[0] == 1 else 0)
                images.append(img_path)
                annotations.append({"boxes": torch.tensor(bboxes, dtype=torch.float32), "labels": torch.tensor(labels, dtype=torch.int64)})
    return images, annotations

dataset_path = os.path.join(output_dir, "parking-space-detection")
image_paths, annotations = load_dataset(dataset_path)

# Step 3: Load Pretrained Faster R-CNN Model
model = fasterrcnn_resnet50_fpn(pretrained=True)

# Modify the classifier for parking detection (2 classes: occupied & empty)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 2)

# Step 4: Convert Model for Deployment
def convert_model(model, device):
    model.eval()
    example_input = torch.rand(1, 3, 224, 224).to(device)
    traced_model = torch.jit.trace(model, example_input)
    traced_model.save("faster_rcnn_parking_scripted.pt")
    print("Model converted and saved as TorchScript!")

# Step 5: Define Inference Function
def predict_and_visualize(image_path):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    image = Image.open(image_path).convert("RGB")
    transform = torchvision.transforms.ToTensor()
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        predictions = model(image_tensor)
    
    boxes = predictions[0]['boxes'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()
    
    draw = ImageDraw.Draw(image)
    occupied_count = 0
    empty_count = 0
    for i in range(len(boxes)):
        if scores[i] > 0.5:  # Confidence threshold
            color = "red" if labels[i] == 1 else "green"
            draw.rectangle(boxes[i], outline=color, width=3)
            if labels[i] == 1:
                occupied_count += 1
            else:
                empty_count += 1
    
    return image, f"Occupied Spots: {occupied_count}, Empty Spots: {empty_count}"

# Step 6: Deploy Model with Gradio
def gradio_interface(image):
    image.save("temp.jpg")
    result, count_info = predict_and_visualize("temp.jpg")
    return result, count_info

iface = gr.Interface(fn=gradio_interface, inputs=gr.Image(type="pil"), outputs=["image", "text"], title="Parking Space Occupancy Detection")

# Step 7: Train and Save Model
dataset = ParkingDataset(image_paths, annotations)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 10
for epoch in range(num_epochs):
    loss = train_one_epoch(model, optimizer, data_loader, device)
    print(f"Epoch {epoch+1}, Loss: {loss}")

# Convert and Save Model for Deployment
convert_model(model, device)

torch.save(model.state_dict(), "faster_rcnn_parking.pth")

# Step 8: Launch Gradio Interface
iface.launch()
