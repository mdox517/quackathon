from ultralytics import YOLO
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import requests
from io import BytesIO
from torchvision.models import efficientnet_b0

detector = YOLO("/root/Projects/Hackathon/MacroVision/runs/detect/train9/weights/best.pt")  # Update with your YOLO model path

class FoodClassifier(nn.Module):
    def __init__(self, num_classes=101):
        super().__init__()
        base_model = efficientnet_b0(pretrained=True)
        
        self.features = base_model.features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(base_model.classifier[1].in_features, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return self.classifier(x)

food_classifier = FoodClassifier(num_classes=101)

food_classifier.load_state_dict(
    torch.load('/root/Projects/Hackathon/MacroVision/food-101/food_classifier.pth', map_location='cpu')
)
food_classifier.eval()

food_classes = [
    "apple_pie",
    "baby_back_ribs",
    "baklava",
    "beef_carpaccio",
    "beef_tartare",
    "beet_salad",
    "beignets",
    "bibimbap",
    "bread_pudding",
    "breakfast_burrito",
    "bruschetta",
    "caesar_salad",
    "cannoli",
    "caprese_salad",
    "carrot_cake",
    "ceviche",
    "cheesecake",
    "cheese_plate",
    "chicken_curry",
    "chicken_quesadilla",
    "chicken_wings",
    "chocolate_cake",
    "chocolate_mousse",
    "churros",
    "clam_chowder",
    "club_sandwich",
    "crab_cakes",
    "creme_brulee",
    "croque_madame",
    "cup_cakes",
    "deviled_eggs",
    "donuts",
    "dumplings",
    "edamame",
    "eggs_benedict",
    "escargots",
    "falafel",
    "filet_mignon",
    "fish_and_chips",
    "foie_gras",
    "french_fries",
    "french_onion_soup",
    "french_toast",
    "fried_calamari",
    "fried_rice",
    "frozen_yogurt",
    "garlic_bread",
    "gnocchi",
    "greek_salad",
    "grilled_cheese_sandwich",
    "grilled_salmon",
    "guacamole",
    "gyoza",
    "hamburger",
    "hot_and_sour_soup",
    "hot_dog",
    "huevos_rancheros",
    "hummus",
    "ice_cream",
    "lasagna",
    "lobster_bisque",
    "lobster_roll_sandwich",
    "macaroni_and_cheese",
    "macarons",
    "miso_soup",
    "mussels",
    "nachos",
    "omelette",
    "onion_rings",
    "oysters",
    "pad_thai",
    "paella",
    "pancakes",
    "panna_cotta",
    "peking_duck",
    "pho",
    "pizza",
    "pork_chop",
    "poutine",
    "prime_rib",
    "pulled_pork_sandwich",
    "ramen",
    "ravioli",
    "red_velvet_cake",
    "risotto",
    "samosa",
    "sashimi",
    "scallops",
    "seaweed_salad",
    "shrimp_and_grits",
    "spaghetti_bolognese",
    "spaghetti_carbonara",
    "spring_rolls",
    "steak",
    "strawberry_shortcake",
    "sushi",
    "tacos",
    "takoyaki",
    "tiramisu",
    "tuna_tartare",
    "waffles"
]

preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def expand_crop(img, bbox, expand_factor=0.15):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    new_x1 = max(0, x1 - int(w * expand_factor))
    new_y1 = max(0, y1 - int(h * expand_factor))
    new_x2 = min(img.width, x2 + int(w * expand_factor))
    new_y2 = min(img.height, y2 + int(h * expand_factor))
    return img.crop((new_x1, new_y1, new_x2, new_y2))

def classify_food(crop_img):
    img_t = preprocess(crop_img).unsqueeze(0)
    with torch.no_grad():
        outputs = torch.nn.functional.softmax(food_classifier(img_t), dim=1)
    probs, indices = torch.topk(outputs, 3)  
    return [(food_classes[idx], prob.item()) for prob, idx in zip(probs[0], indices[0])]

def analyze_food(img, conf_threshold=0.3):
    if isinstance(img, str):
        img = Image.open(img)
    detections = detector.predict(img, conf=conf_threshold)
    
    results = []
    for box in detections[0].boxes:
        if box.conf.item() < conf_threshold:
            continue
            
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        if (x2 - x1) < 50 or (y2 - y1) < 50: 
            continue
            
        expanded_crop = expand_crop(img, (x1, y1, x2, y2))
        predictions = classify_food(expanded_crop)
        
        best_pred = next((p for p in predictions if p[1] > 0.5), None)
        if not best_pred:
            continue
            
        results.append({
            'bbox': [x1, y1, x2, y2],
            'food': best_pred[0],
            'confidence': best_pred[1],
            'all_predictions': predictions
        })
    
    return results