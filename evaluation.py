import torch
from sklearn.metrics import classification_report
import json
import os

from data_loading import test_loader
from model import BlurClassifier
from final_training import save_path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load best hyperparameters from file
with open("best_params.json", "r") as f:
    best_params = json.load(f)

best_dropout_rate = best_params['dropout_rate']

final_model = BlurClassifier(best_dropout_rate).to(device)
final_model.load_state_dict(torch.load(os.path.join(save_path, "final_best_model.pth")))
final_model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = final_model(images)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

print(classification_report(y_true, y_pred, target_names=['Sharp', 'Blur']))
