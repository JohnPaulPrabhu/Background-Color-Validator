import os
import optuna
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn

from ignite.metrics import Accuracy, Loss
from ignite.engine import create_supervised_evaluator
from data_loading import train_loader, val_loader, save_path
from model import BlurClassifier
from training import train_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def objective(trial):
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-4)
    step_size = trial.suggest_int("step_size", 5, 10)
    gamma = trial.suggest_float("gamma", 0.1, 0.5)
    patience = trial.suggest_int("patience", 3, 10)
    
    model = BlurClassifier(dropout_rate).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    train_model(model, criterion, optimizer, scheduler, max_epochs=5, patience=patience)

    model.load_state_dict(torch.load(os.path.join(save_path, "best_model.pth")))

    evaluator = create_supervised_evaluator(model, metrics={'accuracy': Accuracy(), 'loss': Loss(criterion)}, device=device)
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    accuracy = metrics['accuracy']
    return accuracy

study = optuna.create_study(direction="maximize")   
study.optimize(objective, n_trials=10)

print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# Save best hyperparameters
best_params = trial.params
best_params['accuracy'] = trial.value

# Save best_params to a file
import json
with open("best_params.json", "w") as f:
    json.dump(best_params, f)