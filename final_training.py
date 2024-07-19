import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers import EarlyStopping, ModelCheckpoint
import os
import json

from data_loading import train_loader, val_loader, save_path
from model import BlurClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load best hyperparameters from file
with open("best_params.json", "r") as f:
    best_params = json.load(f)

best_dropout_rate = best_params['dropout_rate']
best_learning_rate = best_params['learning_rate']
best_weight_decay = best_params['weight_decay']
best_step_size = best_params['step_size']
best_gamma = best_params['gamma']
best_patience = best_params['patience']

final_model = BlurClassifier(best_dropout_rate).to(device)
final_criterion = nn.CrossEntropyLoss()
final_optimizer = optim.AdamW(final_model.parameters(), lr=best_learning_rate, weight_decay=best_weight_decay)
final_scheduler = StepLR(final_optimizer, step_size=best_step_size, gamma=best_gamma)

final_trainer = create_supervised_trainer(final_model, final_optimizer, final_criterion, device=device)
final_evaluator = create_supervised_evaluator(final_model, metrics={'accuracy': Accuracy(), 'loss': Loss(final_criterion)}, device=device)

final_handler = EarlyStopping(patience=best_patience, score_function=lambda engine: -engine.state.metrics['loss'], trainer=final_trainer)
final_evaluator.add_event_handler(Events.COMPLETED, final_handler)

final_checkpointer = ModelCheckpoint(dirname='.', filename_prefix='best_final', n_saved=1, create_dir=True, require_empty=False)
final_evaluator.add_event_handler(Events.COMPLETED, final_checkpointer, {'model': final_model})

final_best_model_score = [float('inf')]

@final_trainer.on(Events.EPOCH_COMPLETED)
def log_final_training_results(trainer):
    final_evaluator.run(train_loader)
    train_metrics = final_evaluator.state.metrics
    train_loss = train_metrics['loss']
    train_accuracy = train_metrics['accuracy']
    print(f"Final Training Results - Epoch: {trainer.state.epoch}  Avg loss: {train_loss:.2f} Avg accuracy: {train_accuracy:.2f}")

    torch.save(final_model.state_dict(), os.path.join(save_path, f"final_model_epoch_{trainer.state.epoch}.pth"))

@final_trainer.on(Events.EPOCH_COMPLETED)
def log_final_validation_results(trainer):
    final_evaluator.run(val_loader)
    val_metrics = final_evaluator.state.metrics
    val_loss = val_metrics['loss']
    val_accuracy = val_metrics['accuracy']
    print(f"Final Validation Results - Epoch: {trainer.state.epoch}  Avg loss: {val_loss:.2f} Avg accuracy: {val_accuracy:.2f}")
    final_scheduler.step()

    if val_loss < final_best_model_score[0]:
        final_best_model_score[0] = val_loss
        torch.save(final_model.state_dict(), os.path.join(save_path, "final_best_model.pth"))

final_trainer.run(train_loader, max_epochs=100)
