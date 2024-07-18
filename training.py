import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers import EarlyStopping
import os

from data_loading import train_loader, val_loader, save_path
from model import BlurClassifier

def train_model(model, criterion, optimizer, scheduler, max_epochs, patience):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
    evaluator = create_supervised_evaluator(model, metrics={'accuracy': Accuracy(), 'loss': Loss(criterion)}, device=device)

    handler = EarlyStopping(patience=patience, score_function=lambda engine: -engine.state.metrics['loss'], trainer=trainer)
    evaluator.add_event_handler(Events.COMPLETED, handler)

    best_model_score = [float('inf')]

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator.run(train_loader)
        train_metrics = evaluator.state.metrics
        train_loss = train_metrics['loss']
        train_accuracy = train_metrics['accuracy']
        print(f"Training Results - Epoch: {trainer.state.epoch}  Avg loss: {train_loss:.2f} Avg accuracy: {train_accuracy:.2f}")

        torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{trainer.state.epoch}.pth"))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(val_loader)
        val_metrics = evaluator.state.metrics
        val_loss = val_metrics['loss']
        val_accuracy = val_metrics['accuracy']
        print(f"Validation Results - Epoch: {trainer.state.epoch}  Avg loss: {val_loss:.2f} Avg accuracy: {val_accuracy:.2f}")
        scheduler.step()

        if val_loss < best_model_score[0]:
            best_model_score[0] = val_loss
            torch.save(model.state_dict(), os.path.join(save_path, "best_model.pth"))

    trainer.run(train_loader, max_epochs=max_epochs)
