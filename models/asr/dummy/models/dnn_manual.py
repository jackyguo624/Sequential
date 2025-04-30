import pytorch_lightning as pl
import torch.nn as nn
import torch
import os
import copy

class Dnn(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.save_hyperparameters()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.automatic_optimization = False

    def forward(self, x):
        dtype = next(self.parameters()).dtype
        x = x.to(dtype)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


    def pg(self):
        print(f'self.fc1.weight.data:{self.fc1.weight.data[0][0:5]}, grad: {self.fc1.weight.grad[0][0:5] if self.fc1.weight.grad is not None else None}')
        print(f'self.strategy.fc1.weight.data:{self._trainer.strategy.model.fc1.weight.data[0][0:5]}, grad: {self._trainer.strategy.model.fc1.weight.grad[0][0:5] if self._trainer.strategy.model.fc1.weight.grad is not None else None}')

    def training_step(self, batch, batch_idx):
        inputs = batch['inputs']

        if "DeepSpeed" in self._trainer.strategy.__class__.__name__:
            outputs = self._trainer.strategy.model(inputs)
            inputs = inputs.to(outputs.dtype)
            loss = nn.functional.mse_loss(outputs, inputs)
            self._trainer.strategy.model.backward(loss)
            
            # Manually increment the global step counter to trigger checkpoint callbacks
            self._trainer.fit_loop.epoch_loop\
                .manual_optimization\
                .optim_step_progress\
                .total\
                .completed += 1
            
            # Then call model.step() to update weights with DeepSpeed
            self._trainer.strategy.model.step()

        else:
            outputs = self.forward(inputs)
            inputs = inputs.to(outputs.dtype)
            loss = nn.functional.mse_loss(outputs, inputs)
            opt = self.optimizers()
            opt.zero_grad()
            self.manual_backward(loss)
            opt.step()

        self.log('train_loss', loss, on_epoch=True, logger=True, batch_size=inputs.shape[0])

    def validation_step(self, batch, batch_idx):
        inputs = batch['inputs']

        if "DeepSpeed" in self._trainer.strategy.__class__.__name__:
            #outputs = self._trainer.strategy.model(inputs)
            outputs = self.forward(inputs)
            inputs = inputs.to(outputs.dtype)
            loss = nn.functional.mse_loss(outputs, inputs)
        else:
            outputs = self.forward(inputs)
            inputs = inputs.to(outputs.dtype)
            loss = nn.functional.mse_loss(outputs, inputs)

        self.log('val_loss', loss, on_epoch=True, logger=True, batch_size=inputs.shape[0])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())