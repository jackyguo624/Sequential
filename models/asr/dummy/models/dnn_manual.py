import pytorch_lightning as pl
import torch.nn as nn
import torch

class Dnn(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.save_hyperparameters()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.automatic_optimization = False

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        inputs = batch['inputs']
        print(inputs.device)
        outputs = self.forward(inputs)
        inputs = inputs.to(outputs.dtype)
        print(f"outputs.dtype: {outputs.dtype}, inputs.dtype: {inputs.dtype}")
        loss = nn.functional.mse_loss(outputs, inputs)
        
        opt = self.optimizers()
        opt.zero_grad()
        
        self.manual_backward(loss)
        opt.step()
        
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())