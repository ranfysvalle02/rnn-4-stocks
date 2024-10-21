import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from sklearn.preprocessing import StandardScaler

# Generate synthetic stock data with reduced noise
def generate_synthetic_stock_data(seq_length, total_samples):
    x = np.linspace(0, total_samples, total_samples)
    stock_prices = np.sin(0.02 * x) + np.random.normal(scale=0.1, size=total_samples)
    return stock_prices

# Parameters
TOTAL_SAMPLES = 1000
SEQ_LENGTH = 50  # Increased sequence length
TEST_SIZE = 100  # Size of the test set

# Generate data
stock_prices = generate_synthetic_stock_data(SEQ_LENGTH, TOTAL_SAMPLES)

# Prepare the dataset
class StockPriceDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

        # Scale the data
        self.scaler = StandardScaler()
        self.scaled_data = self.scaler.fit_transform(self.data.reshape(-1, 1)).flatten()

    def __len__(self):
        return len(self.scaled_data) - self.seq_length

    def __getitem__(self, idx):
        seq = self.scaled_data[idx:idx + self.seq_length]
        target = self.scaled_data[idx + self.seq_length]
        seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(1)  # Shape: (seq_length, 1)
        target = torch.tensor(target, dtype=torch.float32)
        return seq, target

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data.reshape(-1, 1)).flatten()

# Create dataset and split into training and validation sets
dataset = StockPriceDataset(stock_prices, SEQ_LENGTH)
train_size = len(dataset) - TEST_SIZE
train_dataset, test_dataset = random_split(dataset, [train_size, TEST_SIZE])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=64)

# Define the LSTM model with increased complexity
class StockPriceLSTM(pl.LightningModule):
    def __init__(self, input_size=1, hidden_size=128, num_layers=3, output_size=1, learning_rate=0.001):
        super(StockPriceLSTM, self).__init__()
        self.save_hyperparameters()
        self.lstm = nn.LSTM(
            input_size=self.hparams.input_size,
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.num_layers,
            batch_first=True,
            dropout=0.2  # Add dropout to prevent overfitting
        )
        self.fc = nn.Linear(self.hparams.hidden_size, self.hparams.output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze()

    def training_step(self, batch, batch_idx):
        sequences, targets = batch
        outputs = self(sequences)
        loss = nn.functional.mse_loss(outputs, targets)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        sequences, targets = batch
        outputs = self(sequences)
        loss = nn.functional.mse_loss(outputs, targets)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

# Instantiate and train the model
model = StockPriceLSTM()
trainer = pl.Trainer(
    max_epochs=100,
    logger=False,
    enable_checkpointing=False,
    devices=1,
    accelerator='auto'
)
trainer.fit(model, train_loader, val_loader)

# Evaluate the model
model.eval()

# Prepare test data
test_sequences = []
test_targets = []
for seq, target in test_dataset:
    test_sequences.append(seq)
    test_targets.append(target.item())

test_sequences = torch.stack(test_sequences)
test_targets = np.array(test_targets)

with torch.no_grad():
    predictions = model(test_sequences).cpu().numpy()

# Inverse transform the predictions and targets
predictions_rescaled = dataset.inverse_transform(predictions)
actual_values_rescaled = dataset.inverse_transform(test_targets)

# Print predictions vs actual values
print("\nPredictions vs Actual Values (Rescaled):")
for i in range(len(predictions_rescaled)):
    print(f"Prediction {i+1}: {predictions_rescaled[i]:.4f}, Actual: {actual_values_rescaled[i]:.4f}")

"""
...
Prediction 90: 0.5841, Actual: 0.3548
Prediction 91: 0.3346, Actual: 0.2394
Prediction 92: -0.3991, Actual: -0.4028
Prediction 93: -0.9224, Actual: -0.9778
Prediction 94: 0.9093, Actual: 0.9093
Prediction 95: 0.2804, Actual: 0.1847
Prediction 96: 0.4799, Actual: 0.4354
Prediction 97: -0.9767, Actual: -1.0592
Prediction 98: 0.0725, Actual: 0.0610
Prediction 99: 0.9450, Actual: 0.9535
Prediction 100: -0.7006, Actual: -0.7460
"""
