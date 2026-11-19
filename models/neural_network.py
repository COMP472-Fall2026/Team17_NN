
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class NeuralNetwork:
    def __init__(self, epochs=20, lr=0.001, batch_size=128):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def fit(self, X, y):
        """Train the neural network."""

        X = np.asarray(X, dtype=np.float32)
        self.classes_, y = np.unique(y, return_inverse=True)

        torch.manual_seed(42)

        input_dim = X.shape[1]
        num_classes = len(self.classes_)

        # Define the network        
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, num_classes)
        ).to(self.device)

        # Convert NumPy arrays into PyTorch tensors
        dataset = TensorDataset(
            torch.from_numpy(X),
            torch.tensor(y, dtype=torch.long)
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        loss_fn = nn.CrossEntropyLoss()
        
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=0.001
        )

        self.history = []

        # Training loop
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0

            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()

                outputs = self.model(X_batch)
                loss = loss_fn(outputs, y_batch)

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(X_batch)

            avg_loss = total_loss / len(dataset)
            self.history.append(avg_loss)

            print(
                f"Epoch {epoch + 1}/{self.epochs} "
                f"- Loss: {avg_loss:.4f}"
            )

        return self

    def predict(self, X):
        """Predict class labels."""

        self.model.eval()
        X = np.asarray(X, dtype=np.float32)

        predictions = []

        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                batch = torch.from_numpy(
                    X[start:start + self.batch_size]
                ).to(self.device)

                outputs = self.model(batch)
                predicted = outputs.argmax(dim=1)

                predictions.append(predicted.cpu().numpy())

        encoded = np.concatenate(predictions)
        return self.classes_[encoded]