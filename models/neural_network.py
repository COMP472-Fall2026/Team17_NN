import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class NeuralNetwork:
    def __init__(
        self,
        epochs=20,
        lr=0.001,
        batch_size=128,
        hidden=(256, 64),
        dropout=0.3,
        input_dropout=0.0,
        weight_decay=0.001,
        seed=42,
        verbose=True,
    ):
        """
        epochs          - maximum number of passes over the training data
        lr              - starting learning rate
        batch_size      - rows per weight update
        hidden          - sizes of the hidden layers, e.g. (512, 128)
        dropout         - fraction of neurons dropped after each hidden layer
        input_dropout   - fraction of input features zeroed on each training
                          batch (augmentation against memorising; 0 = off)
        weight_decay    - L2 penalty on the weights
        seed            - random seed for weights and shuffling
        verbose         - print progress every epoch
        """
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.hidden = tuple(hidden)
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.weight_decay = weight_decay
        self.seed = seed
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_model(self, input_dim, num_classes):
        """Stack Linear -> ReLU -> Dropout per hidden layer."""
        layers = []
        previous = input_dim

        if self.input_dropout > 0:
            layers.append(nn.Dropout(self.input_dropout))

        for size in self.hidden:
            layers.append(nn.Linear(previous, size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
            previous = size

        layers.append(nn.Linear(previous, num_classes))
        return nn.Sequential(*layers)

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Train the neural network.

        If X_val and y_val are given, validation accuracy is measured
        after every epoch and the weights from the best epoch are
        restored at the end of training.
        """

        X = np.asarray(X, dtype=np.float32)
        self.classes_, y = np.unique(y, return_inverse=True)

        use_validation = X_val is not None and y_val is not None
        if use_validation:
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val = np.asarray(y_val)

        torch.manual_seed(self.seed)

        input_dim = X.shape[1]
        num_classes = len(self.classes_)

        # Define the network
        self.model = self._build_model(input_dim, num_classes).to(self.device)

        # Convert NumPy arrays into PyTorch tensors
        dataset = TensorDataset(torch.from_numpy(X), torch.tensor(y, dtype=torch.long))

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        loss_fn = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # Smoothly lowers the learning rate towards 0 over training
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs
        )

        self.history = []  # training loss per epoch
        self.val_history = []  # validation accuracy per epoch
        self.best_epoch_ = None
        self.best_val_accuracy_ = None

        best_acc = -1.0
        best_state = None

        # Training loop
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            rows_seen = 0

            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()

                outputs = self.model(X_batch)
                loss = loss_fn(outputs, y_batch)

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(X_batch)
                rows_seen += len(X_batch)

            scheduler.step()

            avg_loss = total_loss / rows_seen
            self.history.append(avg_loss)

            message = f"Epoch {epoch + 1}/{self.epochs} - Loss: {avg_loss:.4f}"

            if use_validation:
                # predict() switches to eval mode; model.train() at the
                # top of the loop switches back for the next epoch
                val_acc = float((self.predict(X_val) == y_val).mean())
                self.val_history.append(val_acc)
                message += f" - Val accuracy: {val_acc:.4f}"

                # Remember the weights from the best epoch so far
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_state = copy.deepcopy(self.model.state_dict())
                    self.best_epoch_ = epoch + 1
                    self.best_val_accuracy_ = val_acc

            if self.verbose:
                print(message)

        # Go back to the weights from the best validation epoch
        if best_state is not None:
            self.model.load_state_dict(best_state)
            if self.verbose:
                print(
                    f"Restored best weights from epoch {self.best_epoch_} "
                    f"(val accuracy {self.best_val_accuracy_:.4f})"
                )

        return self

    def predict(self, X):
        """Predict class labels."""

        self.model.eval()
        X = np.asarray(X, dtype=np.float32)

        predictions = []

        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                batch = torch.from_numpy(X[start : start + self.batch_size]).to(
                    self.device
                )

                outputs = self.model(batch)
                predicted = outputs.argmax(dim=1)

                predictions.append(predicted.cpu().numpy())

        encoded = np.concatenate(predictions)
        return self.classes_[encoded]