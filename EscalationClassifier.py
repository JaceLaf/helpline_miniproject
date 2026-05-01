import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch
from sklearn.metrics import classification_report
import numpy as np

class EscalationClassifier(nn.Module):
    def __init__(self, raw=True):
        super().__init__()
        
        if raw:
            self.lstm = nn.LSTM(
                input_size=768,
                hidden_size=128,
                batch_first=True
            )
        else:
            self.lstm = nn.LSTM(
                input_size=768+10,
                hidden_size=128,
                batch_first=True
            )

        
        self.fc = nn.Linear(128, 3)

    def forward(self, x):
        out, _ = self.lstm(x)
        logits = self.fc(out)
        return logits

def collate(batch):
    Xs, ys = zip(*batch)

    X_padded = pad_sequence(Xs, batch_first=True)
    y_padded = pad_sequence(ys, batch_first=True, padding_value=-1)

    return X_padded, y_padded

def run_classifier(train_data, raw):
    model = EscalationClassifier(raw)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    loader = torch.utils.data.DataLoader(
        train_data, batch_size=8, shuffle=True, collate_fn=collate
    )

    for epoch in range(5):
        model.train()
        
        for X, y in loader:
            logits = model(X)

            logits_flat = logits.view(-1, 3)
            y_flat = y.view(-1)

            mask = y_flat != -1

            loss = criterion(
                logits_flat[mask],
                y_flat[mask]
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    return model


def eval_classifer(model, test_data):
    model.eval()

    loader = torch.utils.data.DataLoader(
        test_data, batch_size=8, shuffle=False, collate_fn=collate
    )

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in loader:
            logits = model(X)  # [B, T, 3]

            preds = torch.argmax(logits, dim=2)  # [B, T]

            # Flatten
            preds = preds.view(-1)
            y = y.view(-1)

            # Remove padding
            mask = y != -1
            preds = preds[mask]
            y = y[mask]

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(all_labels, all_preds))