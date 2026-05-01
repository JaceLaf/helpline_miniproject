import torch.nn as nn
import torch

class EscalationClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(768, 3)

    def forward(self, x):
        return self.linear(x)

model = EscalationClassifier()

def run_classifier(X_train, y_train):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    epochs = 5
    batch_size = 32

    for epoch in range(epochs):
        model.train()
        
        for i in range(0, len(X_train), batch_size):
            xb = X_train[i:i+batch_size]
            yb = y_train[i:i+batch_size]
            
            logits = model(xb)
            loss = criterion(logits, yb)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

def eval_classifer(X_test, y_test):
    model.eval()

    with torch.no_grad():
        logits = model(X_test)
        preds = torch.argmax(logits, dim=1)
        
        accuracy = (preds == y_test).float().mean()
        
    print("Accuracy:", accuracy.item())