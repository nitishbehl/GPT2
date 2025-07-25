import torch.nn as nn

class EarlyExitClassifier(nn.Module):
    def __init__(self, hidden_size=384, vocab_size=50257):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, vocab_size)
        )

    def forward(self, x):  # x shape: (B, T, hidden)
        return self.classifier(x)  # Output: (B, T, vocab_size)
