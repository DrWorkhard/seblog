import random
import torch
from torch import nn

# Vocabulary and helper functions
CHARS = ['<pad>'] + list('0123456789+') + ['=']
stoi = {ch: i for i, ch in enumerate(CHARS)}
itos = {i: ch for ch, i in stoi.items()}
PAD_IDX = stoi['<pad>']
VOCAB_SIZE = len(CHARS)
MAX_LEN = 9  # Max tokens in "99+99=198"

def encode(seq: str):
    return [stoi[c] for c in seq]


def decode(tokens):
    return ''.join(itos[t] for t in tokens if t != PAD_IDX)


class AdditionDataset(torch.utils.data.Dataset):
    """Generate samples of simple addition expressions."""

    def __init__(self, num_samples: int = 5000):
        super().__init__()
        self.samples = []
        for _ in range(num_samples):
            a = random.randint(0, 99)
            b = random.randint(0, 99)
            seq = list(str(a) + '+' + str(b) + '=' + str(a + b))
            ids = encode(''.join(seq))
            ids += [PAD_IDX] * (MAX_LEN - len(ids))
            self.samples.append(torch.tensor(ids, dtype=torch.long))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq = self.samples[idx]
        return seq[:-1], seq[1:]


def collate(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.stack(ys)


class TransformerModel(nn.Module):
    def __init__(self, d_model: int = 128, nhead: int = 8, num_layers: int = 2, dim_feedforward: int = 512):
        super().__init__()
        self.token_emb = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, MAX_LEN - 1, d_model))
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                           dim_feedforward=dim_feedforward, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.fc = nn.Linear(d_model, VOCAB_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq]
        x = self.token_emb(x) + self.pos_emb[:, :x.size(1)]
        x = self.encoder(x)
        return self.fc(x)


def train(model: nn.Module, epochs: int = 10, batch_size: int = 32, lr: float = 1e-3, device: str | None = None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = AdditionDataset()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: loss {total_loss/len(loader):.4f}")


def generate(model: nn.Module, a: int, b: int, device: str | None = None) -> str:
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    seq = list(str(a) + '+' + str(b) + '=')
    ids = encode(''.join(seq))
    with torch.no_grad():
        for _ in range(MAX_LEN - len(ids)):
            x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
            logits = model(x)
            next_id = logits[0, -1].argmax(-1).item()
            ids.append(next_id)
            if next_id == PAD_IDX:
                break
    return decode(ids)


if __name__ == "__main__":
    model = TransformerModel()
    train(model, epochs=50)
    print(generate(model, 12, 27))
