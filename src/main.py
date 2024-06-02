import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Example amino acid encoding
amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
aa_dict = {aa: i for i, aa in enumerate(amino_acids)}

def encode_sequence(seq):
    return [aa_dict[aa] for aa in seq]

# Dataset class
class ProteinDataset(Dataset):
    def __init__(self, sequences, labels, max_len):
        self.sequences = [encode_sequence(seq) for seq in sequences]
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        # Padding/truncating
        seq = seq + [0] * (self.max_len - len(seq))
        label = label + [0] * (self.max_len - len(label))
        return torch.tensor(seq, dtype=torch.long), torch.tensor(label, dtype=torch.long)

# Example data
sequences = ['ACDEFGHIKLMNPQRSTVWY', 'LMNPQRSTVWYACDEFGHIK']  # Replace with actual sequences
labels = [[0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
          [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]  # Replace with actual labels
max_len = max(len(seq) for seq in sequences)
dataset = ProteinDataset(sequences, labels, max_len)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, hidden_dim, num_layers, output_dim, max_len):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        encoder_layers = nn.TransformerEncoderLayer(embed_dim, num_heads, hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(embed_dim, output_dim)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        x = self.transformer_encoder(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x

input_dim = len(amino_acids)
embed_dim = 64
num_heads = 8
hidden_dim = 256
num_layers = 4
output_dim = 2  # Signal or not signal
max_len = max_len

model = TransformerModel(input_dim, embed_dim, num_heads, hidden_dim, num_layers, output_dim, max_len)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, output_dim), targets.view(-1))
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Evaluate on a single example
with torch.no_grad():
    example_seq = torch.tensor(encode_sequence('ACDEFGHIKLMNPQRSTVWY'), dtype=torch.long).unsqueeze(0)
    prediction = model(example_seq)
    predicted_labels = torch.argmax(prediction, dim=2)
    print(predicted_labels)
