import torch
from torch import nn
from torch.optim import AdamW
from point_transformer_pytorch import PointTransformerLayer


# Define a Multi-layer Perceptron (MLP)
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.layers(x)


# Define the model
class MyModel(nn.Module):
    def __init__(self, dim=128, pos_mlp_hidden_dim=64, attn_mlp_hidden_mult=4, num_neighbors=16,
                 num_classes=1):
        super().__init__()

        # Point Transformer layers
        self.point_transformer1 = PointTransformerLayer(
            dim=dim,
            pos_mlp_hidden_dim=pos_mlp_hidden_dim,
            attn_mlp_hidden_mult=attn_mlp_hidden_mult,
            num_neighbors=num_neighbors,
        )
        self.point_transformer2 = PointTransformerLayer(
            dim=dim,
            pos_mlp_hidden_dim=pos_mlp_hidden_dim,
            attn_mlp_hidden_mult=attn_mlp_hidden_mult,
            num_neighbors=num_neighbors,
        )

        # Multi-layer Perceptron
        self.mlp = MLP(dim, dim // 2, num_classes)

    def forward(self, feats, pos, mask):
        # Pass through Point Transformers
        x = self.point_transformer1(feats, pos, mask)
        x = self.point_transformer2(x, pos, mask)

        # Aggregate across points
        x = x.mean(dim=1)

        # Pass through MLP
        out = self.mlp(x)

        return out


# Define the training step function
def train_step(model, optimizer, criterion, feats, pos, mask, labels):
    model.train()
    optimizer.zero_grad()

    out = model(feats, pos, mask)
    loss = criterion(out, labels)

    loss.backward()
    optimizer.step()

    return loss.item()


# Function to prepare data
def prepare_data():
    # Prepare your actual data here (replace the random data)
    feats = torch.randn(1, 16, 3)  # replace with your actual features (including the binary feature)
    pos = torch.randn(1, 16, 3)  # replace with your actual point positions
    mask = torch.ones(1, 16).bool()  # replace with your actual mask
    labels = torch.randn(1, 1)  # replace with your actual labels
    return feats, pos, mask, labels


# Function to initialize the model, optimizer and criterion
def initialize_model():
    # Instantiate the model
    model = MyModel()

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters())

    return model, optimizer, criterion


def main():
    # Initialize model, optimizer and criterion
    model, optimizer, criterion = initialize_model()

    # Prepare data
    feats, pos, mask, labels = prepare_data()

    # Run a training step
    loss = train_step(model, optimizer, criterion, feats, pos, mask, labels)
    print('Training loss:', loss)


if __name__ == "__main__":
    main()
