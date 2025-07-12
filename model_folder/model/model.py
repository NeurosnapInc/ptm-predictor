import torch
import torch.nn as nn

# Define an Adapter module
class Adapter(nn.Module):
  def __init__(self, input_dim, adapter_dim=64, dropout_prob=0.1):
    super().__init__()
    self.norm = nn.LayerNorm(input_dim)
    self.down_project = nn.Linear(input_dim, adapter_dim)
    self.activation = nn.GELU()
    self.up_project = nn.Linear(adapter_dim, input_dim)
    self.dropout = nn.Dropout(dropout_prob)

    # Learnable scaling factor for residual connection
    self.scale = nn.Parameter(torch.tensor(1e-3))

    # Initialize with small weights
    nn.init.normal_(self.down_project.weight, std=1e-3)
    nn.init.normal_(self.up_project.weight, std=1e-3)
    nn.init.zeros_(self.up_project.bias)

  def forward(self, x):
    # Normalize, project down, activate, project up, apply dropout
    x_norm = self.norm(x)
    down = self.down_project(x_norm)
    activated = self.activation(down)
    up = self.up_project(activated)
    dropped = self.dropout(up)

    # Scale and return residual connection
    return self.scale * dropped

# Create a PTM prediction model with adapters
class PTMAdapterModel(nn.Module):
  def __init__(self, esm_model, num_ptm_classes, adapter_dim=64):
    super().__init__()
    self.esm_model = esm_model

    # Freeze the base ESM model
    for param in self.esm_model.parameters():
      param.requires_grad = False

    # Get embedding dimension from the model
    self.embed_dim = self.esm_model.embed_dim

    # Add adapter after the ESM model
    self.adapter = Adapter(self.embed_dim, adapter_dim)

    # Add PTM classification layer
    self.ptm_classifier = nn.Linear(self.embed_dim, num_ptm_classes)

  def forward(self, tokens):
    # Extract features from ESM-2 (frozen)
    results = self.esm_model(tokens, repr_layers=[32], return_contacts=False)
    token_representations = results["representations"][32]

    # Pass through adapter with residual connection
    adapted_representations = token_representations + self.adapter(token_representations)

    # Apply classification head to each position
    ptm_logits = self.ptm_classifier(adapted_representations)

    return ptm_logits