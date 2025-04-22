import torch
import torch.nn as nn
from esm import pretrained

# Load pretrained ESM-2 model
model_name = "esm2_t33_650M_UR50D"
esm_model, alphabet = pretrained.load_model_and_alphabet(model_name)
batch_converter = alphabet.get_batch_converter()


# Define an Adapter module
class Adapter(nn.Module):
  def __init__(self, input_dim, adapter_dim=64):
    super().__init__()
    self.down_project = nn.Linear(input_dim, adapter_dim)
    self.activation = nn.GELU()
    self.up_project = nn.Linear(adapter_dim, input_dim)

    # Initialize with small weights
    nn.init.normal_(self.down_project.weight, std=1e-3)
    nn.init.normal_(self.up_project.weight, std=1e-3)
    nn.init.zeros_(self.up_project.bias)

  def forward(self, x):
    # Down-project, activate, up-project
    down = self.down_project(x)
    activated = self.activation(down)
    up = self.up_project(activated)
    return up


# Create a PTM prediction model with adapters
class PTMAdapterModel(nn.Module):
  def __init__(self, esm_model, num_ptm_classes, adapter_dim=64):
    super().__init__()
    self.esm_model = esm_model

    # Freeze the base ESM model
    for param in self.esm_model.parameters():
      param.requires_grad = False

    # Get embedding dimension from the model
    self.embed_dim = self.esm_model.args.embed_dim

    # Add adapter after the ESM model
    self.adapter = Adapter(self.embed_dim, adapter_dim)

    # Add PTM classification layer
    self.ptm_classifier = nn.Linear(self.embed_dim, num_ptm_classes)

  def forward(self, tokens):
    # Extract features from ESM-2 (frozen)
    results = self.esm_model(tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33]

    # Pass through adapter
    adapted_representations = token_representations + self.adapter(token_representations)

    # Apply classification head to each position
    ptm_logits = self.ptm_classifier(adapted_representations)

    return ptm_logits


# Example usage
num_ptm_classes = 10  # No modification + 9 different PTM types
ptm_model = PTMAdapterModel(esm_model, num_ptm_classes)

# Only train the adapters and classification head
optimizer = torch.optim.Adam([{"params": ptm_model.adapter.parameters()}, {"params": ptm_model.ptm_classifier.parameters()}], lr=1e-4)
