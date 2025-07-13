import torch

def save_model(model, filename):
    """Save PyTorch model state dict."""
    torch.save(model.state_dict(), filename)

def load_model(cls, filename):
    """Load model from state dict."""
    model = cls()
    model.load_state_dict(torch.load(filename))
    return model
