import torch

# Check for Mac GPU acceleration (MPS)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Mac GPU (MPS)")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Move your tensor to the selected device
x = torch.randn(5, 5).to(device)
