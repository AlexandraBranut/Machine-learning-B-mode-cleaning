import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from models import UNet2D, WeightNet
from data_pipeline import simulate_patch, nside, freqs

# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 50
batch_size_sf = 4
batch_size_vr = 8
r_fixed = 0.0  # baseline

# 1) Generate & split dataset
maps, true_b = zip(*(simulate_patch(i, r_fixed) for i in range(450)))  # 450 patches

# 2) Models & optimizers
sf_model = UNet2D(in_ch=len(freqs), out_ch=len(freqs), wf=4).to(device)
vr_model = WeightNet(nf=len(freqs)).to(device)
opt_sf = optim.Adam(sf_model.parameters(), lr=1e-3)
opt_vr = optim.Adam(vr_model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# 3) Training loops (skeleton)
for epoch in range(1, epochs + 1):
    sf_model.train()
    # ... loop over sf DataLoader ...
    #     pred = sf_model(x)
    #     loss = loss_fn(pred, y)
    #     opt_sf.zero_grad(); loss.backward(); opt_sf.step()
    #
    vr_model.train()
    # ... similarly for vr_model ...

    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{epochs} done")

# 4) Save checkpoints
torch.save(sf_model.state_dict(), "sf_nn_v0.pt")
torch.save(vr_model.state_dict(), "var_nn_v0.pt")
