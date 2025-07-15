import os, pickle, time
import numpy as np
import healpy as hp
import pysm3
from astropy import units as u

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from models import UNet2D, WeightNet
from utils import save_model
from data_pipeline import simulate_patch, compute_analytic_ilc, create_mask, nside, freqs

class SFResidualDataset(Dataset):
    def __init__(self, n_patches=450, r=0.0):
        self.data = [simulate_patch(i, r) for i in range(n_patches)]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        maps, _ = self.data[idx]  # shape (n_freq, 3, npix)
        ilcA = compute_analytic_ilc(
            {f: maps[freqs.index(f)] for f in freqs},
            {f: maps[freqs.index(f)] for f in freqs},
            freqs, create_mask(nside, 0.1)
        )
        ilcB = compute_analytic_ilc(
            {f: maps[freqs.index(f)] for f in freqs},
            {f: maps[freqs.index(f)] for f in freqs},
            freqs, create_mask(nside, 0.1)
        )
        target = ilcB - ilcA
        return torch.tensor(maps, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

def main():
    # Prepare dataset and loaders
    full_ds       = SFResidualDataset(n_patches=450, r=0.0)
    train_ds, test_ds = random_split(full_ds, [400, 50])
    train_loader  = DataLoader(train_ds, batch_size=4, shuffle=True)
    test_loader   = DataLoader(test_ds,  batch_size=4, shuffle=False)

    # Device and models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sf_model = UNet2D(in_ch=len(freqs), out_ch=len(freqs), wf=4).to(device)
    vr_model = WeightNet(nf=len(freqs)).to(device)

    # Optimizers & loss
    opt_sf  = optim.Adam(sf_model.parameters(), lr=1e-3)
    opt_vr  = optim.Adam(vr_model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Training loop
    epochs = 50
    for epoch in range(1, epochs + 1):
        sf_model.train(); vr_model.train()
        tot_sf = 0.0; tot_vr = 0.0

        for maps, target in train_loader:
            maps, target = maps.to(device), target.to(device)

            # SF‑NN step
            pred_sf = sf_model(maps)
            loss_sf = loss_fn(pred_sf, target)
            opt_sf.zero_grad(); loss_sf.backward(); opt_sf.step()
            tot_sf += loss_sf.item()

            # VAR‑NN step (residual correction)
            with torch.no_grad():
                res_sf = pred_sf
            pred_vr = vr_model(res_sf)
            loss_vr = loss_fn(pred_vr, target - res_sf)
            opt_vr.zero_grad(); loss_vr.backward(); opt_vr.step()
            tot_vr += loss_vr.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: SF-loss={tot_sf/len(train_loader):.4f}, "
                  f"VR-loss={tot_vr/len(train_loader):.4f}")

    # Save final models
    save_model(sf_model, "sf_nn_v0.pt")
    save_model(vr_model, "var_nn_v0.pt")

if __name__ == "__main__":
    main()
