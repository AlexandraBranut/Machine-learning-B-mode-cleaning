import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader

from models import UNet2D, WeightNet
from utils import load_model
from data_pipeline import nside, freqs, simulate_patch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) Load & prepare 30 random-r test patches
test_D      = np.load("test_patches/test_D.npy")        # (30, 6, 128, 128)
test_w_true = np.load("test_patches/test_w_true.npy")   # (30, 6)

D      = torch.from_numpy(test_D.astype(np.float32))
W_true = torch.from_numpy(test_w_true.astype(np.float32))

# analytic ILC residual stamps
stamp_ilc = (W_true.unsqueeze(-1).unsqueeze(-1) * D).sum(dim=1, keepdim=True)

loader_sf = DataLoader(TensorDataset(D, stamp_ilc), batch_size=1, shuffle=False)
loader_vr = DataLoader(TensorDataset(D, W_true),   batch_size=1, shuffle=False)

# 2) Load trained models
sf_model  = UNet2D(in_ch=D.shape[1], out_ch=1, wf=4).to(device)
sf_model.load_state_dict(torch.load("sf_nn_v0.pt", map_location=device))
sf_model.eval()

var_model = WeightNet(nf=D.shape[1]).to(device)
var_model.load_state_dict(torch.load("var_nn_v0.pt", map_location=device))
var_model.eval()

# 3) Per-patch MSE scatter (ILC vs SF-NN & VAR-NN)
mse_ilc, mse_sf, mse_var = [], [], []
for (x_sf, y_sf), (D_vr, w_true) in zip(loader_sf, loader_vr):
    x_sf, y_sf   = x_sf.to(device), y_sf.to(device)
    D_vr, w_true = D_vr.to(device), w_true.to(device)

    # ILC residual MSE
    mse_ilc.append((y_sf**2).mean().item())

    # SF-NN residual MSE
    with torch.no_grad():
        y_hat_sf = sf_model(x_sf)
    mse_sf.append(((y_hat_sf - y_sf)**2).mean().item())

    # VAR-NN residual MSE
    with torch.no_grad():
        w_pred = var_model(D_vr)
    stamp_ilc_var = (w_true.unsqueeze(-1).unsqueeze(-1) * D_vr).sum(dim=1, keepdim=True)
    stamp_var      = (w_pred.unsqueeze(-1).unsqueeze(-1)   * D_vr).sum(dim=1, keepdim=True)
    mse_var.append(((stamp_var - stamp_ilc_var)**2).mean().item())

plt.figure()
plt.scatter(mse_ilc, mse_sf, alpha=0.7, label="SF-NN vs ILC")
plt.scatter(mse_ilc, mse_var, alpha=0.7, label="VAR-NN vs ILC")
mn, mx = min(mse_ilc + mse_sf + mse_var), max(mse_ilc + mse_sf + mse_var)
plt.plot([mn, mx], [mn, mx], 'k--', color='gray')
plt.xlabel("MSE of ILC residual")
plt.ylabel("MSE of ML residual")
plt.legend()
plt.title("Per-patch MSE Scatter (30 random-r patches)")
plt.tight_layout()
plt.show()

# 4) Average auto-spectrum (median ± IQR) + theory BB(r=0.001, 0.005, 0.010)
def compute_spectrum(stamp):
    proj   = hp.gnomview(stamp, return_projected_map=True, no_plot=True)
    fullhp = hp.gnom2map(proj, nside, verbose=False)
    return hp.anafast(fullhp)

cls_ilc, cls_sf, cls_var = [], [], []
for (x_sf, y_sf), (D_vr, w_true) in zip(loader_sf, loader_vr):
    with torch.no_grad():
        y_hat_sf = sf_model(x_sf)
        w_pred   = var_model(D_vr)
    ilc_np = y_sf.cpu().numpy()[0,0]
    sf_np  = y_hat_sf.cpu().numpy()[0,0]
    var_np = (w_pred.unsqueeze(-1).unsqueeze(-1) * D_vr).sum(dim=1)[0].cpu().numpy()
    cls_ilc.append(compute_spectrum(ilc_np))
    cls_sf.append(compute_spectrum(sf_np))
    cls_var.append(compute_spectrum(var_np))

ells = np.arange(len(cls_ilc[0]))
plt.figure(figsize=(8,5))
for arr, label, col in zip(
    [cls_ilc, cls_sf, cls_var],
    ["Analytic ILC", "SF-NN", "VAR-NN"],
    ["C1", "C2", "C3"]
):
    med = np.median(arr, axis=0)
    p25 = np.percentile(arr, 25, axis=0)
    p75 = np.percentile(arr, 75, axis=0)
    plt.plot(ells, med, label=label, color=col)
    plt.fill_between(ells, p25, p75, color=col, alpha=0.3)

# load CAMB BB(r=1) template
bb_r1 = np.load("templates/bb_template_r1.npy") * 1e12  # μK²
# beam & pixel window
bl      = hp.gauss_beam(np.radians(30/60.), lmax=ells[-1])
pixw    = hp.pixwin(nside)[:ells[-1]+1]
window2 = bl**2 * pixw**2

for r in [0.001, 0.005, 0.010]:
    plt.plot(ells, r * bb_r1 * window2, '--k', label=f"Theory BB r={r:.3f}")

plt.xscale("log"); plt.yscale("log")
plt.xlim(2,500)
plt.xlabel(r"$\ell$")
plt.ylabel(r"$C_\ell^{BB}\ [\mu{\rm K}^2]$")
plt.grid(which="both", axis="y", ls="--", alpha=0.3)
plt.legend(loc="lower left", ncol=2)
plt.tight_layout()
plt.show()

# 5) Matched-filter Δr statistics over 30 full-sky patches (ℓ=2–ℓmax)
cls_ilc_full = np.load("results/cls_ilc_full.npy")
cls_sf_full  = np.load("results/cls_sfh_full.npy")
cls_var_full = np.load("results/cls_vr_full.npy")

mask = np.arange(bb_r1.size) >= 2
bb_win = bb_r1[mask] * window2[mask]
norm   = bb_win.dot(bb_win)

dr_ilc = cls_ilc_full[:, mask].dot(bb_win) / norm
dr_sf  = cls_sf_full[:,  mask].dot(bb_win) / norm
dr_var = cls_var_full[:,mask].dot(bb_win) / norm

def stats(a): return a.mean(), a.std(), a.min(), a.max()
s_ilc, s_sf, s_var = stats(dr_ilc), stats(dr_sf), stats(dr_var)

print("Table 3: Uncalibrated Δr")
print(f"ILC:    ⟨Δr⟩={s_ilc[0]:.4e}, σ={s_ilc[1]:.4e}, min={s_ilc[2]:.4e}, max={s_ilc[3]:.4e}")
print(f"SF-NN:  ⟨Δr⟩={s_sf[0]:.4e}, σ={s_sf[1]:.4e}, min={s_sf[2]:.4e}, max={s_sf[3]:.4e}")
print(f"VAR-NN: ⟨Δr⟩={s_var[0]:.4e}, σ={s_var[1]:.4e}, min={s_var[2]:.4e}, max={s_var[3]:.4e}")

# calibrated
dr_ilc_cal = dr_ilc - s_ilc[0]
dr_sf_cal  = dr_sf  - s_sf[0]
dr_var_cal = dr_var - s_var[0]
print("\nTable 4: Calibrated Δr")
for name, arr in [("ILC-cal", dr_ilc_cal), ("SF-NN-cal", dr_sf_cal), ("VAR-NN-cal", dr_var_cal)]:
    m,s,mn,mx = stats(arr)
    print(f"{name}: mean≈{m:.4e}, σ={s:.4e}, min={mn:.4e}, max={mx:.4e}")

# 6) Map visualization: ILC vs SF-NN vs VAR-NN stamps
mse_list = [(y**2).mean().item() for (x,y),_ in zip(loader_sf, loader_vr)]
best = int(np.argmin(mse_list))

st_ilc = loader_sf.dataset[best][1].numpy()[0,0]
maps_b,_ = loader_sf.dataset[best]
inp = maps_b.unsqueeze(0).to(device)
with torch.no_grad():
    st_sf = sf_model(inp).cpu().numpy()[0,0]
D_b, w_b = loader_vr.dataset[best]
with torch.no_grad():
    w_p = var_model(D_b.unsqueeze(0).to(device))
st_ilc_var = (w_b.unsqueeze(-1).unsqueeze(-1)*D_b).sum(dim=1).numpy()
st_var     = (w_p.cpu().numpy()[0,:,None,None]*D_b.numpy()).sum(axis=0)

vmin,vmax = st_ilc.min(), st_ilc.max()
vmin = min(vmin, st_sf.min(), st_var.min())
vmax = max(vmax, st_sf.max(), st_var.max())

fig, axes = plt.subplots(1,3,figsize=(12,4))
for ax, data, title in zip(
    axes, [st_ilc, st_sf, st_var],
    ["ILC Residual", "SF-NN Residual", "VAR-NN Residual"]
):
    im = ax.imshow(data, origin="lower", vmin=vmin, vmax=vmax)
    ax.set_title(title); ax.axis("off")
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
cbar.set_label("μK")
plt.suptitle(f"Easiest Patch (idx={best})", y=0.98)
plt.tight_layout()
plt.show()

# 7) Residual Analysis
cls_raw    = np.load("results/cls_ilc_raw.npy")
cls_smooth = np.load("results/cls_ilc_smooth.npy")

raw_m    = np.mean(cls_raw, axis=0)
smooth_m = np.mean(cls_smooth, axis=0)
abs_sf   = raw_m - mean_sf
abs_var  = raw_m - mean_var
frac_sf  = abs_sf / raw_m
frac_var = abs_var / raw_m

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,4))
ax1.loglog(ells, abs_sf,   label="SF-NN")
ax1.loglog(ells, abs_var,  label="VAR-NN")
ax1.loglog(ells, raw_m,    ":", color="gray", label="ILC raw")
ax1.loglog(ells, smooth_m, "--", color="black", label="ILC smooth")
ax1.set(xlabel=r"$\ell$", ylabel=r"$\Delta C_\ell\ [\mu{\rm K}^2]$", title="Absolute residual power")
ax1.legend()

ax2.semilogx(ells, frac_sf, label="SF-NN")
ax2.semilogx(ells, frac_var, label="VAR-NN")
ax2.axhline(1.0, color="green", linestyle="-")
ax2.set(xlabel=r"$\ell$", ylabel="Fraction of raw power removed", title="Fractional residual suppression")
ax2.legend()

plt.tight_layout()
plt.show()
