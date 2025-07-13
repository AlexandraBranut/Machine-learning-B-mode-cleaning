# CMB B-Mode ML Cleaning (v0.1.0)
CMB-HybridClean: end-to-end pipeline for CMB B-mode cleaning. • Simulate gnomonic patches (nside=512, fsky=0.1) with PySM3 • Analytic ILC baseline + U-Net (SF-NN) &amp; Conv+MLP (VAR-NN) hybrids • Training, evaluation (MSE scatter, power spectra, r-fitting) • Example notebooks &amp; scripts for full reproducibility

**Version 0.1.0** — initial release of main architectures and pipeline.

## Overview
Implements two neural-network based methods for cleaning foregrounds in CMB B-mode maps:

- **UNet2D** — a U-Net residual-cleaning architecture.  
- **WeightNet** — a small conv→FC network to refine ILC weights.

## Files
- `models.py`          — SF-NN & VAR-NN code  
- `data_pipeline.py`   — simulation & ILC utility functions  
- `train.py`           — quickstart training skeleton  
- `utils.py`           — save/load helpers  
- `requirements.txt`   — dependencies  

## Quickstart
```bash
pip install -r requirements.txt
python train.py
