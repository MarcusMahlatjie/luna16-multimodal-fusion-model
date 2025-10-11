# Marcus Mahlatjie DST481 2025 - AI POWERED DISEASE DIAGNOSIS 
## Submitted to Belgium Campus iTversity as part of Research and Examination for Dissertation 481
> A multimodal deep learning framework that fuses 3-D CT imaging and structured clinical data for lung cancer risk prediction, built on the LUNA16 dataset.

---

## Overview
**This project** is a research-grade project demonstrating how multimodal learning can improve lung cancer detection.  
It integrates:
- **3-D Convolutional Neural Networks (CNNs)** on CT scan patches (from LUNA16), and  
- **Tabular Multilayer Perceptrons (MLPs)** on structured clinical profiles (synthetic risk factors).
- Due to size limitation the model files have been gitignored - run the notebooks to reproduce the pytorch model files

The framework outputs patch-level malignancy predictions, which are aggregated into **scan-level risk scores**.

---

## Architecture

                ┌──────────────────────────┐
                │   64³ CT Patch (HU-Norm) │
                └──────────┬───────────────┘
                           │
                     [3-D ResNet-18]
                           │
                     512-D Imaging Embedding
                           │
                           │        ┌───────────────────────┐
                           │        │ Clinical Profile (8f) │
                           │        └──────────┬────────────┘
                           │              [Tabular MLP]
                           │                  64-D
                           ├──────────────┬───────────────┤
                           │          [Concatenate]       │
                           └──────────────┴───────────────┘
                                   │
                             [Fusion Head]
                                   │
                             Final Probability

---

## Datasets

| Dataset | Purpose | Contents |
|----------|----------|-----------|
| **LUNA16 (CT scans)** | Primary imaging data | 888 annotated thoracic CT volumes with voxel spacing metadata |
| **patches_64mm** | Positive samples | 1186 × 64×64×64 voxel nodule-centered patches |
| **bg_patches_64mm** | Real negative samples | Random lung-region cubes ≥15 mm from any annotated nodule |
| **synthetic_profiles.csv** | Tabular inputs | Simulated clinical features: `age`, `sex`, `smoking_status`, `pack_years`, `years_since_quit`, `family_history`, `copd_dx` |

All preprocessing was performed with **SimpleITK**, **NumPy**, and **pandas**, and stored as `.npy` cubes and `.csv` indices.

---

## Preprocessing Workflow

1. **Resample** all CT volumes to 1 mm³ voxels.  
2. **Window & normalize** HU values to [-1000, 400] → [0, 1].  
3. **Extract 64³ patches** centered on annotated nodules.  
4. **Mine real negatives**: random lung-region cubes ≥15 mm from any nodule.  
5. **Generate clinical profiles** (synthetic tabular data).  
6. **Persist** as:
   ```
   patches_64mm/
   bg_patches_64mm/
   patch_index.csv
   bg_index.csv
   synthetic_profiles.csv
   ```

---

## Model Components

### 1️⃣ Imaging Branch — 3-D CNN
```python
from torchvision.models.video import r3d_18
cnn = r3d_18(weights=None)
cnn.stem[0] = nn.Conv3d(1,64,7,2,3,bias=False)
cnn.fc = nn.Identity()  # outputs 512-D embedding
```

### 2️⃣ Tabular Branch — MLP
```python
class TabMLP(nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, out_dim), nn.ReLU()
        )
```

### 3️⃣ Fusion Head
```python
self.head = nn.Sequential(
    nn.Linear(512+64, 128),
    nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(128, 1)
)
```

### 4️⃣ Loss & Optimization
- **Loss:** `BCEWithLogitsLoss()`  
- **Optimizer:** `AdamW(lr=3e-4, weight_decay=1e-4)`  
- **Scheduler:** optional cosine annealing  
- **Metrics:** AUROC (patch-level & scan-level)

---

## Training Procedure

| Phase | Description |
|-------|--------------|
| **Baseline CNN** | Train 3-D ResNet-18 on positive & real negative patches. |
| **Fusion Model** | Combine frozen CNN + MLP; then fine-tune end-to-end. |
| **Validation Split** | 80 / 20 grouped by `seriesuid` (no scan leakage). |
| **Augmentation** | Random axis flips & rotations for positives. |
| **Batch Size** | 16 (fits on a 16 GB GPU). |
| **Epochs** | 8–10 typical. |

Saved artifacts:
```
fusion_cnn_tab_realnegs.pt
cnn_baseline_realnegs.pt
preproc.joblib
tab_feature_names.json
run_meta.json
```

---

## Evaluation

| Metric | Description | Typical Result (real negatives) |
|---------|--------------|---------------------------------|
| **Patch-level AUROC (baseline)** | Imaging only | 0.80 – 0.85 |
| **Patch-level AUROC (fusion)** | Imaging + tabular | 0.83 – 0.87 |
| **Fusion lift** | Δ AUROC | +0.03 – 0.05 |
| **Scan-level AUROC** | Max-aggregated patches | 0.86 – 0.89 |
| **At 90% sensitivity** | Specificity ≈ 0.75 – 0.80, F1 ≈ 0.82 |

Artifacts saved automatically:
- `fusion_roc_realnegs.png`
- `fusion_pr_realnegs.png`
- `metrics_summary_realnegs.json`
- `tabular_feature_importance.csv`

---

## Tabular Feature Importance (Top 5)

| Rank | Feature | Importance |
|------|----------|-------------|
| 1 | `family_history_1` | High |
| 2 | `copd_dx_1` | High |
| 3 | `smoking_status_never` | Medium |
| 4 | `sex_F` | Medium |
| 5 | `years_since_quit` | Moderate |

(Values derived from the first linear layer of the TabMLP.)

---

## Reproducibility

| Component | File |
|------------|------|
| Model weights | `fusion_cnn_tab_realnegs.pt` |
| Tabular preprocessor | `preproc.joblib` |
| Feature names | `tab_feature_names.json` |
| Run metadata | `run_meta.json` |

Set random seeds for full determinism:
```python
import torch, random, numpy as np, os
SEED=42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
```

---

## Example Inference

```python
import joblib, torch, numpy as np, pandas as pd
from torchvision.models.video import r3d_18
import torch.nn as nn

# load model + preproc
state = torch.load("fusion_cnn_tab_realnegs.pt", map_location="cpu")
model = FusionNet(tab_in_dim=len(json.load(open("tab_feature_names.json"))))
model.load_state_dict(state); model.eval()

preproc = joblib.load("preproc.joblib")

# prepare inputs
patch = np.load("patches_64mm/sample.npy")  # (64,64,64)
x_img  = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0)
profile = pd.DataFrame([{
    "age":65,"sex":"M","smoking_status":"former",
    "pack_years":30,"years_since_quit":10,
    "family_history":1,"copd_dx":0
}])
x_tab = torch.tensor(preproc.transform(profile), dtype=torch.float32)

# inference
with torch.no_grad():
    prob = torch.sigmoid(model(x_img, x_tab)).item()
print(f"Predicted malignancy probability: {prob:.3f}")
```

---

## Key Learnings
- Adding structured clinical features provides **consistent AUROC lift** on realistic negative samples.  
- Mining **true background patches** (inside lung, far from nodules) is crucial for valid evaluation.  
- Synthetic profiles can emulate real-world clinical risk factors for research reproducibility.

---

## Limitations
- Clinical features are **synthetic**; real patient data may change learned relationships.  
- Patch-level labels approximate nodule presence, not histopathological malignancy.  
- Background mining relies on HU/mask heuristics for lung segmentation.

---

## References
- LUNA16: https://luna16.grand-challenge.org  
- Aerts et al., *Radiomics: The bridge between medical imaging and personalized medicine*, Nat. Commun. 2014.  
- Hara et al., *Learning spatio-temporal features with 3D residual networks for action recognition*, ICCV 2017.  
- SimpleITK, PyTorch, Scikit-learn (toolchain)

---

## Citation
If you use this repository, please cite:
```
Mahlatjie, M. (2025). LunaFusion: A Multimodal Deep Learning Framework for AI-Powered Lung Cancer Diagnosis.  
Honours Research Project, [University Name].
```

---

## Repo Structure

```
luna-16-multimodal-model-fusion/
│
├── notebooks/
|   ├── 01_load_data.ipynb          
│   ├── 02_preprocess.ipynb                 # patch extraction
│   ├── 03_cnn_baseline_model.ipynb         # tabular data generation
│   ├── 04_multimodal-fusion.ipynb          # imaging-only training
│   ├── 05_negative_mining.ipynb            # real background mining
│   ├── 06_cnn_baseline_realnegs.ipynb      # multimodal training
├── resuls/
│   ├── fusion_cnn_tab_realnegs.pt
│   ├── preproc.joblib
│   ├── fusion_roc_realnegs.png
│   └── metrics_summary_realnegs.json
├── model/
│   ├── cnn_baseline.pt
│   ├── fusion_cnn_tab_realnegs.pt
│   └── fusion_cnn_tab.pt
│
└── README.md
```

