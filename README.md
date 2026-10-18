# Parkinson motion EDA and reconstruction

Three reproducible exploratory notebooks that generated nine animations from authentic Parkinson movement datasets. No classifier is trained. This public code repository excludes raw clinical records, rendered videos, private Kaggle inputs, credentials, and third-party reference notebooks.

## Start here

- `notebooks/eda_mmu.ipynb`, `notebooks/eda_bmclab.ipynb`, `notebooks/eda_parkinsons_pose.ipynb`: source notebooks.
- `outputs/executed/`: locally executed notebooks.
- `outputs/videos/{mmu,bmclab,pose_estimation}/`: three MP4s per dataset.
- `outputs/figures/`: one PNG preview per video plus notebook charts.
- `outputs/tables/simulation_manifest.csv`: source ID and frame-range traceability.
- `outputs/tables/*_distribution_profile.csv`: numeric skewness and distribution scope for each dataset.
- `outputs/figures/*_score_categories.png`: per-dataset bar charts for display categories 1–4, with source score 0 retained where present.
- `outputs/reports/dataset_comparison.md`: verified findings, limitations and recommendations.
- `docs/dataset_comparison.md`: aggregate comparison report included in the public repository.
- `docs/figures/`: public aggregate score-category bar charts for the three datasets.
- `outputs/reports/video_verification.json`: decoded frame counts and durations.

The original large C3D archive and downloaded pose/MMU source data live in ignored `raw/` files or root archives. They are excluded from repository commits. Python requirements are in `requirements.txt`; local MP4 encoding uses `ffmpeg` or the `imageio-ffmpeg` binary. Execute locally with `python scripts/execute_local.py bmclab` (also `mmu` or `parkinsons_pose`). Rebuild notebooks with `python scripts/build_notebooks.py`. Recreate the original BMCLab inventory with `python scripts/prepare_inventory.py`. Verify videos with `python scripts/verify_outputs.py`.

## Kaggle

`scripts/prepare_kaggle.py` stages a minimized BMCLab private input and notebook metadata when the source data are available locally. The MMU notebook uses the public `teeconnie/mmu-visual-based-parkinsons-disease-dataset`; the clinical pose notebook uses the original author's public `limi44/parkinsons-visionbased-pose-estimation-dataset`. All three notebooks completed privately on Kaggle; their downloaded run artifacts are under the ignored `outputs/kaggle/` directory in the working copy. The BMCLab sample was uploaded after explicit user authorization. Verify downloaded videos with `python scripts/verify_kaggle_outputs.py`.

The notebooks use the supplied CARE-PD notebook as a presentation reference for overview tables, duration plots, label distributions, and written interpretations. Skewness is calculated only for numeric distributions; categorical labels are summarized with class counts and imbalance ratios. The MMU output FPS is illustrative because the source JSON lacks video FPS. BMCLab C3D Y is the vertical source axis. The clinical pose archive contains seated assessment trajectories and no original images. See the comparison report for evidence and source licenses.
