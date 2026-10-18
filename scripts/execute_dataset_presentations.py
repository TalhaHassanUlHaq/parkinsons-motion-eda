"""Execute the three instructor notebooks locally with generated motion media."""

from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "executed"
OUT.mkdir(parents=True, exist_ok=True)

for slug in ("mmu", "bmclab", "pose_estimation"):
    source = ROOT / "notebooks" / f"eda_{slug}_presentation.ipynb"
    nb = nbformat.read(source, as_version=4)
    NotebookClient(
        nb,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    ).execute()
    target = OUT / source.name
    nbformat.write(nb, target)
    print(f"Executed {target.relative_to(ROOT)}")
