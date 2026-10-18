"""Inventory the original archive without expanding 1.3 GB of motion files."""
import csv
from pathlib import Path
import re
from zipfile import ZipFile

root=Path(__file__).resolve().parents[1]
rows=[]
with ZipFile(root/"C3Dfiles.zip") as z:
    for info in z.infolist():
        if not info.filename.lower().endswith(".c3d"): continue
        name=Path(info.filename).name
        subject=re.search(r"SUB\d+",name,re.I)
        med=re.search(r"_(on|off)_",name,re.I)
        rows.append({"archive_path":info.filename,"file_name":name,"subject":subject.group(0) if subject else "Unknown",
                     "medication":med.group(1).upper() if med else "Unknown",
                     "trial_type":"walk" if "_walk_" in name.lower() else "static" if "_static_" in name.lower() else "other",
                     "uncompressed_bytes":info.file_size})
dest=root/"outputs/tables/bmclab_inventory.csv";dest.parent.mkdir(parents=True,exist_ok=True)
with dest.open("w",newline="",encoding="utf8") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(f"Wrote {len(rows)} C3D entries to {dest}")
