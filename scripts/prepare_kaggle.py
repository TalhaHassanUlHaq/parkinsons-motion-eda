"""Stage small private Kaggle inputs and three notebook source folders."""
import json
import csv
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.pipeline import read_simple_xlsx
USER="talhahassanulhaq"  # Verified with `kaggle datasets list --mine`.
KAGGLE=ROOT/"kaggle"

def metadata(folder,slug,title,license_name):
    folder.mkdir(parents=True,exist_ok=True)
    obj={"title":title,"id":f"{USER}/{slug}","licenses":[{"name":license_name}]}
    (folder/"dataset-metadata.json").write_text(json.dumps(obj,indent=2),encoding="utf8")

bm=KAGGLE/"bmclab_input"
metadata(bm,"parkinsons-bmclab-c3d-eda-sample-20261018","BMCLab original C3D EDA sample", "CC-BY-4.0")
(bm/"SUB01_off_walk_1.c3d").unlink(missing_ok=True)  # superseded preview sample
for filename in ["SUB01_off_walk_14.c3d","SUB01_on_walk_19.c3d","SUB02_off_walk_6.c3d"]:
    shutil.copy2(ROOT/"raw/bmclab"/filename,bm/filename)
(bm/"PDGinfo.xlsx").unlink(missing_ok=True)
clinical_columns=["ID","FoG group","ON - UPDRS-III","OFF - UPDRS-III"]
clinical_rows=[{col:row.get(col,"") for col in clinical_columns}
               for row in read_simple_xlsx(ROOT/"PDGinfo.xlsx")
               if row.get("ID") in {"SUB01","SUB02"}]
with (bm/"bmclab_selected_metadata.csv").open("w",newline="",encoding="utf8") as f:
    writer=csv.DictWriter(f,fieldnames=clinical_columns);writer.writeheader();writer.writerows(clinical_rows)
shutil.copy2(ROOT/"outputs/tables/bmclab_inventory.csv",bm/"bmclab_inventory.csv")

settings={
 "mmu":("eda_mmu.ipynb","mmu-parkinson-motion-eda","MMU Parkinson motion EDA","teeconnie/mmu-visual-based-parkinsons-disease-dataset"),
 "bmclab":("eda_bmclab.ipynb","bmclab-original-c3d-gait-eda","BMCLab original C3D gait EDA",f"{USER}/parkinsons-bmclab-c3d-eda-sample-20261018"),
 "pose_estimation":("eda_parkinsons_pose.ipynb","parkinson-clinical-pose-eda","Parkinson clinical pose EDA","limi44/parkinsons-visionbased-pose-estimation-dataset")
}
for folder,(notebook,slug,title,source) in settings.items():
    dest=KAGGLE/folder;dest.mkdir(parents=True,exist_ok=True)
    shutil.copy2(ROOT/"notebooks"/notebook,dest/notebook)
    obj={"id":f"{USER}/{slug}","title":title,"code_file":notebook,"language":"python",
         "kernel_type":"notebook","is_private":True,"enable_gpu":False,"enable_internet":folder=="bmclab",
         "dataset_sources":[source],"competition_sources":[],"kernel_sources":[]}
    (dest/"kernel-metadata.json").write_text(json.dumps(obj,indent=2),encoding="utf8")
print("Prepared BMCLab private input and three notebook folders")
