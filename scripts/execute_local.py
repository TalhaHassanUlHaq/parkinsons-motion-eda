"""Execute a notebook locally and save the notebook with outputs for review."""
import argparse
import os
from pathlib import Path
import nbformat
from nbclient import NotebookClient

root=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument("name",choices=["mmu","bmclab","parkinsons_pose"])
ap.add_argument("--bm-input",default=None)
a=ap.parse_args()
if a.bm_input: os.environ["PARKINSON_BM_INPUT"]=a.bm_input
source=root/"notebooks"/f"eda_{a.name}.ipynb"
nb=nbformat.read(source,as_version=4)
client=NotebookClient(nb,timeout=1800,kernel_name="python3",resources={"metadata":{"path":str(root)}})
client.execute()
out=root/"outputs/executed";out.mkdir(parents=True,exist_ok=True)
target=out/f"eda_{a.name}{'_staged' if a.bm_input else ''}_executed.ipynb"
nbformat.write(nb,target)
print(f"Executed {source} -> {target}")
