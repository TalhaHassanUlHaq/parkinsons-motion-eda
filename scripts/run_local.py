"""Render the nine documented measured-pose samples; run one dataset at a time."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.pipeline import load_mmu,load_bmclab,load_pose_record,render,quality,save_manifest

MMU=["normal_pexels_03032022_10.json","mild_youtube_12122021_1.json",
     "moderate_youtube_03032022_02.json"]
BM=["SUB01_off_walk_14.c3d","SUB01_on_walk_19.c3d","SUB02_off_walk_6.c3d"]
POSE=[("Communication_all_export.txt","26-1"),
      ("Drinking_all_export.txt","217"),
      ("Communication_all_export.txt","22-3")]

def run(dataset, root=ROOT):
    root=Path(root)
    if dataset=="mmu":
        motions=[load_mmu(root/"raw/mmu"/name) for name in MMU]
        out_name="mmu"
    elif dataset=="bmclab":
        motions=[load_bmclab(root/"raw/bmclab"/name) for name in BM]
        out_name="bmclab"
    elif dataset=="pose_estimation":
        subjects=json.loads((root/"raw/pose/sn_numbers.txt").read_text(encoding="utf8"))
        motions=[load_pose_record(root/"raw/pose"/name,key,subjects) for name,key in POSE]
        out_name="pose_estimation"
    else: raise ValueError(dataset)
    summary=[quality(m) for m in motions]
    table=root/"outputs/tables";table.mkdir(parents=True,exist_ok=True)
    import csv
    with (table/f"{dataset}_selected_quality.csv").open("w",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    manifest=[]
    for i,m in enumerate(motions,1):
        stem=f"{out_name}_{i:02d}"
        video=root/"outputs/videos"/out_name/f"{stem}.mp4"
        preview=root/"outputs/figures"/out_name/f"{stem}.png"
        row=render(m,video,preview)
        manifest.append(row)
        print(f"Rendered {video} ({row['rendered_frames']} frames)",flush=True)
    save_manifest(manifest,root/"outputs/tables"/f"{dataset}_simulation_manifest.csv")
    return manifest

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("dataset",choices=["mmu","bmclab","pose_estimation"])
    ap.add_argument("--root",default=str(ROOT));a=ap.parse_args()
    run(a.dataset,a.root)
