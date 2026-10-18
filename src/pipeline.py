"""Measured pose loading, quality checks, and traceable animation rendering."""
from __future__ import annotations

import csv
import json
import re
from xml.etree import ElementTree as ET
from zipfile import ZipFile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HALPE_NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
 "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
 "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
 "head", "neck", "hip", "left_big_toe", "right_big_toe", "left_small_toe", "right_small_toe",
 "left_heel", "right_heel"]
HALPE_EDGES = [(17,18),(18,5),(18,6),(5,7),(7,9),(6,8),(8,10),(18,19),
 (19,11),(19,12),(11,13),(13,15),(12,14),(14,16),(15,20),(15,22),(15,24),
 (16,21),(16,23),(16,25)]
CPM_NAMES = ["head","neck","Lsho","Lelb","Lwri","Rsho","Relb","Rwri",
 "Lhip","Lkne","Lank","Rhip","Rkne","Rank"]
CPM_EDGES = [(0,1),(1,2),(2,3),(3,4),(1,5),(5,6),(6,7),(1,8),(8,9),(9,10),
 (1,11),(11,12),(12,13),(8,11)]
BM_MARKERS = ["CLAV","C7","RSHO","REL","RWL","LSHO","LEL","LWL",
 "R.ASIS","L.ASIS","R.Knee","R.Ankle","R.Heel","R.MT2",
 "L.Knee","L.Ankle","L.Heel","L.MT2"]
BM_EDGES = [(1,0),(0,2),(2,3),(3,4),(0,5),(5,6),(6,7),
 (0,8),(0,9),(8,9),(8,10),(10,11),(11,12),(11,13),
 (9,14),(14,15),(15,16),(15,17)]


@dataclass
class Motion:
    dataset: str
    source: str
    points: np.ndarray       # frames x joints x dimension; metres for BMCLab, pixels for 2D
    confidence: np.ndarray   # frames x joints
    frame_ids: np.ndarray
    fps: float | None
    names: list[str]
    edges: list[tuple[int,int]]
    label: str
    subject: str | None = None
    warning: str = ""
    clinical_label: str | None = None


def _numeric_frame(value):
    m = re.search(r"(\d+)(?:\.[A-Za-z]+)?$", str(value))
    if not m:
        raise ValueError(f"No numeric frame ID: {value}")
    return int(m.group(1))


def load_mmu(path: str | Path, confidence_threshold=0.25) -> Motion:
    path = Path(path)
    rows = json.loads(path.read_text(encoding="utf-8"))
    by_frame = {}
    for row in rows:
        arr = np.asarray(row["keypoints"], dtype=float)
        if arr.size != 408:
            continue
        frame = _numeric_frame(row["image_id"])
        by_frame.setdefault(frame, []).append((row, arr.reshape(136, 3)))
    if len(by_frame) < 2:
        raise ValueError(f"No reconstructible sequence: {path}")
    selected, previous = [], None
    for frame in sorted(by_frame):
        candidates = by_frame[frame]
        def cost(item):
            row, pose = item
            center = np.nanmean(pose[[11,12], :2], axis=0)
            score = float(np.asarray(row.get("score", 0)).reshape(-1)[0])
            if previous is None:
                return -score
            return np.linalg.norm(center-previous) - min(score, 10.0) * 5
        chosen = min(candidates, key=cost)
        previous = np.nanmean(chosen[1][[11,12], :2], axis=0)
        selected.append((frame, chosen[1]))
    frame_ids = np.array([x[0] for x in selected], dtype=int)
    pose = np.stack([x[1] for x in selected])
    points = pose[:, :26, :2]
    conf = pose[:, :26, 2]
    points[conf < confidence_threshold] = np.nan
    label = next((part for part in path.parts if part.upper() in {"MILD","MODERATE","NORMAL","SEVERE"}),
                 path.stem.split("_")[0]).title()
    duplicate_count = sum(len(v)-1 for v in by_frame.values())
    gaps = int(np.sum(np.diff(frame_ids) > 1))
    return Motion("MMU",path.name,points,conf,frame_ids,None,HALPE_NAMES,HALPE_EDGES,label,
                  warning=f"{duplicate_count} extra detections; {gaps} frame gaps; source FPS unavailable",
                  clinical_label=label)


def load_bmclab(path: str | Path) -> Motion:
    import ezc3d
    path = Path(path)
    c = ezc3d.c3d(str(path))
    params = c["parameters"]["POINT"]
    labels = list(params["LABELS"]["value"])
    units = params["UNITS"]["value"][0].strip().lower()
    if units not in {"mm","m"}:
        raise ValueError(f"Unexpected C3D point units: {units}")
    raw = c["data"]["points"]
    if not all(name in labels for name in BM_MARKERS):
        raise ValueError(f"Missing anatomical markers: {set(BM_MARKERS)-set(labels)}")
    points = np.transpose(raw[:3, [labels.index(n) for n in BM_MARKERS], :], (2,1,0)).copy()
    if units == "mm":
        points /= 1000.0
    residual = np.transpose(raw[3, [labels.index(n) for n in BM_MARKERS], :], (1,0))
    valid = np.isfinite(points).all(axis=2) & np.isfinite(residual) & (residual >= 0)
    points[~valid] = np.nan
    fps = float(params["RATE"]["value"][0])
    sub = re.search(r"SUB\d+",path.stem,re.I)
    med = re.search(r"_(on|off)_",path.stem,re.I)
    return Motion("BMCLab",path.name,points,valid.astype(float),np.arange(len(points)),fps,
                  BM_MARKERS,BM_EDGES,(med.group(1).upper() if med else "Unknown"),
                  subject=(sub.group(0) if sub else None),
                  warning=f"Original C3D coordinates; {units} converted to metres; C3D Y is vertical",
                  clinical_label=(med.group(1).upper() if med else None))


def load_pose_file(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_pose_record(path: str | Path, key: str, subjects: dict | None = None) -> Motion:
    path = Path(path)
    record = load_pose_file(path)[key]
    if "idx" in record:
        raise ValueError("Leg-agility split indices require a separate mapping; select communication or drinking")
    lens = [len(record["position"][name]) for name in CPM_NAMES]
    n = min(lens)
    points = np.stack([np.asarray(record["position"][name][:n],float) for name in CPM_NAMES],axis=1)
    confidence = np.stack([np.asarray(record["resp"][name][:n],float) for name in CPM_NAMES],axis=1)
    if points.shape != (n,14,2):
        raise ValueError("Unexpected CPM trajectory shape")
    points[~np.isfinite(points).all(axis=2)] = np.nan
    points[confidence < 0.2] = np.nan
    task = "Communication" if path.name.startswith("Communication") else "Drinking"
    trial = re.split(r"[-\s]",key)[0]
    ratings_path=path.parent/"UDysRS.txt"
    ratings=load_pose_file(ratings_path).get(task,{}).get(trial) if ratings_path.exists() else None
    return Motion("Pose estimation",f"{path.name}:{key}",points,confidence,np.arange(n),30.,
                  CPM_NAMES,CPM_EDGES,task,subject=(subjects or {}).get(trial),
                  warning="2D camera coordinates; 30 FPS from Li et al. 2018; no original video in archive",
                  clinical_label=("UDysRS subscores " + json.dumps(ratings) if ratings is not None else None))


def quality(m: Motion) -> dict:
    p=m.points
    valid=np.isfinite(p).all(axis=2)
    disp=np.linalg.norm(np.diff(p,axis=0),axis=2)
    return {"dataset":m.dataset,"source":m.source,"label":m.label,"subject":m.subject or "Unknown",
            "frames":len(p),"duration_s":(m.frame_ids[-1]-m.frame_ids[0]+1)/m.fps if m.fps else None,
            "source_fps":m.fps,"joint_valid_fraction":float(valid.mean()),
            "median_displacement_per_frame":float(np.nanmedian(disp)),
            "frame_gaps":int(np.sum(np.diff(m.frame_ids)>1)),"warning":m.warning}


def _view_indices(m: Motion, out_fps=20, max_seconds=8):
    if m.fps:
        end=min(float(m.frame_ids[-1]-m.frame_ids[0])/m.fps,max_seconds)
        times=np.arange(0,end+1e-9,1/out_fps)
        ids=np.searchsorted(m.frame_ids, m.frame_ids[0]+times*m.fps).clip(0,len(m.frame_ids)-1)
    else:
        # Source FPS absent: one recorded pose per output frame; speed is illustrative only.
        ids=np.arange(min(len(m.frame_ids),int(max_seconds*out_fps)))
    return np.unique(ids)


def render(m: Motion, video: str | Path, preview: str | Path, out_fps=20, max_seconds=8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    import shutil
    video,preview=Path(video),Path(preview)
    video.parent.mkdir(parents=True,exist_ok=True); preview.parent.mkdir(parents=True,exist_ok=True)
    encoder=shutil.which("ffmpeg")
    if encoder is None:
        import imageio_ffmpeg
        encoder=imageio_ffmpeg.get_ffmpeg_exe()
    matplotlib.rcParams["animation.ffmpeg_path"] = encoder
    idx=_view_indices(m,out_fps,max_seconds)
    if len(idx)<2: raise ValueError("Fewer than two animation frames")
    # In these files C3D Y is vertical: shoulder Y exceeds hip, knee, and heel Y.
    # Matplotlib uses its third displayed axis as vertical, so display X/Z/Y.
    pts=m.points[idx][:,:,[0,2,1]] if m.points.shape[2]==3 else m.points[idx]
    good=pts[np.isfinite(pts).all(axis=2)]
    lo=np.nanpercentile(good,1,axis=0); hi=np.nanpercentile(good,99,axis=0)
    span=np.maximum(hi-lo,0.1 if pts.shape[2]==3 else 10)
    lo-=span*0.12; hi+=span*0.12
    if pts.shape[2]==3:
        fig=plt.figure(figsize=(6,6)); ax=fig.add_subplot(111,projection="3d")
        # Keep equal physical scale and include the ground plane.
        center=(lo+hi)/2; radius=max((hi-lo))/2
        limits=[(float(c-radius),float(c+radius)) for c in center]
        def setup():
            ax.set_xlim(*limits[0]); ax.set_ylim(*limits[1]); ax.set_zlim(*limits[2]);
            ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)"); ax.set_zlabel("Y up (m)")
            ax.view_init(elev=16,azim=-65)
            ax.set_box_aspect((1,1,1),zoom=1.2)
            xx=np.linspace(*limits[0],2); yy=np.linspace(*limits[1],2)
            X,Y=np.meshgrid(xx,yy); ax.plot_surface(X,Y,np.zeros_like(X),alpha=.08,color="gray")
    else:
        fig,ax=plt.subplots(figsize=(6,6))
        limits=[(float(lo[i]),float(hi[i])) for i in range(2)]
        def setup():
            ax.set_xlim(*limits[0]); ax.set_ylim(*limits[1]); ax.invert_yaxis(); ax.set_aspect("equal",adjustable="box")
            ax.set_xlabel("image X (pixels)"); ax.set_ylabel("image Y (pixels)")
    def draw(j):
        ax.clear(); setup(); p=pts[j]
        for a,b in m.edges:
            if np.isfinite(p[[a,b]]).all():
                q=p[[a,b]]
                if p.shape[1]==3: ax.plot(q[:,0],q[:,1],q[:,2],color="#2563eb",lw=2.4)
                else: ax.plot(q[:,0],q[:,1],color="#2563eb",lw=2.4)
        ok=np.isfinite(p).all(axis=1)
        if p.shape[1]==3: ax.scatter(p[ok,0],p[ok,1],p[ok,2],s=13,color="#ef4444",depthshade=False)
        else:
            sizes=10+22*np.clip(m.confidence[idx[j],ok],0,1)
            ax.scatter(p[ok,0],p[ok,1],s=sizes,color="#ef4444")
        ax.set_title(f"{m.dataset} | {m.source}\n{m.label} | source frame {m.frame_ids[idx[j]]}",fontsize=10)
    draw(len(idx)//2); fig.savefig(preview,dpi=130,bbox_inches="tight")
    anim=FuncAnimation(fig,draw,frames=len(idx),interval=1000/out_fps)
    anim.save(str(video),writer=FFMpegWriter(fps=out_fps,codec="libx264",bitrate=1200))
    plt.close(fig)
    return {"dataset":m.dataset,"source_recording_id":m.source,"selected_sequence_range":f"{m.frame_ids[idx[0]]}-{m.frame_ids[idx[-1]]}",
            "original_fps":m.fps if m.fps else "Unknown","output_fps":out_fps,"output_filename":str(video),
            "rendered_frames":len(idx),"modality":"3D" if m.points.shape[2]==3 else "2D",
            "status":"measured markers" if m.dataset=="BMCLab" else "estimated 2D pose from real video",
            "clinical_label":m.clinical_label or "Unavailable","subject_id":m.subject or "Unknown","preview_filename":str(preview)}


def save_manifest(rows, path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def read_simple_xlsx(path):
    """Read the first worksheet of PDGinfo.xlsx without an optional Excel package."""
    ns={"m":"http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(path) as z:
        strings=["".join(x.itertext()) for x in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si",ns)]
        root=ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    rows=[]
    for row in root.findall(".//m:sheetData/m:row",ns):
        cells={}
        for c in row.findall("m:c",ns):
            value=c.find("m:v",ns)
            if value is None: continue
            col=re.match(r"[A-Z]+",c.get("r")).group()
            cells[col]=strings[int(value.text)] if c.get("t")=="s" else value.text
        rows.append(cells)
    header=rows[0]
    return [{header[k]:v for k,v in row.items() if k in header} for row in rows[1:]]
