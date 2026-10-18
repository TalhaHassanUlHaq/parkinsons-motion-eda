"""Decode every generated MP4 and check source traceability."""
from pathlib import Path
import csv
import hashlib
import json
import imageio_ffmpeg

root=Path(__file__).resolve().parents[1]
reports=[]
rows=[]
for dataset in ["mmu","bmclab","pose_estimation"]:
    manifest=root/"outputs/tables"/f"{dataset}_simulation_manifest.csv"
    for row in csv.DictReader(manifest.open(encoding="utf8")):
        rows.append(row)
        video=Path(row["output_filename"])
        preview=Path(row["preview_filename"])
        assert video.exists() and video.stat().st_size>1000,video
        assert preview.exists() and preview.stat().st_size>1000,preview
        stream=imageio_ffmpeg.read_frames(str(video))
        meta=next(stream)
        count=sum(1 for _ in stream)
        assert count>1 and count==int(row["rendered_frames"]),(video,count,row["rendered_frames"])
        assert meta["fps"]>0 and meta["duration"]>0,meta
        reports.append({"dataset":dataset,"video":str(video),"source":row["source_recording_id"],
                        "decoded_frames":count,"duration_s":meta["duration"],"fps":meta["fps"],
                        "bytes":video.stat().st_size,"preview":str(preview)})
out=root/"outputs/reports/video_verification.json";out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(reports,indent=2),encoding="utf8")
combined=root/"outputs/tables/simulation_manifest.csv"
with combined.open("w",newline="",encoding="utf8") as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
checks=[]
for row in rows:
    dataset={"MMU":"mmu","BMCLab":"bmclab","Pose estimation":"pose"}[row["dataset"]]
    name=row["source_recording_id"].split(":")[0]
    source=root/"raw"/dataset/name
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    checks.append({"dataset":row["dataset"],"source_recording_id":row["source_recording_id"],
                   "source_file":str(source),"source_bytes":source.stat().st_size,"sha256":digest})
with (root/"outputs/tables/source_checksums.csv").open("w",newline="",encoding="utf8") as f:
    writer=csv.DictWriter(f,fieldnames=list(checks[0]));writer.writeheader();writer.writerows(checks)
print("Verified",len(reports),"MP4s with",sum(x["decoded_frames"] for x in reports),"decoded frames")
