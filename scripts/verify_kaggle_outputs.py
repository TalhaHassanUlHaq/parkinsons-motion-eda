"""Verify the three downloaded completed Kaggle runs against their manifests."""
from pathlib import Path
import csv
import json
import imageio_ffmpeg

root=Path(__file__).resolve().parents[1]
rows=[]
for dataset in ["mmu","bmclab","pose_estimation"]:
    run=root/"outputs/kaggle"/dataset/"outputs"
    manifest=run/"tables"/f"{dataset}_simulation_manifest.csv"
    for item in csv.DictReader(manifest.open(encoding="utf8")):
        name=Path(item["output_filename"]).name
        video=run/"videos"/dataset/name
        preview=run/"figures"/dataset/(Path(name).stem+".png")
        assert video.exists() and preview.exists(),name
        reader=imageio_ffmpeg.read_frames(str(video));meta=next(reader)
        count=sum(1 for _ in reader)
        assert count==int(item["rendered_frames"]) and meta["duration"]>0,(name,count)
        rows.append({"dataset":dataset,"file":str(video),"frames":count,
                     "duration_s":meta["duration"],"fps":meta["fps"],"source":item["source_recording_id"]})
assert len(rows)==9
out=root/"outputs/reports/kaggle_video_verification.json"
out.write_text(json.dumps(rows,indent=2),encoding="utf8")
print("Verified",len(rows),"downloaded Kaggle MP4s with",sum(x["frames"] for x in rows),"decoded frames")
