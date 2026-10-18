"""Build three instructor-facing notebooks with public aggregate outputs."""

from pathlib import Path
from shutil import copy2
from tempfile import TemporaryDirectory

import nbformat as nbf
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"

CONFIGS = [
    {
        "key": "MMU",
        "slug": "mmu",
        "title": "MMU visual-based Parkinson's dataset",
        "intro": "Walking-video detections represented as 2D AlphaPose keypoints. The source folders give Normal, Mild, Moderate, and Severe labels. Inspected JSONs do not establish participant identities or source FPS.",
        "structure": "Each JSON contains detections keyed by image frame. The source-specific notebook selects one continuous person track and extracts the documented 26 HALPE body joints from a 136-point detector output. The 292 count refers to labeled JSON files, while quality and animations here refer to only three inspected files.",
        "duration": "True duration is unavailable because the original video FPS is not recorded in the inspected JSON. The bars below therefore show **frame counts**, not seconds. Output video playback at 20 FPS is illustrative and must not be read as capture timing.",
        "score": "The numbers 1–4 below order the source **folder labels**: 1 Normal, 2 Mild, 3 Moderate, 4 Severe. They are not recorded UPDRS scores. No score-0 folder exists in this four-class view.",
        "other": "No independently validated participant-level UPDRS or medication status is present in the inspected JSON. The four severity folders provide the available labels. Filenames cannot be used to count unique patients reliably.",
        "quality": "Low-confidence keypoints are masked by the source loader. The selected recordings contain duplicate detections and three frame-number gaps in one file; continuity tracking chooses one detection per frame. Image coordinates depend on camera position, so pixel movement is not clinical walking speed.",
        "motion": "Motion plots show a selected image-space joint trajectory and displacement. Pixel rates are per source frame, because source seconds cannot be recovered. The observed rate is not a gait-speed diagnosis.",
        "skew": "The +1.19 skewness describes frame counts of **three deliberately chosen JSON files**. It is not an archive-wide length estimate. Class imbalance is better described by the 150:29:61:52 folder counts; assigning a numeric skewness to those nominal labels would be misleading.",
        "mesh": "Only 2D estimated keypoints are available in the inspected JSON. A 3D body or SMPL surface would require additional inference and source imagery; none is claimed in this notebook.",
        "takeaway": "Use MMU for an accessible 2D walking-pose prototype and severity-folder distribution. Resolve source video timing, person tracking, subject identity, and image availability before making temporal or clinical claims.",
        "source": "[MMU Kaggle data card](https://www.kaggle.com/datasets/teeconnie/mmu-visual-based-parkinsons-disease-dataset)",
    },
    {
        "key": "BMCLab",
        "slug": "bmclab",
        "title": "BMCLab original C3D gait dataset",
        "intro": "Measured optical marker trajectories from overground walking. The original archive contains 935 C3D files across 26 subject IDs, and local metadata contains participant-level ON/OFF UPDRS-III scores.",
        "structure": "Each C3D stores timed 3D marker positions, not an SMPL body mesh. The selected files report 150 Hz and millimetres; the loader converts positions to metres. Source Y is vertical. The archive inventory distinguishes 887 walking files and 48 static calibration files.",
        "duration": "The selected C3D walking examples last 2.93–15.57 seconds, calculated from the actual 150 Hz header. These three examples do not characterize the full archive duration distribution.",
        "score": "The bar chart uses the source's **UPDRS-III walking item**, scored 0–4 separately for ON and OFF participant assessments. No score 3 or 4 appears in these available walking-item rows. UPDRS-III **total** is a different numeric field; its median and skewness appear below.",
        "other": "Source filenames give ON and OFF medication conditions. The local metadata also groups 13 freezer and 13 nonfreezer participants. A freezer group is not an annotated freezing event in every trial. File-count differences cannot estimate medication effect.",
        "quality": "All selected marker coordinates pass the residual and finite-coordinate checks for the 18 displayed markers. This verifies the three plotted C3Ds only, not every trial in the 935-file archive. Source X/Z/Y is mapped to display X/Y/Z for a conventional upright view.",
        "motion": "Plots use real 3D coordinates in metres and source timing at 150 Hz. The selected pelvis-rate summary uses metres per second; walking trials differ in length and protocol, so the three values are descriptive examples.",
        "skew": "Trial counts per subject have adjusted sample skewness −1.37 across the 26 archive IDs. UPDRS-III total skewness is +0.43 for 25 available ON participant scores and +1.10 for 23 OFF scores. These are participant observations, not repeated walk counts.",
        "mesh": "Measured 3D marker positions make BMCLab the strongest of the three sources for a future full-body reconstruction. A surface mesh still requires marker registration, model fitting, and separately licensed body-model assets. This notebook shows original markers rather than a fabricated SMPL mesh.",
        "takeaway": "Use BMCLab when accurate 3D marker trajectories and known timing matter. Interpret ON/OFF clinical scores at participant level, and fit a body model only as a separate, documented step.",
        "source": "[Original BMCLab Figshare archive](https://figshare.com/articles/dataset/A_dataset_of_overground_walking_full-body_kinematics_and_kinetics_in_individuals_with_Parkinson_s_disease/14896881)",
    },
    {
        "key": "Clinical pose",
        "slug": "pose_estimation",
        "title": "Parkinson's pose-estimation clinical assessments",
        "intro": "2D CPM body trajectories from communication, drinking, and leg-agility clinical tasks. The official archive links trials to UPDRS, UDysRS, and CAPSIT ratings; original scene video is absent from the downloaded archive.",
        "structure": "The 643 full-body pose entries comprise 387 communication, 124 drinking, and 132 leg-agility trajectories. A separate toe-tapping file contains optical flow, not a full-body skeleton. The trial map contains nine distinct subject IDs. The three displayed pose examples come from communication and drinking.",
        "duration": "The associated paper reports 30 FPS capture. Full-archive task medians are shown alongside the three selected example durations. Task durations differ, so the pooled distribution should be interpreted with the task breakdown.",
        "score": "The released UPDRS **gait item 3.10** is averaged across raters and can be fractional. The chart keeps exact zero separate and bins positive values as category *k* = `(k−1, k]`. These are presentation bins, not original integer ratings. The separate **UPDRS Total** has its own median and skewness below.",
        "other": "Trial-linked UDysRS and CAPSIT fields are available in the source, but the plotted gait-item bins and Total summaries are specifically UPDRS. Medication ON/OFF is not an equivalent per-file condition in this inspected archive. Toe tapping is optical flow and is not included in the 643 full-body pose count.",
        "quality": "CPM confidence is used to mask uncertain 2D points. The mean valid-body-point fraction across the three selected trajectories is 97.6%; one selected drinking recording is lower at 92.8%. These checks do not validate every released trajectory.",
        "motion": "Motion plots are in camera pixels and use the reported 30 FPS. Communication and drinking include seated upper-body movement; pixel displacement depends on viewpoint and is not a physical gait-speed measure.",
        "skew": "Trajectory duration skewness is +0.58 across 643 entries; it differs by task. Released UPDRS Total skewness is +1.01 across 79 trial IDs; gait item 3.10 skewness is +0.59 across the same 79 IDs. Trial IDs may repeat a participant.",
        "mesh": "These are 2D pose estimates. Without source images, depth calibration, and an independently validated 3D model, a full body surface would be inferred rather than measured. The animation therefore displays the released 2D skeleton.",
        "takeaway": "Use this source for task-linked clinical pose analysis and score-distribution work. Distinguish trial IDs from subjects, preserve fractional ratings, and resolve original video access before an image-based VLM study.",
        "source": "[Official pose repository](https://github.com/limi44/Parkinson-s-Pose-Estimation-Dataset)",
    },
]


def build(cfg):
    nb = nbf.v4.new_notebook()

    def md(s):
        nb.cells.append(nbf.v4.new_markdown_cell(s))

    def code(s):
        nb.cells.append(nbf.v4.new_code_cell(s))

    key, slug = cfg["key"], cfg["slug"]
    md(f"""# {cfg['title']} — Instructor EDA

## What is this dataset?

{cfg['intro']}

This notebook follows the supplied CARE-PD example's presentation flow: setup, load, overview, duration, scores, clinical context, individual motion, animation, and final summary. It uses **this original source's actual format and fields**, not CARE-PD's derived SMPL representation.

**Evidence scope:** The public plots use verified aggregate results in `docs/presentation_data.json`. Motion figures and three MP4s display only when generated locally by `notebooks/eda_{slug if slug != 'pose_estimation' else 'parkinsons_pose'}.ipynb`. They are not included in the public repository. No classifier is trained or synthetic patient trajectory created.
""")
    md("""---
## 1. Setup

Open from the repository root or its `notebooks/` directory. Sections using public aggregate data work without the original archives. Run the detailed source notebook locally first for the motion gallery and videos.
""")
    code(f"""import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image, Video

ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'docs' / 'presentation_data.json').exists()), None)
if ROOT is None:
    raise FileNotFoundError('Run from the repository root or notebooks/ directory.')
DATA = json.loads((ROOT / 'docs' / 'presentation_data.json').read_text(encoding='utf-8'))
KEY, SLUG = {key!r}, {slug!r}
COLOR = {{'MMU': '#3268a8', 'BMCLab': '#e08035', 'Clinical pose': '#2a9d8f'}}[KEY]
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({{'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False}})
print('Aggregate snapshot:', DATA['as_of'])
print('Source:', KEY)
""")
    md(f"""---
## 2. Load and understand the source

{cfg['structure']}

The detailed source notebook documents the actual loader, coordinate mapping, frame IDs, and per-recording checks. This presentation reads a small aggregate snapshot so it remains runnable in the public repository.
""")
    code("""info = pd.Series(next(x for x in DATA['overview'] if x['dataset'] == KEY), name='value')
display(info.to_frame())
quality = pd.Series(DATA['selected_quality'][KEY], name='value')
print('Selected-example quality summary:')
display(quality.to_frame())
""")
    md("""---
## 3. Overview — how much data?

This chart counts the source's own file or trajectory types. It is not a count of independent people. The observation unit appears in the plot title and table.
""")
    code("""if KEY == 'MMU':
    counts = pd.Series({k: v for k, v in DATA['score_categories']['MMU folder labels'].items() if v is not None})
    counts.index = ['Normal', 'Mild', 'Moderate', 'Severe']
    unit = 'labeled JSON files'
elif KEY == 'BMCLab':
    counts = pd.Series({'Walking': 887, 'Static': 48})
    unit = 'C3D files'
else:
    counts = pd.Series(DATA['clinical_pose_tasks'])
    unit = 'pose trajectories'
fig, ax = plt.subplots(figsize=(7.5, 3.8))
bars = ax.bar(counts.index, counts.values, color=COLOR)
ax.bar_label(bars, padding=3)
ax.set(ylabel=unit, title=f'{KEY}: source inventory by type')
ax.set_ylim(0, counts.max() * 1.17)
plt.show()
print('Inventory total:', int(counts.sum()), unit)
""")
    md(f"""---
## 4. Recording duration

{cfg['duration']}
""")
    code("""if KEY == 'MMU':
    lengths = pd.Series([188, 122, 361], index=['selected 1', 'selected 2', 'selected 3'])
    ylabel = 'frames (source FPS unknown)'
else:
    rows = pd.DataFrame(DATA['selected_durations']).query('dataset == @KEY')
    lengths = pd.Series(rows.seconds.values, index=rows['sample'])
    ylabel = 'seconds'
fig, ax = plt.subplots(figsize=(7, 3.5))
bars = ax.bar(lengths.index, lengths.values, color=COLOR)
ax.bar_label(bars, fmt='%.1f' if KEY != 'MMU' else '%d', padding=3)
ax.set(ylabel=ylabel, title=f'{KEY}: three selected recordings')
ax.set_ylim(0, lengths.max() * 1.18)
plt.show()
if KEY == 'Clinical pose':
    print('Complete three-task duration summaries:')
    display(pd.DataFrame(DATA['pose_duration_by_task']).round(2))
""")
    md(f"""---
## 5. Score or source-label distribution

{cfg['score']}
""")
    code("""if KEY == 'MMU':
    score = pd.Series(DATA['score_categories']['MMU folder labels']).dropna().astype(int)
    score.index = ['1 Normal', '2 Mild', '3 Moderate', '4 Severe']
    fig, ax = plt.subplots(figsize=(7, 3.7))
    bars = ax.bar(score.index, score.values, color=COLOR)
    ax.set(ylabel='JSON recordings', title='MMU source folder labels (not UPDRS)')
elif KEY == 'BMCLab':
    score = pd.DataFrame({'ON': DATA['score_categories']['BMCLab ON walking item'],
                          'OFF': DATA['score_categories']['BMCLab OFF walking item']}).astype(int)
    fig, ax = plt.subplots(figsize=(7, 3.7))
    score.plot.bar(ax=ax, color=['#2a9d8f', '#e08035'], width=0.75)
    bars = ax.containers
    ax.set(xlabel='UPDRS-III walking item', ylabel='participants', title='BMCLab source walking-item ratings')
else:
    score = pd.Series(DATA['score_categories']['Clinical pose gait item bins']).astype(int)
    fig, ax = plt.subplots(figsize=(7, 3.7))
    bars = ax.bar(score.index, score.values, color=COLOR)
    ax.set(xlabel='Gait-item 3.10 display bin', ylabel='trial IDs', title='Clinical pose averaged UPDRS gait item')
for container in ([bars] if KEY != 'BMCLab' else bars):
    ax.bar_label(container, padding=2)
ax.set_ylim(0, (score.values.max() if KEY != 'BMCLab' else score.to_numpy().max()) * 1.2)
plt.show()
display(score.to_frame('count') if KEY != 'BMCLab' else score)
""")
    md("""### UPDRS total scores, when available

UPDRS-III Total and gait item are separate fields. A total score is not converted into categories 1–4 without a validated threshold. Skewness is the adjusted Fisher–Pearson sample statistic from the original numeric rows.
""")
    code("""updrs = pd.DataFrame(DATA['updrs_summary'])
updrs = updrs[updrs.dataset == KEY].drop(columns='dataset')
if updrs.empty:
    print('No verified numeric UPDRS score is available for MMU in the inspected JSON.')
else:
    display(updrs.round({'median': 2, 'skewness': 2}))
    fig, ax = plt.subplots(figsize=(7, 3.4))
    bars = ax.barh(updrs.field, updrs.skewness, color=COLOR)
    ax.bar_label(bars, fmt='%+.2f', padding=3)
    ax.invert_yaxis()
    ax.set(xlabel='Adjusted sample skewness', title=f'{KEY}: UPDRS field skewness')
    ax.set_xlim(0, updrs.skewness.max() * 1.3)
    plt.show()
""")
    md(f"""---
## 6. Medication status and clinical context

{cfg['other']}
""")
    code("""if KEY == 'BMCLab':
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    for ax, counts, title, ylabel in [
        (axes[0], DATA['medication_file_counts'], 'ON/OFF source files', 'C3D files'),
        (axes[1], DATA['fog_group_participants'], 'Participant FoG groups', 'participants')]:
        bars = ax.bar(list(counts), list(counts.values()), color=COLOR)
        ax.bar_label(bars, padding=3)
        ax.set(title=title, ylabel=ylabel)
        ax.set_ylim(0, max(counts.values()) * 1.2)
    plt.show()
elif KEY == 'Clinical pose':
    display(pd.Series(DATA['clinical_pose_tasks'], name='pose trajectories').to_frame())
    print('UPDRS, UDysRS and CAPSIT annotations are trial linked; toe tapping uses optical flow.')
else:
    print('Four folder labels are available. No source-confirmed medication or numeric UPDRS field is present.')
""")
    md(f"""---
## 7. Data quality and missing movement points

{cfg['quality']}
""")
    code("""q = DATA['selected_quality'][KEY]
display(pd.DataFrame([{
    'selected recordings': q['selected_recordings'],
    'source frames': q['source_frames'],
    'mean valid-joint fraction': q['mean_valid_joint_fraction'],
    'minimum valid-joint fraction': q['minimum_valid_joint_fraction'],
    'frame gaps': q['frame_gaps'],
}]).round(3))
fig, ax = plt.subplots(figsize=(5.5, 3.3))
vals = [q['mean_valid_joint_fraction'], q['minimum_valid_joint_fraction']]
bars = ax.bar(['Mean', 'Minimum'], vals, color=COLOR)
ax.bar_label(bars, fmt='%.1f%%', labels=[f'{v:.1%}' for v in vals], padding=3)
ax.set(ylim=(0, 1.12), ylabel='Valid selected body-point fraction', title=f'{KEY}: selected pose visibility')
plt.show()
""")
    md(f"""---
## 8. Visualising individual movement

{cfg['motion']}

The local source notebook generates the trajectory, confidence or missing-marker, and elbow-angle plots from actual coordinates. They appear below if present. The public version shows the source-backed summaries above and below without exposing individual movement.
""")
    code("""for suffix, title in [('movement', 'Joint trajectory and displacement'),
                      ('missing_confidence', 'Missing joints or confidence'),
                      ('elbow_angles', 'Selected elbow motion')]:
    path = ROOT / 'outputs' / 'figures' / f'{SLUG}_{suffix}.png'
    print(title)
    if path.exists():
        display(Image(filename=str(path), width=850))
    else:
        print('Local figure unavailable; run the detailed source notebook.')
q = DATA['selected_quality'][KEY]
print('Median of the three selected recording-median pelvis rates:',
      q['median_of_recording_median_rates'], q['rate_unit'])
""")
    md(f"""---
## 9. Distribution skewness

{cfg['skew']}
""")
    code("""skew = pd.DataFrame(DATA['skewness']).query('dataset == @KEY').drop(columns='dataset')
display(skew.round({'skewness': 2}))
fig, ax = plt.subplots(figsize=(8, max(3.2, len(skew) * 0.75 + 1)))
bars = ax.barh(skew.measure, skew.skewness, color=COLOR)
ax.bar_label(bars, fmt='%+.2f', padding=3)
ax.axvline(0, color='#555', linewidth=1)
ax.invert_yaxis()
ax.set(xlabel='Adjusted sample skewness', title=f'{KEY}: numeric distribution shape')
ax.set_xlim(min(-1.8, skew.skewness.min() * 1.2), max(1.6, skew.skewness.max() * 1.25))
plt.show()
""")
    md("""---
## 10. Animated human movement

The three clips use authentic poses or marker trajectories from the selected source recordings. Playback video is an analysis visualization, not original camera video. Local previews and MP4s are displayed when the detailed source notebook has generated them. They are intentionally absent from the public repository.
""")
    code("""preview = ROOT / 'outputs' / 'figures' / f'{SLUG}_static_skeletons.png'
if preview.exists():
    display(Image(filename=str(preview), width=900))
else:
    print('Local static skeleton gallery unavailable.')
for i in range(1, 4):
    video = ROOT / 'outputs' / 'videos' / SLUG / f'{SLUG}_{i:02d}.mp4'
    print(f'Selected recording {i}')
    if video.exists():
        display(Video(str(video), embed=True, width=540))
    else:
        print('Local MP4 unavailable; run the detailed source notebook.')
""")
    md(f"""---
## 11. Full 3D body mesh

{cfg['mesh']}
""")
    md(f"""---
## 12. Final summary

{cfg['takeaway']}

**Evidence to present:** inventory chart, correctly named score field and observation unit, numeric skewness with its sample size, selected motion quality, then the three source animations. The detailed reproducible analysis is in [`eda_{slug if slug != 'pose_estimation' else 'parkinsons_pose'}.ipynb`](eda_{slug if slug != 'pose_estimation' else 'parkinsons_pose'}.ipynb); the [comparison report](../docs/dataset_comparison.md) records provenance and limitations.

**Source:** {cfg['source']}.
""")
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    target = OUT / f"eda_{slug}_presentation.ipynb"
    nbf.write(nb, target)
    with TemporaryDirectory(prefix=f"parkinsons_{slug}_public_") as scratch:
        isolated = Path(scratch)
        (isolated / "docs").mkdir()
        copy2(ROOT / "docs" / "presentation_data.json", isolated / "docs" / "presentation_data.json")
        public_nb = nbf.read(target, as_version=4)
        NotebookClient(
            public_nb,
            timeout=180,
            kernel_name="python3",
            resources={"metadata": {"path": str(isolated)}},
        ).execute()
        nbf.write(public_nb, target)
    print(f"Built and rendered {target.relative_to(ROOT)} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for config in CONFIGS:
        build(config)
