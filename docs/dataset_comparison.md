# Parkinson motion datasets: verified exploratory comparison

**Scope and evidence.** This report combines a local inventory of the original BMCLab archive, the official Parkinson pose archive, three inspected MMU JSON files, and nine decoded simulation MP4s. Full-file counts and published participant counts are identified separately from the three-sequence reconstruction samples. No model was trained and no pose or clinical label was invented.

| Criterion | MMU visual-based | BMCLab original C3D | Parkinson pose estimation |
|---|---|---|---|
| Data modality | AlphaPose JSON detections from walking video | Optical marker C3D plus processed gait files | CPM trajectory JSON text, CPM confidence, toe-tapping optical flow |
| 2D or 3D | 2D image pixels | Measured 3D; C3D millimetres converted to metres | 2D image pixels |
| Participants | 167 reported on Kaggle data card; identities not verifiable from sampled JSON | 26 distinct `SUB` IDs in original C3D archive and metadata | 9 distinct IDs in official `sn_numbers.txt` |
| Sequences | 292 files listed on Kaggle data card; 3 JSON recordings inspected and animated | 935 C3D files in local archive: 887 walk, 48 static; 3 walks animated | 387 communication, 124 drinking, 132 leg-agility pose entries; 133 toe-tapping flow entries; 3 pose entries animated |
| Temporal information | Numeric `image_id` frame numbers; source FPS unavailable; 3 frame gaps in one sampled file | 150 Hz in selected C3D headers, 3 trial lengths 2.93–15.57 s | 30 FPS capture in associated paper; selected recordings 9.53–14.17 s |
| Clinical labels | Normal, Mild, Moderate, Severe folder labels; sampled Normal, Mild, Moderate | ON/OFF source names; participant FoG and ON/OFF scales in `PDGinfo.xlsx` | Trial-linked UDysRS, UPDRS, CAPSIT and subject map; sampled UDysRS subscores |
| Movement types | Walking in variable video settings | Overground walking and static calibration | Seated communication, drinking, leg agility; toe tapping represented as flow |
| Skeleton reconstruction | Yes, 26 HALPE body joints extracted from 136-point detections | Yes, 18 named anatomical markers displayed from original 3D trajectories | Yes, 14 CPM body joints plus head and neck from original trajectories |
| True 3D motion availability | No | Yes, marker positions | No |
| Human avatar feasibility | 3D would be inferred and camera dependent | Marker-driven rig or SMPL-style fitting is technically plausible, subject to marker registration and separately licensed body model assets; no mesh was produced | 3D would be inferred; source video absent from archive |
| Data quality limitations | No capture FPS or trustworthy participant IDs in sampled files; duplicate detections, missing frames and low-confidence joints | Selected 18 markers have 100% valid residual/coordinate checks; short walking trials; additional files need individual checks | CPM uncertainty and occlusion; no original images in archive; leg-agility split indices need special mapping |
| Future VLM suitability | Useful 2D gait pose and category prototype; actual image/video availability must be resolved | Strong 3D motion foundation; no scene imagery in C3D | Strongest task-specific clinical annotations; no images in downloaded archive |

## Observed reconstruction quality

The four MMU category folders contain **292 JSON recordings**: 150 Normal, 29 Mild, 61 Moderate and 52 Severe. File counts are imbalanced and cannot be converted into participant counts from filenames alone. The selected sample has **671 unique pose frames** across three JSON files. One mild frame and two moderate frames had extra detections; the normal recording has three frame-number gaps. A nearest-hip continuity rule selects one detection per frame. At a 0.25 confidence threshold, the selected body-point validity fractions are 57.7%, 63.6%, and 59.0%. Output playback is 20 FPS for viewing only; true duration and physical velocity cannot be recovered without source timing. A severity label in a folder is not an independently validated subject-level diagnosis.

The BMCLab sample contains **3,233 measured frames** from `SUB01` OFF, `SUB01` ON, and `SUB02` OFF walking trials. C3D headers report 150 Hz and millimetres. Median marker heights verify source **Y** as the vertical axis: heel about 0.055 m, ankle about 0.097 m, knee about 0.509 m, and C7 about 1.476 m in an inspected trial. The animation maps X/Z/Y to display X/Y/Z and uses fixed axes. The longest trial contributes its first eight seconds; the shorter trials preserve their full duration. All are resampled to 20 FPS without changing source timing. The archive inventory has 498 ON and 437 OFF C3Ds including static trials; this is a file count, not a balanced participant comparison. The metadata has 26 distinct participant IDs, split 13 freezer and 13 nonfreezer IDs in its rows.

The clinical pose sample contains **1,061 trajectory frames** from communication trial `26-1`, drinking trial `217`, and communication trial `22-3`, mapped to three different subject IDs. Valid body-point fractions after a 0.2 CPM response threshold are 100%, 92.8%, and 100%. Videos show the first eight seconds of each measured trajectory at a resampled 20 FPS. The dataset authors state that the original cameras captured 30 FPS; no original videos are present in the archive. Leg agility provides active/rest split arrays with separate `idx` mappings, so the current renderer intentionally selects communication and drinking recordings. Toe tapping is optical flow and cannot be rendered as a whole-body skeleton from the released flow alone.

The Kaggle pose mirror is published under the same `limi44` account as the official repository. Its file names and byte sizes match the official archive; downloaded `sn_numbers.txt`, `UDysRS.txt`, and the complete `Drinking_all_export.txt` trajectory file are byte-identical to the official archive copies by SHA-256 comparison. Kaggle labels its mirror license as `other`, while the repository's archive explicitly supplies a CC BY 4.0 license file.

## Distribution skewness and imbalance

Skewness below is the adjusted Fisher–Pearson coefficient calculated by `pandas.Series.skew` on numeric values. Positive values indicate a longer right tail. Category labels are nominal, so their counts and imbalance ratios are reported separately; assigning them numbers would create a meaningless skewness statistic.

| Dataset | Numeric distribution and scope | Skewness | Categorical imbalance |
|---|---|---:|---|
| MMU | Frame counts of the three selected JSON recordings only; full archive lengths and source FPS unavailable | +1.19 (selection only) | 292 labeled JSONs: 150 Normal, 29 Mild, 61 Moderate, 52 Severe; largest/smallest **5.17:1** |
| BMCLab | C3D trial counts per subject across 26 subjects in the complete 935-file archive inventory | −1.37 | 498 ON vs 437 OFF files, **1.14:1**; repeated trials are not independent subjects |
| Clinical pose | Durations of 643 communication, drinking, and leg-agility trajectories at reported 30 FPS | +0.58 pooled; +0.94 communication, +1.23 drinking, +0.41 leg agility | 387 communication, 124 drinking, 132 leg agility; largest/smallest **3.12:1** |

**UPDRS score skewness:** In the local full BMCLab metadata, ON UPDRS-III scores have skewness **+0.43** across 25 usable participant scores, and OFF scores **+1.10** across 23. The private Kaggle input retains only two participant rows, so its score skewness is correctly undefined. In the clinical pose archive's `UPDRS.txt`, the released **Total** score has skewness **+1.01** across 79 trial IDs; the separate **3.10 gait item** has skewness **+0.59** across the same 79 trial IDs. These are distinct source scales and units of observation: BMCLab rows are participants, whereas the pose ratings are trial linked and may repeat participants. Pooled pose duration skew should be read alongside the per-task values because the tasks have different recording lengths. These are descriptive dataset properties, not evidence of motor symptom severity. The executed notebooks save the underlying tables and display the corresponding plots.

## Recommendations

1. **Accurate 3D motion:** BMCLab. Its original C3D coordinates, known units, timing, and named markers support a defensible spatial reconstruction. A surface avatar needs extra model fitting and licensed body assets.
2. **Accessible pose-based walking analysis:** MMU. Its 2D HALPE JSON is straightforward to animate, with frame-order and tracking checks. Source FPS and subject identity remain unknown in sampled files.
3. **Clinically annotated assessment:** Parkinson pose estimation. UDysRS/UPDRS/CAPSIT files and trial-to-subject mapping align with multiple seated tasks. It is the best choice for studying task-specific labels, subject to the available annotation granularity.
4. **Future VLM proof of concept:** MMU if original frame imagery can be accessed lawfully, because it targets walking and supplies detailed 2D body/hand/face points. If the first experiment uses pose-only inputs and clinical labels, the Parkinson pose dataset is a better research testbed. Neither released pose archive contains the full original assessment video in the inspected files.

## Verification and execution limits

The locally generated `simulation_manifest.csv` lists all nine source recordings, frame ranges, FPS, video paths, and clinical labels. Local video verification decoded all nine MP4s and counted **1,206 frames** in total, with a corresponding PNG preview for each. All three notebooks were executed locally. Generated videos, source recordings, and executed notebook artifacts are excluded from this public code repository.

All three notebooks also completed privately on Kaggle. The final MMU run found 292 JSON files in the four labeled folders and 10,413 auxiliary JSON files in the same mount, which were excluded from the category chart. Three MP4s per remote notebook were downloaded and decoded successfully, totaling 1,206 frames.

The BMCLab Kaggle run used a private, minimized sample after explicit authorization. The full clinical spreadsheet was excluded. This public repository contains analysis code and aggregate findings only.

## Sources and licensing

- [MMU Kaggle data card](https://www.kaggle.com/datasets/teeconnie/mmu-visual-based-parkinsons-disease-dataset), CC BY-NC 4.0; [HALPE keypoint order](https://github.com/Fang-Haoshu/Halpe-FullBody).
- [Original BMCLab Figshare dataset](https://figshare.com/articles/dataset/A_dataset_of_overground_walking_full-body_kinematics_and_kinetics_in_individuals_with_Parkinson_s_disease/14896881), CC BY 4.0; [Shida et al. data paper](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.992585/full).
- [Official pose repository and CC BY 4.0 archive link](https://github.com/limi44/Parkinson-s-Pose-Estimation-Dataset); [Li et al. assessment paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6219082/).
