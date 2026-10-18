"""Clinical pose distribution profile; embedded in the generated notebook."""
from IPython.display import display, Markdown

pose_rows = []
for task, fname in [('Communication','Communication_all_export.txt'),
                    ('Drinking','Drinking_all_export.txt'),
                    ('Leg agility','LA_split_all_export.txt')]:
    for trial, record in load_pose_file(pose_root/fname).items():
        frames = len(record['position']['head'])
        pose_rows.append({'task':task, 'trial':trial, 'frames':frames, 'duration_s':frames/30.0})
pose_lengths = pd.DataFrame(pose_rows)
task_counts = pose_lengths.task.value_counts().reindex(['Communication','Drinking','Leg agility'])
task_table = pd.DataFrame({'task':task_counts.index, 'trajectories':task_counts.values,
                           'share_pct':(100*task_counts/task_counts.sum()).round(1).values})
display(pd.DataFrame([{'pose_trajectories':len(pose_lengths), 'mapped_participants':len(set(subjects.values())),
                       'source_FPS':'30 Hz reported capture rate', 'pose_dimension':'estimated 2D pixels',
                       'labels':'task, participant map, trial-linked clinical ratings'}]))
display(task_table)
task_table.to_csv(OUTPUT/'tables'/'pose_estimation_task_distribution.csv', index=False)

profile_rows=[]
for task, group in [('All three pose tasks',pose_lengths),
                    *list(pose_lengths.groupby('task'))]:
    length=group.duration_s
    profile_rows.append({'measure':'trajectory duration', 'scope':task, 'n':len(length),
                         'median_s':length.median(), 'p90_s':length.quantile(.9),
                         'skewness':length.skew() if len(length)>=3 and length.nunique()>1 else float('nan')})
profile=pd.DataFrame(profile_rows)
display(profile.round(2))
profile.to_csv(OUTPUT/'tables'/'pose_estimation_distribution_profile.csv', index=False)

# The archive supplies trial-linked UPDRS ratings separately from the pose tasks.
# "Total" is the released total score; 3.10 is its gait item, not another total.
updrs = load_pose_file(pose_root/'UPDRS.txt')
updrs_rows = []
for field, name in [('Total','UPDRS Total'),('3.10','UPDRS item 3.10 (gait)')]:
    values = pd.to_numeric(pd.Series(updrs[field]), errors='coerce').dropna()
    updrs_rows.append({'measure':name, 'n_trial_ids':len(values), 'median':values.median(),
                       'mean':values.mean(), 'min':values.min(), 'max':values.max(),
                       'skewness':values.skew() if len(values)>=3 and values.nunique()>1 else float('nan')})
updrs_profile = pd.DataFrame(updrs_rows)
display(updrs_profile.round(2))
updrs_profile.to_csv(OUTPUT/'tables'/'pose_estimation_updrs_profile.csv', index=False)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.7))
for ax, field, title, color in zip(axes, ['Total','3.10'], ['UPDRS Total','UPDRS item 3.10 (gait)'], ['#2563eb','#0f766e']):
    values = pd.to_numeric(pd.Series(updrs[field]), errors='coerce').dropna()
    ax.hist(values, bins=15, color=color, edgecolor='white')
    ax.axvline(values.median(), color='#dc2626', linestyle='--', label=f'Median {values.median():.1f}')
    ax.set(xlabel='Released score', ylabel='Trial IDs', title=f'{title}; skew {values.skew():+.2f}')
    ax.legend()
fig.tight_layout(); fig.savefig(OUTPUT/'figures'/'pose_estimation_updrs_skewness.png',dpi=150); plt.show()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].bar(task_table.task, task_table.trajectories, color=['#2563eb','#0f766e','#7c3aed'])
axes[0].tick_params(axis='x', rotation=20)
axes[0].set(ylabel='Pose trajectories', title='Task representation in archive')
axes[1].hist(pose_lengths.duration_s, bins=30, color='#2563eb', edgecolor='white')
axes[1].axvline(pose_lengths.duration_s.median(),color='#dc2626',linestyle='--',
                label=f'Median {pose_lengths.duration_s.median():.1f} s')
axes[1].set(xlabel='Seconds at reported 30 FPS', ylabel='Trajectories', title='All pose trajectory durations')
axes[1].legend()
axes[2].boxplot([pose_lengths.loc[pose_lengths.task==task,'duration_s'] for task in task_counts.index],
                tick_labels=task_counts.index, patch_artist=True)
axes[2].tick_params(axis='x', rotation=20)
axes[2].set(ylabel='Seconds', title='Duration by task')
fig.tight_layout(); fig.savefig(OUTPUT/'figures'/'pose_estimation_distribution_skewness.png',dpi=150); plt.show()

ratio=task_counts.max()/task_counts.min()
pooled_skew=profile.loc[profile.scope=='All three pose tasks','skewness'].iloc[0]
display(Markdown(f"**Interpretation.** The {len(pose_lengths)} full-body pose trajectories have pooled duration skewness **{pooled_skew:.2f}**; compare each task row because task lengths can differ. The released UPDRS Total scores have skewness **{updrs_profile.loc[0,'skewness']:+.2f}** across {int(updrs_profile.loc[0,'n_trial_ids'])} trial IDs; gait item 3.10 has skewness **{updrs_profile.loc[1,'skewness']:+.2f}**. These are trial-linked ratings and may repeat participants, so they are not independent patient-level prevalence estimates. The largest pose task has {ratio:.2f} times as many trajectories as the smallest. Toe-tapping optical flow is excluded because it is not a full-body pose trajectory. The subject map contains {len(set(subjects.values()))} distinct IDs. Durations use the paper's 30 FPS capture rate and stored head-trajectory lengths."))
