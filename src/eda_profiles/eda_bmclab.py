"""BMCLab distribution profile; embedded in the generated notebook."""
from IPython.display import display, Markdown

subject_counts = inventory.groupby('subject').size().rename('C3D trials')
medication = inventory.medication.value_counts().reindex(['ON','OFF'], fill_value=0)
trial_types = inventory.trial_type.value_counts()
selected_durations = pd.Series(overview.duration_s.dropna().to_numpy(), dtype=float)
on_scores = pd.to_numeric(meta['ON - UPDRS-III'], errors='coerce').dropna()
off_scores = pd.to_numeric(meta['OFF - UPDRS-III'], errors='coerce').dropna()
display(pd.DataFrame([{'C3D_recordings':len(inventory), 'participants':subject_counts.size,
                       'source_FPS':'150 Hz in the three selected headers',
                       'pose_dimension':'measured 3D metres',
                       'labels':'ON/OFF; FoG group and UPDRS-III at participant level'}]))

# The full local spreadsheet contains the actual 0–4 walking item. The
# approved two-subject Kaggle input contains totals only, so skip there.
walking_columns={'ON':'ON - UPDRS-III - walking','OFF':'OFF - UPDRS-III - walking'}
if all(col in meta.columns for col in walking_columns.values()):
    walking_counts=pd.DataFrame({'score':range(5)})
    for medication_label,column in walking_columns.items():
        values=pd.to_numeric(meta[column],errors='coerce').dropna()
        assert values.between(0,4).all() and values.eq(values.round()).all(), column
        walking_counts[medication_label]=values.astype(int).value_counts().reindex(range(5),fill_value=0).values
    display(walking_counts)
    walking_counts.to_csv(OUTPUT/'tables'/'bmclab_score_categories.csv',index=False)
    fig,ax=plt.subplots(figsize=(7,4))
    x=np.arange(5);width=.38
    ax.bar(x-width/2,walking_counts.ON,width,label='ON',color='#0f766e')
    ax.bar(x+width/2,walking_counts.OFF,width,label='OFF',color='#ea580c')
    ax.set_xticks(x,[str(i) for i in x])
    ax.set(xlabel='Recorded UPDRS-III walking item score',ylabel='Participants',
           title='BMCLab UPDRS gait item by medication')
    ax.legend();fig.tight_layout()
    fig.savefig(OUTPUT/'figures'/'bmclab_score_categories.png',dpi=150);plt.show()
    display(Markdown('Scores **1–4** are shown as recorded; **0** is also shown because it occurs in the source. Counts are participant ratings, not numbers of C3D trials.'))
else:
    display(Markdown('The approved private Kaggle sample contains UPDRS-III **totals** but not the 0–4 walking-item columns. Its gait-category bar chart is available only in the full local-data execution; totals cannot be relabeled as 1–4 gait scores.'))

def numeric_profile(name, values, scope, units):
    values = pd.Series(values, dtype=float).dropna()
    skew = values.skew() if len(values) >= 3 and values.nunique() > 1 else float('nan')
    return {'measure':name, 'scope':scope, 'n':len(values), 'units':units,
            'median':values.median() if len(values) else float('nan'),
            'p90':values.quantile(.9) if len(values) else float('nan'), 'skewness':skew}

profile = pd.DataFrame([
    numeric_profile('C3D trials per participant', subject_counts, 'full 935-file archive inventory', 'trials'),
    numeric_profile('selected walk duration', selected_durations, 'three selected C3Ds only', 'seconds'),
    numeric_profile('ON UPDRS-III', on_scores, f'{meta.ID.nunique()} metadata participants available here', 'score'),
    numeric_profile('OFF UPDRS-III', off_scores, f'{meta.ID.nunique()} metadata participants available here', 'score')
])
display(profile.round(2))
profile.to_csv(OUTPUT/'tables'/'bmclab_distribution_profile.csv', index=False)
display(pd.DataFrame({'medication':medication.index, 'C3D_files':medication.values,
                      'share_pct':(100*medication/medication.sum()).round(1).values}))

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].hist(subject_counts, bins=range(int(subject_counts.min()),int(subject_counts.max())+2),
             color='#2563eb', edgecolor='white')
axes[0].axvline(subject_counts.median(), color='#dc2626', linestyle='--', label=f'Median {subject_counts.median():.0f}')
axes[0].set(xlabel='C3D trials per participant', ylabel='Participants', title='Full archive trial-count distribution')
axes[0].legend()
axes[1].bar(medication.index, medication.values, color=['#0f766e','#ea580c'])
axes[1].set(ylabel='C3D files', title='Archive medication conditions')
axes[2].bar(range(len(overview)), overview.duration_s, color='#7c3aed')
axes[2].set_xticks(range(len(motions)),
                   [f"{m.subject} {m.label} #{m.source.split('_')[-1].split('.')[0]}" for m in motions],
                   rotation=20, ha='right')
axes[2].set(ylabel='Seconds at measured 150 Hz', title='Selected walking durations only')
fig.tight_layout(); fig.savefig(OUTPUT/'figures'/'bmclab_distribution_skewness.png', dpi=150); plt.show()

trial_skew = profile.loc[profile.measure=='C3D trials per participant','skewness'].iloc[0]
med_ratio = medication.max()/medication[medication>0].min()
score_scope = 'Full local metadata' if meta.ID.nunique()>2 else 'Private Kaggle metadata for two selected participants'
display(Markdown(f"**Interpretation.** The full inventory has {len(inventory)} C3Ds from {subject_counts.size} participants, including {trial_types.get('walk',0)} walks and {trial_types.get('static',0)} static trials. Per-participant trial-count skewness is **{trial_skew:.2f}**; this measures how recording volume differs by participant, not gait severity. ON/OFF file counts have a {med_ratio:.2f}:1 imbalance ratio, but repeated trials from the same person are not independent participants. {score_scope} supplies {meta.ID.nunique()} participant rows; score skewness is left undefined when fewer than three usable scores are present. Only the three selected C3Ds have measured durations in this notebook."))
