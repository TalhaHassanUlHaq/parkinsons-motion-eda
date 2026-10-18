"""MMU distribution profile; embedded in the generated notebook."""
from IPython.display import display, Markdown

order = ['NORMAL', 'MILD', 'MODERATE', 'SEVERE']
class_table = pd.DataFrame({'folder_label': order, 'recordings': [counts.get(label, 0) for label in order]})
class_table['share_pct'] = 100 * class_table.recordings / class_table.recordings.sum()
class_table.insert(0, 'display_category', [1,2,3,4])
display(pd.DataFrame([{'recordings':int(class_table.recordings.sum()), 'participants':'Unknown',
                       'source_FPS':'Unknown', 'pose_dimension':'2D pixels',
                       'labels':'Normal, Mild, Moderate, Severe (dataset classes, not UPDRS)', 'count_scope':count_scope}]))
display(class_table.round(1))
class_table.to_csv(OUTPUT/'tables'/'mmu_label_distribution.csv', index=False)
class_table.to_csv(OUTPUT/'tables'/'mmu_score_categories.csv', index=False)

fig, ax = plt.subplots(figsize=(7,4))
bars=ax.bar(class_table.display_category, class_table.recordings,
            color=['#64748b','#60a5fa','#2563eb','#1e3a8a'])
ax.set_xticks([1,2,3,4], [f'{i}\n{label.title()}' for i,label in zip(class_table.display_category,class_table.folder_label)])
ax.set(xlabel='Display category (source label; not UPDRS)', ylabel='JSON recordings',
       title='MMU source categories 1–4')
for bar, count in zip(bars,class_table.recordings):
    ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+2,str(count),ha='center')
ax.set_ylim(0,class_table.recordings.max()*1.17)
fig.tight_layout();fig.savefig(OUTPUT/'figures'/'mmu_score_categories.png',dpi=150);plt.show()

frame_lengths = pd.Series(overview.frames.to_numpy(), dtype=float)
selected_skew = frame_lengths.skew() if frame_lengths.nunique() > 1 else float('nan')
profile = pd.DataFrame([
    {'measure':'folder label counts', 'scope':count_scope, 'n':int(class_table.recordings.sum()),
     'skewness':float('nan'), 'note':'Categorical: use class shares and imbalance ratio'},
    {'measure':'recording frame count', 'scope':'three deliberately selected recordings only', 'n':len(frame_lengths),
     'skewness':selected_skew, 'note':'Selection estimate only; source FPS and full-population length skew unavailable'}
])
display(profile)
profile.to_csv(OUTPUT/'tables'/'mmu_distribution_profile.csv', index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
bars = axes[0].bar(class_table.folder_label, class_table.recordings, color=['#64748b','#60a5fa','#2563eb','#1e3a8a'])
for bar, share in zip(bars, class_table.share_pct):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+2, f'{share:.1f}%', ha='center', fontsize=9)
axes[0].set(ylabel='JSON recordings', title='All labeled MMU recordings')
axes[0].set_ylim(0, class_table.recordings.max()*1.18)
axes[1].bar(range(len(overview)), overview.frames, color='#0f766e')
axes[1].set_xticks(range(len(overview)), [f'{label} sample' for label in overview.label], rotation=20, ha='right')
axes[1].set(ylabel='Recorded pose frames', title='Three selected recordings; duration unknown')
fig.tight_layout(); fig.savefig(OUTPUT/'figures'/'mmu_distribution_skewness.png', dpi=150); plt.show()

positive = class_table[class_table.recordings > 0].recordings
ratio = positive.max()/positive.min()
display(Markdown(f"**Interpretation.** The largest folder contributes {class_table.share_pct.max():.1f}% of the {int(class_table.recordings.sum())} labeled JSON files; its count is {ratio:.2f} times the smallest folder. This is class imbalance, not numerical skewness of severity. The selected frame-count skew is {selected_skew:.2f} across only three chosen files, so it does not estimate the archive's recording-length distribution. Source FPS is unavailable; video playback duration must not be treated as capture duration."))
