# Part 2 — Nehanth185: Model, baseline and error evaluation

Open one shared Google Colab notebook. Add three CODE cells. Nehanth185 should paste the complete code below into cell 2. Run cells 1, 2 and 3 in that order in the same runtime. Do not paste Markdown fences or document instructions into a code cell. Use a standard Python CPU runtime; the normal Colab libraries are sufficient. An internet connection is required to download the public dataset. Each participant should understand and adapt their assigned part; follow the competition’s AI-assistance rules.

```python
# PART 2 — Nehanth185: model, baseline and error evaluation
# Paste into the SECOND code cell. Part 1 must have run first.
assert 'X_train' in globals(), 'Run Part 1 first.'
model = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=42))
# All preprocessing learns from training data only.
model.fit(X_train, y_train)
scores = model.predict_proba(X_test)[:, 1]
majority_label = int(y_train.mode().iloc[0])
baseline = np.full(len(y_test), majority_label)
THRESHOLD = 0.35  # Illustrative, not clinically chosen or validated.

def metrics(name, predictions):
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    return {'Method': name, 'Accuracy (%)': round(100*(tp+tn)/len(y_test), 2),
            'True positives': int(tp), 'True negatives': int(tn),
            'Missed cases (FN)': int(fn), 'False alarms (FP)': int(fp),
            'FN rate (%)': round(100*fn/(tp+fn), 2),
            'FP rate (%)': round(100*fp/(tn+fp), 2)}

predictions = (scores >= THRESHOLD).astype(int)
results = pd.DataFrame([metrics('Always-negative baseline', baseline),
                        metrics('Logistic regression', predictions)])
display(results.set_index('Method'))
print(f'Model ROC AUC: {roc_auc_score(y_test, scores):.4f}; constant baseline AUC: 0.5000')
print('FN rate uses all actual positives; FP rate uses all actual negatives.')
print('Accuracy alone hides the 67 positives missed by the baseline.')

evidence = X_test.copy()
evidence.insert(0, 'record_number', test_ids + 1)
evidence['actual_label'] = y_test.to_numpy()
evidence['model_score'] = scores
evidence['predicted_label'] = predictions
errors = evidence[evidence.actual_label != evidence.predicted_label]
print('\nA real mistake (1 means positive):')
example = errors[errors.actual_label == 1].iloc[0]
display(example.to_frame('Value'))
print('This is a missed positive case at threshold 0.35.')
print('Four measurements overlap between classes. Missing values or unmeasured')
print('health factors may contribute; this record does not prove the cause.')

fig, ax = plt.subplots(figsize=(5, 4))
cm = confusion_matrix(y_test, predictions, labels=[0, 1])
ax.imshow(cm, cmap='Greens')
for row in range(2):
    for col in range(2):
        ax.text(col, row, str(cm[row, col]), ha='center', va='center', fontsize=20)
ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['No flag', 'Flag'],
       yticklabels=['Negative', 'Positive'], xlabel='Model prediction',
       ylabel='Actual dataset label', title='Held-out test set: threshold 0.35')
plt.tight_layout()
plt.show()
print('Part 2 complete. Run Part 3 next.')
```
