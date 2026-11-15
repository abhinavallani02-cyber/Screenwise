# Screenwise

An educational AI diabetes screening prototype for Ana Martinez, a fictional community health nurse in Phoenix, Arizona. It explores how routine measurements could help prioritize screening follow-up reviews.

**This project is a prototype and is not medical advice.**

## Run the demo

No build or API key is required:

```sh
python -m http.server 8000 --directory dist
```

Open http://localhost:8000. Select a held-out public record, inspect the model result, and change the threshold to explore missed cases and false alarms. No private patient records are collected.

## Results

On a stratified held-out set of 192 public records, at the illustrative threshold of 0.35:

| Method | Accuracy | False negatives | False positives |
| --- | --- | --- | --- |
| Always-negative baseline | 65.1% | 67 | 0 |
| Logistic regression | 72.4% | 24 | 29 |

The model catches 43 of 67 positive cases. Its ROC AUC is 0.8086. Threshold exploration on this same test set is not independent validation.

## Real example and failure

PUBLIC-636 has glucose 104 mg/dL, diastolic blood pressure 72 mm Hg, BMI 31.2 kg/m², and age 38 years. Its model score is 0.177405, below the flag threshold, but its dataset label is positive: a false negative. Overlapping measurements and omitted health factors may contribute; the cause cannot be established from this record.

## Original research question

Does the model miss more positive cases among younger or older participants? At the default threshold, it misses 20/45 positives aged 21–39 (44.4%) and 4/22 positives aged 40+ (18.2%). Small subgroups and this historical dataset limit interpretation; these findings do not establish fairness or causation.

## Public dataset and reproducibility

Uses only the [public Pima Indians Diabetes CSV mirror](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv), originally attributed to NIDDK. Its 768 records cover women aged 21+ of Pima heritage. It does not establish performance for other populations or modern clinical practice.

```sh
python -m pip install -r requirements.txt
python train.py
```

Training downloads the CSV if absent and writes `dist/model.json`. The split is stratified, 75% training and 25% testing, seed 42. Zero glucose, blood pressure, and BMI measurements are treated as missing. Median imputation and standardization are fitted only on training records, followed by logistic regression. The target is the existing dataset diabetes label, not future disease onset.

See `dist/report.txt` for the full project write-up, dataset checksum, methods, limitations, and next steps. Browser-calculated scores were verified against all 192 Python test predictions; browser layout testing has not been completed.

## Colab submission and final demonstration

The [three-part Colab notebook](colab/Screenwise_Team.ipynb) contains the full pipeline. Run its code cells in order; no GPU or API key is required. [Open in Colab](https://colab.research.google.com/github/abhinavallani02-cyber/Screenwise/blob/main/colab/Screenwise_Team.ipynb) and save a copy in Drive. The repository copy defaults to 0.35; our uploaded team notebook was also run at 0.15.

The saved team notebook outputs show the following exploratory comparison on the same 192 test records:

| Threshold | Accuracy | Positive cases caught | Missed cases (FN) | False alarms (FP) |
| --- | --- | --- | --- | --- |
| 0.35 | 72.40% | 43 / 67 | 24 / 67 (35.82%) | 29 / 125 (23.2%) |
| 0.15 | 65.62% | 64 / 67 | 3 / 67 (4.48%) | 63 / 125 (50.4%) |

Record 636 is missed at 0.35 but flagged at 0.15. Lowering the threshold catches more positive cases while increasing false alarms; it does not improve every metric. At 0.15, ages 21–39 have 3/45 positives missed (6.7%), and ages 40+ have 0/22 missed. These small exploratory subgroup results do not demonstrate clinical reliability. Neither threshold is clinically recommended.

## What we added to the Health challenge

We designed the prototype around one nurse's end-of-session review task, used four routine measurements, compared a trained classifier with an always-negative baseline, made threshold trade-offs visible, demonstrated an actual missed case, and investigated false-negative rates by age group. We used an existing public dataset, not a newly collected or local dataset. Our specific user scenario, threshold demonstration and subgroup question are our project choices within the supplied challenge.

## Credits and AI assistance

- **Dataset:** [Pima Indians Diabetes public CSV mirror](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv), hosted in [Jason Brownlee's Datasets repository](https://github.com/jbrownlee/Datasets); the original dataset is attributed to the National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK).
- **Model:** [scikit-learn LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), trained from scratch on the training split. No pretrained model or pretrained weights were used for predictions.
- **Libraries:** [NumPy](https://numpy.org/), [pandas](https://pandas.pydata.org/), [Matplotlib](https://matplotlib.org/), [IPython](https://ipython.org/) and [scikit-learn](https://scikit-learn.org/).
- **AI tools:** [OpenAI ChatGPT](https://chatgpt.com/) and [Codex](https://openai.com/codex/) substantially assisted with the project concept, code generation, model evaluation, debugging, notebook/document preparation, website and write-up. The implementation was AI-assisted, not entirely hand-written by the team.
- **Platform tools:** [Google Colab](https://colab.research.google.com/) for running and sharing the notebook; [GitHub](https://github.com/) for source hosting. OpenAI's Sites tooling was used to prepare the earlier web prototype.
- **Tutorials:** No tutorial code was directly copied during this build; the dataset mirror above was used as the data source. Any additional resources used separately by team members must also be credited before submission.

## Intended repository team

- abhinavallani02-cyber — repository owner
- Nehanth185 — intended collaborator
- Sreel712 — intended collaborator

Listing usernames here does not grant access or claim contributions. Collaborator invitations must be accepted on GitHub. Each person should contribute through their own authenticated account.
