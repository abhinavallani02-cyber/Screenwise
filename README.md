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

## Intended repository team

- abhinavallani02-cyber — repository owner
- Nehanth185 — intended collaborator
- Sreel712 — intended collaborator

Listing usernames here does not grant access or claim contributions. Collaborator invitations must be accepted on GitHub. Each person should contribute through their own authenticated account.
