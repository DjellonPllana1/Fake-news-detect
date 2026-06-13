# Fake News Classification

Ky projekt perdor datasetet `Fake.csv` dhe `True.csv` per te trajnuar modele qe klasifikojne lajmet si `fake` ose `true`.

## Datasetet

- `Fake.csv` - lajme te rreme, me label `fake`.
- `True.csv` - lajme te verteta, me label `true`.

Te dy datasetet bashkohen ne nje dataset te vetem. Pastaj hiqen duplikatet, kombinohen kolonat `title`, `subject` dhe `text` ne kolonen `content`, dhe krijohet target-i numerik:

- `fake = 0`
- `true = 1`

Per trajnim perdoret nje subset i balancuar me `7000` shembuj nga secila klase.

## Struktura e kodit

- `common.py` - ngarkimi i datasetit, preprocessing, ndarja train/validation/test, TF-IDF, metrikat dhe ruajtja e rezultateve.
- `naive_bayes_model.py` - trajnimi dhe tuning per Multinomial Naive Bayes.
- `logistic_regression_model.py` - trajnimi dhe tuning per Logistic Regression.
- `svm_model.py` - trajnimi dhe tuning per Linear SVM.
- `train_news_models.py` - ekzekuton te gjitha modelet dhe krijon tabelen krahasuese.

## Modelet

- Multinomial Naive Bayes
- Logistic Regression
- Linear SVM

Teksti kthehet ne vecori numerike me `TfidfVectorizer`.

## Instalimi

```powershell
pip install -r requirements.txt
```

## Ekzekutimi

```powershell
python train_news_models.py
```

Rezultatet ruhen automatikisht ne:

- `results/model_comparison.csv`
- `results/hyperparameter_tuning.csv`
- `results/classification_report_*.csv`
- `figures/confusion_matrix_*.png`
