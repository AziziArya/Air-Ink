# Air-Ink image baseline experiments

This is an additive ML scaffold. It does not replace `tracking_mvp.py` or the existing `train_model.py`.

## Run in the Python 3.8 ML environment (PowerShell)

```powershell
& .\.venv_ml\Scripts\Activate.ps1
python -m pip install scikit-image matplotlib joblib
python run_experiments.py
```

Expected input: `dataset_manual\0` through `dataset_manual\9`, each containing PNG images.

The runner validates the dataset, compares HOG + SVM / Logistic Regression / Random Forest, then trains three CNN variants. It creates outputs only under `experiments\image_baselines`. Existing project files and datasets are not overwritten.

Training defaults: 120-epoch ceiling, early stopping (patience 15), LR reduction, batch size 16. The ceiling is not a target; training stops when validation loss stops improving. Best weights are checkpointed as `best.weights.h5`; final model is saved in HDF5 format to avoid the native-Keras checkpoint serialization issue seen in the prior run.

**Interpretation limit:** with 20 samples per class from one writer, results are a pipeline smoke test / within-writer baseline only. They do not establish generalization to unseen writers. Keep the test set untouched; collect multi-writer data and use writer-disjoint splits for research claims.

## Important
- Augmentation is applied inside the CNN and therefore only during training.
- No horizontal flip is used because it can change digit identity.
- Synthetic-font samples are not mixed into the manual dataset in this first comparison.
- This image experiment is a supporting baseline; the main Air-Ink question remains Touch-to-Air domain gap and trajectory-based recognition.
