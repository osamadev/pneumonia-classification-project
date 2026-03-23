# Pneumonia Detection from Chest X-Rays

This project builds and evaluates deep learning models for binary pneumonia classification (`NORMAL` vs `PNEUMONIA`) using chest X-ray images.

It includes:
- A notebook pipeline for EDA, CNN baselines, and transfer learning
- Shared reusable utilities under `helpers/`
- A Streamlit app for interactive inference on saved models

## Project Structure

- `01_Pneumonia_Detection_EDA.ipynb` - exploratory data analysis and dataset checks
- `02_Pneumonia_Detection_CNN_Models.ipynb` - baseline CNN training, tuning, and evaluation
- `03_Pneumonia_Detection_Transfer_Learning.ipynb` - VGG16 transfer learning and fine-tuning
- `helpers/` - shared data, model, training, and visualization utilities
- `saved_models/` - exported `.keras` models and `_meta.json` sidecar files
- `app.py` - Streamlit inference app

## Dataset Layout

Expected local dataset layout:

```text
dataset/
  train/
    NORMAL/
    PNEUMONIA/
  val/
    NORMAL/
    PNEUMONIA/
  test/
    NORMAL/
    PNEUMONIA/
```

Notes:
- The helper pipeline re-splits training data for a larger validation split.
- Labels are mapped as: `0 = NORMAL`, `1 = PNEUMONIA`.

## Environment Setup

Create and activate your own virtual environment, then install dependencies:

```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Notebook Workflow (Recommended Order)

1. Run `01_Pneumonia_Detection_EDA.ipynb`
2. Run `02_Pneumonia_Detection_CNN_Models.ipynb`
3. Run `03_Pneumonia_Detection_Transfer_Learning.ipynb`

Why this order:
- EDA validates data and assumptions
- CNN notebook builds baseline and tuned references
- Transfer learning notebook applies VGG16 strategy (partial re-training of the pre-trained model).

## Saving and Loading Models

Notebooks save models with:
- Model file: `<name>.keras`
- Metadata file: `<name>_meta.json` (threshold, metrics, hyperparameters)

The Streamlit app auto-discovers models from `saved_models/` using these pairs.

## Run the Streamlit App

From project root:

```bash
streamlit run app.py
```

Then open the local URL shown in terminal (usually `http://localhost:8501`).

Examples:

<img width="1182" height="538" alt="Screenshot 2026-03-23 at 09 16 05" src="https://github.com/user-attachments/assets/c5992e18-3e65-4495-86ce-0fb70c97bb20" />

<img width="1095" height="539" alt="Screenshot 2026-03-23 at 09 16 26" src="https://github.com/user-attachments/assets/b8951780-f21f-42a1-b97b-2ec7520075c9" />


## Using the Streamlit App

1. Select a model from the sidebar
2. Upload a chest X-ray image (`jpg`, `jpeg`, `png`, `bmp`, `webp`)
3. Review:
   - Predicted class (`NORMAL` or `PNEUMONIA`)
   - Probability `P(PNEUMONIA)`
   - Decision threshold from model metadata
   - Confidence and model metrics

## Common Troubleshooting

- **No models found in app**: Run notebook save cells first so `saved_models/` contains `.keras` and `_meta.json` files.
- **Import/runtime errors**: Reinstall dependencies with `pip install -r requirements.txt`.
- **Wrong dataset path**: Verify `DATASET_ROOT` settings in notebooks for local/Colab/Kaggle environments.
