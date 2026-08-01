# Real-Time Squat Posture Classification and Repetition Counting

This project implements a lightweight squat-monitoring pipeline that combines:

- **MediaPipe Pose** for markerless body-landmark detection;
- **engineered biomechanical features** for posture representation;
- **machine-learning classifiers** for six-class squat-posture classification; and
- a **state-based repetition counter** for completed squat repetitions.

The project is organised as three Jupyter notebooks covering dataset exploration, model training and evaluation, and real-time deployment.

> **Important:** This system is a research prototype and is not medical, physiotherapy, or professional coaching advice.

## Research Questions

1. Which classifier performs best for squat-posture classification using MediaPipe-derived pose features?
2. Can the selected classifier be integrated into a real-time system that predicts posture and counts squat repetitions?

## Project Pipeline

1. Download and inspect the Kaggle dataset.
2. Select 12 numerical posture features.
3. Split the data by source video to reduce leakage.
4. Compare Logistic Regression, Random Forest, and HistGradientBoosting.
5. Evaluate the models using held-out videos, grouped validation, and statistical tests.
6. Save the strongest model and its feature metadata.
7. Load the model in a MediaPipe/OpenCV application.
8. Predict posture and count repetitions from webcam or saved video.

## Repository Structure

```text
.
├── 01_downloading_and_exploring_dataset (3)(1).ipynb
├── 02_training_and_evaluating (4)(1).ipynb
├── 03_realtime_counter_and_posture_checker (1)(1).ipynb
├── data/
│   └── processed/
│       ├── manifest.json
│       └── schema_summary.csv
├── models/
│   └── squat_posture_classifier_improved.joblib
├── outputs/
│   ├── figures/
│   ├── classification_report.csv
│   ├── grouped_cross_validation_summary.csv
│   ├── model_comparison.csv
│   ├── model_metadata.json
│   ├── permutation_importance.csv
│   ├── random_forest_feature_importance.csv
│   ├── squat_session_features.csv
│   └── annotated_squat.mp4
└── results/
    └── per_video_holdout_results.csv
```

The folders and generated files are created when the corresponding notebook cells are executed.

## Dataset

The project uses the [Squat Exercise Pose Dataset](https://www.kaggle.com/datasets/thashmiladewmini/squat-exercise-pose-dataset).

The augmented CSV contains:

- **47,442** labelled observations derived from **15 source videos**;
- **six balanced numeric posture classes**, labelled `0` to `5`;
- **7,907 observations per class**; and
- **12 numerical pose features**:

```text
left_knee_angle
right_knee_angle
left_hip_angle
right_hip_angle
left_ankle_angle
right_ankle_angle
spine_angle
torso_lean
left_knee_lateral
right_knee_lateral
symmetry_score
hip_depth
```

The source-video identifier is retained only for grouped splitting and is not used as a predictive feature.

## Installation

Run the project from the repository root so that the relative `data`, `models`, `outputs`, and `results` paths are created in the correct location.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
pip install jupyter kagglehub numpy pandas matplotlib scikit-learn scipy joblib opencv-python mediapipe
```

Internet access is required when the dataset is downloaded for the first time. Kaggle authentication may be requested depending on the local environment.

Launch Jupyter:

```bash
jupyter notebook
```

## Running the Project

Execute the notebooks in this order.

### 1. Dataset Exploration

Open:

```text
01_downloading_and_exploring_dataset (3)(1).ipynb
```

This notebook:

- downloads the dataset through `kagglehub`;
- locates and loads the CSV files;
- inspects columns, types, missing values, and class distributions;
- creates descriptive plots; and
- saves `data/processed/schema_summary.csv` and `data/processed/manifest.json`.

### 2. Training and Evaluation

Open:

```text
02_training_and_evaluating (4)(1).ipynb
```

This notebook:

- loads `squat_features_augmented.csv`;
- removes the frame identifier from the predictors;
- uses the source video as the grouping variable;
- creates a leakage-aware train/test split;
- applies median imputation and feature standardisation inside a pipeline;
- compares three classifiers;
- generates confusion matrices and feature-importance analyses;
- performs grouped validation and statistical comparison; and
- saves the selected classifier and its metadata.

The training partition contains **40,266 observations from 12 videos**. The held-out test partition contains **7,176 observations from three videos**, with no video overlap.

### 3. Real-Time Monitoring

Open:

```text
03_realtime_counter_and_posture_checker (1)(1).ipynb
```

Notebook 02 must be completed first because Notebook 03 loads:

```text
models/squat_posture_classifier_improved.joblib
outputs/model_metadata.json
```

Configure the input source in Notebook 03:

```python
SOURCE = 0
```

Use `0` for the default webcam or provide a saved video path:

```python
SOURCE = "sample_data/sample_squat.mp4"
```

A saved recording is preferable for reproducible demonstrations.

Other useful settings are:

```python
SAVE_OUTPUT_VIDEO = False
DISPLAY_WINDOW = True
MIN_LANDMARK_VISIBILITY = 0.50
MIN_PREDICTION_CONFIDENCE = 0.60
```

Set `DISPLAY_WINDOW = False` in environments without desktop-window support. When the OpenCV window is enabled, press **q** to stop processing.

## Models

The following classifiers are compared:

| Model | Main configuration |
|---|---|
| Logistic Regression | `C=1.0`, balanced class weights, maximum 5,000 iterations |
| Random Forest | 700 trees, maximum depth 20, balanced-subsample weights |
| HistGradientBoosting | learning rate 0.08, 300 iterations, 31 leaf nodes, L2 regularisation 0.1 |

A most-frequent-class dummy classifier is also evaluated as a baseline.

## Evaluation

The main metrics are:

- accuracy;
- balanced accuracy;
- macro F1;
- weighted F1;
- class-level precision and recall; and
- confusion matrices.

Video-level grouping is used because observations from the same video are not independent. The statistical comparison uses Leave-One-Video-Out scores from the training videos:

1. a Friedman test compares all three classifiers;
2. paired Wilcoxon signed-rank tests compare model pairs; and
3. Holm correction controls for multiple comparisons.

The significance threshold is `alpha = 0.05`.

## Main Results

### Held-Out Test Set

| Classifier | Accuracy | Macro F1 |
|---|---:|---:|
| HistGradientBoosting | **0.6060** | **0.6138** |
| Logistic Regression | 0.4972 | 0.5086 |
| Random Forest | 0.5020 | 0.4813 |
| Most-frequent baseline | 0.1667 | 0.0476 |

HistGradientBoosting was selected for the real-time pipeline.

### Leave-One-Video-Out Statistical Comparison

| Classifier | Mean Macro F1 | Standard Deviation | Median |
|---|---:|---:|---:|
| HistGradientBoosting | **0.5641** | 0.2334 | **0.6125** |
| Random Forest | 0.4819 | 0.2073 | 0.5081 |
| Logistic Regression | 0.4545 | 0.1153 | 0.4399 |

The Friedman test found an overall model difference:

```text
Friedman statistic = 8.0000
p-value = 0.018316
```

After Holm correction:

- HistGradientBoosting significantly outperformed Random Forest (`p = 0.0103`);
- HistGradientBoosting did not significantly outperform Logistic Regression (`p = 0.1543`);
- Logistic Regression and Random Forest were not significantly different (`p = 0.5186`).

The large variation between videos indicates that performance depends strongly on the held-out recording.

## Repetition Counter

The repetition counter is independent of the posture classifier. It uses the mean knee angle and a two-state `up`/`down` state machine:

- five-frame median smoothing;
- down threshold: `105°`;
- up threshold: `165°`; and
- at least three consecutive down frames before entering the `down` state.

A repetition is counted only after a valid down phase followed by a return to the up threshold. The counter passed nine deterministic unit tests covering complete and incomplete repetitions, prolonged bottom holds, missing measurements, resets, and invalid angles.

## Generated Outputs

Notebook 02 generates, among other files:

```text
models/squat_posture_classifier_improved.joblib
outputs/model_metadata.json
outputs/model_comparison.csv
outputs/classification_report.csv
outputs/grouped_cross_validation_summary.csv
outputs/permutation_importance.csv
outputs/figures/model_comparison.png
outputs/figures/confusion_matrix_raw.png
outputs/figures/confusion_matrix_normalized.png
outputs/figures/permutation_importance.png
outputs/figures/learning_curve.png
results/per_video_holdout_results.csv
```

Notebook 03 generates:

```text
outputs/squat_session_features.csv
outputs/annotated_squat.mp4        # only when video saving is enabled
```

## Important Configuration and Limitations

### Numeric Class Labels

The dataset and trained model use class IDs `0` to `5`, but their verified human-readable meanings are not stored in Notebook 02. Before presenting posture feedback, update these objects in Notebook 03:

```python
CLASS_LABELS = {
    "0": "...",
    "1": "...",
    "2": "...",
    "3": "...",
    "4": "...",
    "5": "...",
}

POSTURE_MESSAGES = {
    "0": "...",
    "1": "...",
    "2": "...",
    "3": "...",
    "4": "...",
    "5": "...",
}

GOOD_POSTURE_CLASSES = {...}
```

Do not assume that class `0` represents correct posture without verifying the dataset label definitions.

### Feature Consistency

Notebook 03 validates that the live feature names match the model metadata. However, the original feature-generation script used to create the dataset CSV is not included. The live formulas should therefore be checked against the original dataset-generation procedure before claiming exact semantic equivalence.

### Real-Time Evaluation

The webcam run demonstrates technical integration, but it does not establish posture-classification or repetition-counting accuracy. A quantitative evaluation should use stored videos with manually annotated posture labels and repetition counts.

### Generalisation

The dataset contains only 15 source videos. Performance varied substantially between held-out videos, so the results should not be interpreted as evidence of reliable performance across all users, camera positions, or recording environments.

## Reproducibility

For reproducible experiments:

- run notebooks from the repository root;
- execute them in numerical order;
- keep `RANDOM_STATE = 42`;
- preserve the video-grouped split;
- do not use the held-out test set for model selection;
- save the model and metadata together; and
- prefer stored videos over webcam sessions for demonstrations and evaluation.

## Future Work

Possible extensions include:

- verifying and documenting all six class meanings;
- sharing the original feature-generation procedure;
- collecting more participants and source videos;
- evaluating repetition counts against manually annotated recordings;
- comparing the current models with a neural-network classifier; and
- improving the mapping from predicted classes to actionable feedback.
