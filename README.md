# Multimodal Seizure Prediction Using EEG Time-Series Features and Clinical Metadata

This repository contains the Spring 2026 CSCI 7090 project exploring EEG-based seizure prediction through two complementary tracks:

- reproducible preprocessing and feature extraction on benchmark EEG datasets
- a real-time EEG streaming pipeline for future online seizure-risk experiments

The broader research goal is to study how EEG time-series features and clinical metadata can be combined in a more standardized, reproducible seizure prediction workflow. The accompanying paper draft and LaTeX sources in this repository document the literature review, research gaps, and proposed multimodal pipeline.

## Why This Project

Epileptic seizure prediction remains difficult because published studies often vary in:

- preprocessing choices
- feature extraction methods
- validation protocols
- reporting metrics
- reliance on patient-specific data

This project is motivated by a simple question: how can seizure prediction be made more reproducible, comparable, and eventually more useful in real-world settings?

## Research Focus

The current project is centered on the following research questions distilled from the paper draft:

1. Which EEG features are most useful for seizure prediction, and how should they be extracted?
2. Where should clinical metadata be integrated into the modeling pipeline?
3. Which architectures are best suited for multimodal temporal seizure prediction?
4. How well can a workflow generalize across patients rather than only within a single patient?
5. How can inconsistent benchmarking and evaluation practices be reduced?

## What Is Implemented in This Repository

This repository currently provides the data-engineering side of the project rather than a finished prediction model.

### 1. Static EEG preprocessing workflow

The notebook at `eegStatic/notebooks/DataWrangling.ipynb` supports:

- EEG preprocessing and harmonization
- sliding-window epoch construction
- time-domain feature extraction
- outlier handling and exploratory analysis

Current milestone scope uses representative EDF files from each dataset to validate the pipeline before scaling to more recordings and explicit seizure interval labeling.

### 2. Real-time EEG dataset builder

The script `realtimeeeg.py` builds fixed-length EEG windows from a BrainFlow stream. It currently:

1. collects EEG chunks from BrainFlow
2. cleans channel names
3. converts signals to a target bipolar montage
4. resamples signals to 256 Hz
5. applies notch and bandpass filtering
6. maintains a rolling buffer
7. creates 30-second windows every 5 seconds
8. runs basic quality-control checks
9. z-score normalizes each channel
10. extracts lightweight statistical features
11. saves both the window and feature vector

By default, the script runs against the BrainFlow synthetic board so the pipeline can be tested without physical EEG hardware.

### 3. Paper and methodology sources

The research write-up is maintained in:

- `main.tex`
- `references.bib`
- `main.pdf`

These files capture the literature review, methodological framing, and the proposed multimodal seizure prediction pipeline that motivates the code in this repository.

## Proposed End-to-End Pipeline

The draft paper frames the intended full workflow as:

1. collect raw multi-channel EEG and clinical metadata
2. preprocess EEG and clean/encode metadata
3. extract EEG features or learned signal representations
4. represent metadata as structured model inputs
5. fuse EEG-derived features with clinical context
6. train seizure prediction models
7. evaluate with patient-specific and cross-patient protocols
8. report metrics in a more standardized way

The current repository mainly covers steps 1 through 3 for EEG data preparation, along with the project documentation for the later modeling stages.

## Repository Layout

```text
.
|-- README.md
|-- requirements.txt
|-- realtimeeeg.py
|-- realTimeLabeling.ipynb
|-- main.tex
|-- main.pdf
|-- references.bib
`-- eegStatic/
    |-- notebooks/
    |   `-- DataWrangling.ipynb
    `-- data/
        |-- chbmit/
        |   `-- chb01/
        |       |-- chb01-summary.txt
        |       `-- chb01_01.edf
        `-- siena/
            `-- pn00/
                |-- PN00-1.edf
                `-- Seizures-list-PN00.txt
```

## Datasets

This repository currently includes data from:

- CHB-MIT Scalp EEG Database
- Siena Scalp EEG Database

It also supports real-time or simulated streaming acquisition through BrainFlow.

### Static data

Static benchmark data is stored under:

- `eegStatic/data/chbmit/`
- `eegStatic/data/siena/`

These files are used for preprocessing, feature engineering, and early-stage seizure prediction experiments.

### Real-time data

`realtimeeeg.py` uses BrainFlow to stream EEG data from supported devices. In the current default configuration, it uses the synthetic board for safe local testing.

## Environment Setup

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

### 3. Main packages

- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `seaborn`
- `mne`
- `brainflow`

## Running the Static Workflow

Launch Jupyter and open the notebook:

```bash
jupyter lab
```

Primary notebook:

- `eegStatic/notebooks/DataWrangling.ipynb`

### Dataset setup for local and Colab runs

The notebook expects a dataset root that directly contains both `chbmit/` and `siena/`.
The required structure is:

```text
<dataset root>/
  chbmit/chb01/chb01_01.edf
  siena/pn00/PN00-1.edf
```

Local execution works when the dataset root is either:

- `eegStatic/data/` inside this repository
- a separate top-level `data/` directory that contains `chbmit/` and `siena/`

Colab execution works when the dataset is:

- uploaded into `/content/data/`
- uploaded into `/content/eegStatic/data/`
- stored anywhere in mounted Google Drive where the expected EDF files exist
- pointed to explicitly with `EEG_DATA_ROOT`

Important detail:

- `EEG_DATA_ROOT` must point to the folder that directly contains `chbmit/` and `siena/`, not to one of those subfolders

Local example using the repository layout:

```text
eeg-Spr2026-CSCI7090/
  eegStatic/
    notebooks/DataWrangling.ipynb
    data/
      chbmit/chb01/chb01_01.edf
      siena/pn00/PN00-1.edf
```

Colab example using Google Drive:

```python
from google.colab import drive
import os

drive.mount("/content/drive")
os.environ["EEG_DATA_ROOT"] = "/content/drive/MyDrive/path/to/eegStatic/data"
```

After mounting Drive or changing `EEG_DATA_ROOT`, rerun the notebook cell that resolves dataset paths.

The notebook first checks common local and Colab locations such as `eegStatic/data`, `data`, `/content/data`, and `/content/eegStatic/data`. If those do not match, it falls back to a recursive search of mounted Google Drive for `chb01_01.edf` and `PN00-1.edf`.

Recommended workflow:

1. run notebook cells from top to bottom
2. validate preprocessing on the included EDF examples
3. inspect extracted features and EDA outputs
4. extend to additional recordings and seizure labels

## Running the Real-Time Pipeline

Run the streaming dataset builder:

```bash
python3 realtimeeeg.py
```

Default behavior:

- uses the BrainFlow synthetic board
- polls incoming data in short chunks
- saves a processed EEG window every 5 seconds once enough data is buffered

### Output artifacts

Generated files are written to:

- `live_eeg_dataset/windows/`
- `live_eeg_dataset/manifest.csv`

Saved artifacts include:

- compressed EEG windows as `.npz`
- feature vectors as `.npy`
- a manifest row with timing, channel order, QC status, and file paths

## Current Status

- Completed: baseline static EEG preprocessing and wrangling workflow
- Completed: prototype real-time EEG windowing and feature extraction pipeline
- In progress: seizure labeling integration and supervised modeling workflow
- In progress: stronger evaluation design for patient-specific versus cross-patient testing
- Planned: multimodal fusion of EEG features with clinical metadata

## Limitations

At the current stage, this repository does not yet provide:

- a finalized multimodal fusion model
- a complete labeled training set spanning many patients
- standardized benchmark reporting across multiple experiments
- clinical deployment or medical decision support

The implemented code should be treated as a research pipeline for data preparation and experimentation.

## Contributors

- Luke Blevins
- Jacob Rawlins
- Arnoldo Vilches-Arteaga
