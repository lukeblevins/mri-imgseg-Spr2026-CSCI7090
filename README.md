# Multimodal Seizure Prediction Using EEG Time-Series Features and Clinical Metadata

This repository is organized around the research workflow instead of a single `eegStatic/` subtree. It currently covers two active tracks:

- benchmark EEG preprocessing and feature extraction
- realtime EEG window generation and downstream labeling

The paper draft in `paper/` captures the literature review, research gaps, and the proposed multimodal seizure prediction pipeline that motivates the codebase.

## Repository Layout

```text
.
|-- README.md
|-- requirements.txt
|-- pyproject.toml
|-- src/eeg_project/
|-- notebooks/
|   |-- static/data_wrangling.ipynb
|   `-- realtime/labeling.ipynb
|-- scripts/
|   `-- run_realtime_builder.py
|-- paper/
|   |-- main.tex
|   |-- references.bib
|   `-- main.pdf
|-- data/
|   `-- README.md
`-- artifacts/
    `-- README.md
```

## Workflow Areas

### Static benchmark workflow

Reusable static EEG logic lives under `src/eeg_project/static/`. The notebook at `notebooks/static/data_wrangling.ipynb` is now a thin entrypoint that:

- resolves the dataset root
- preprocesses CHB-MIT and Siena example recordings
- constructs fixed-length epochs
- extracts basic time-domain features

### Realtime workflow

Realtime acquisition and labeling code lives under `src/eeg_project/realtime/`.

- `scripts/run_realtime_builder.py` streams BrainFlow EEG data and writes normalized windows plus features
- `notebooks/realtime/labeling.ipynb` labels saved windows from seizure annotations and prepares a training manifest

## Dataset Contract

Benchmark data is intentionally externalized from version control.

Set `EEG_DATA_ROOT` to a directory that directly contains:

```text
<dataset root>/
  chbmit/chb01/chb01_01.edf
  siena/pn00/PN00-1.edf
```

If `EEG_DATA_ROOT` is unset, the code falls back to the ignored local `./data/` directory. Colab-compatible search locations under `/content/data` and mounted Google Drive are also supported.

## Realtime Output Contract

Generated realtime artifacts go to:

- `EEG_OUTPUT_ROOT`, when set
- otherwise `artifacts/realtime/`

Expected files inside that output root:

- `manifest.csv`
- `seizure_annotations.csv`
- `labeled_manifest.csv`
- `training_manifest.csv`
- `windows/`

## Environment Setup

Create a virtual environment and install the project in editable mode:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

Main dependencies:

- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `seaborn`
- `mne`
- `brainflow`

## Running The Workflows

Open the static notebook:

```bash
jupyter lab notebooks/static/data_wrangling.ipynb
```

Run the realtime dataset builder:

```bash
python scripts/run_realtime_builder.py
```

Then open the labeling notebook:

```bash
jupyter lab notebooks/realtime/labeling.ipynb
```

## Paper Sources

Paper assets live in `paper/`. Build from that directory with your preferred LaTeX toolchain, for example:

```bash
cd paper
latexmk -pdf main.tex
```
