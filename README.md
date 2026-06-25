# real-time-bp

A DVC-backed analysis of near real-time radio backpropagation pipeline for ionospheric scintillation mitigation and monitoring.

The project consumes simulated GNSS L1 tracking outputs, estimates the Fresnel
time scale used by radio backpropagation, detects cycle slips with weighted
total-variation denoising, evaluates batch and sliding-window backpropagation,
and renders the CSV tables and figures used by the paper workflow.

## Pipeline Overview

The executable workflow is defined in `dvc.yaml` and configured by
`params.yaml`.

| Stage | Command module | Main outputs |
| --- | --- | --- |
| `get_dataset` | `real_time_bp.scripts.get_dataset` | `data/raw/backpropagation` |
| `dataset_summary` | `real_time_bp.scripts.get_dataset_summary` | `outputs/01_dataset` |
| `filter_bank_results` | `real_time_bp.scripts.get_filter_bank_results` | `outputs/03_filter_bank` |
| `sliding_results` | `real_time_bp.scripts.get_sliding_results` | `outputs/04_sliding/sliding_var_orders.csv`, `sliding_practical_metrics.csv`, `sliding_forecast_example.csv` |
| `sliding_heatmap` | `real_time_bp.scripts.get_sliding_heatmap` | `outputs/04_sliding/sliding_latency_heatmap.csv` |
| `representative_results` | `real_time_bp.scripts.get_representative_results` | TVD, filter-bank, and sliding example CSVs |
| `result_plots` | `real_time_bp.plots.main` | `results/figures`, `results/csv_tables` |

The main numerical methods live in `real_time_bp/methodology`:

- `backpropagation.py`: propagation kernels, batch BP, sliding block BP, and phase alignment.
- `filter_bank.py`: amplitude standard deviation Fresnel time scaling scoring.
- `tvd.py`: fade-constrained weighted TVD cycle-slip detection.
- `sliding.py`: VAR order selection and VAR-filled sliding-window BP.
- `signal_model.py`, `metrics.py`, and `constants.py`: shared signal reconstruction, metrics, and unit conversions.

## Data

By default, the raw dataset is downloaded from Kaggle using the handle in
`params.yaml`:

```yaml
dataset:
  handle: rodrigodlf/backpropagation
  output_dir: data/raw/backpropagation
```

The pipeline expects a `tracking_outputs` layout with:

- `tracking_outputs/diagnostics.csv`
- `tracking_outputs/tracking_measurements_<severity>.csv`
- `tracking_outputs/meta.json`

Measurement CSVs contain one or more tracking cases with columns such as
`tracking_case_id`, `epoch_idx`, `time_sec`, `tracked_intensity`,
`estimated_scintillation_phase_m`, `scint_phase_m`,
`true_refractive_phase_m`, and `true_intensity`. Case metadata such as
severity, C/N0, Doppler initialization error, loss-of-lock status, and true
Fresnel time scale comes from `diagnostics.csv`.

`data/`, `outputs/`, and `results/` are generated artifacts and are ignored by
Git. DVC tracks the dependency graph and output checksums.

## Setup

This project targets Python 3.12 and uses `uv` with the committed `uv.lock`.

On Linux, install the native libraries required by `prox-tv` before syncing the
Python environment:

```bash
sudo apt update
sudo apt install -y build-essential libblas-dev liblapack-dev liblapacke-dev
uv sync
```

The dataset download uses `kagglehub`, so configure Kaggle credentials before
running the pipeline, if needed. A standard `~/.config/kaggle/kaggle.json` works.

The MATLAB CPSSM simulator is included as a supporting submodule. It is not
needed to run the DVC paper pipeline from the Kaggle dataset, but initialize it
when working with the simulator utilities:

```bash
git submodule update --init --recursive
```

## Reproduce Results

Run the complete DVC graph through the project entry point:

```bash
uv run python -m real_time_bp.main
```

This is equivalent to:

```bash
uv run dvc repro
```

Useful targeted commands:

```bash
uv run dvc status
uv run dvc repro dataset_summary
uv run dvc repro filter_bank_results
uv run dvc repro sliding_results
uv run dvc repro result_plots
```

The full graph can be expensive. Tune runtime behavior in `params.yaml`,
especially `runtime.csv_chunksize`, `runtime.parallel_workers`, and the
experiment subsets under `experiments`.

## Configuration

Key parameter groups in `params.yaml`:

- `dataset`: Kaggle handle, local raw-data directory, and force-download flag.
- `runtime`: CSV chunk size and shared process-pool worker count.
- `constants`: physical constants used for GPS L1 wavelength conversion.
- `quality_control`: loss-of-lock and gross diffractive-phase rejection rules.
- `experiments`: severity filters, `N_init`, Fresnel-time grid, and sensitivity sweeps.
- `tvd`: fade threshold and weighted TVD jump penalties.
- `sliding`: VAR order limits, FFT block sizes, latency settings, and example cases.
- `plots`: plot-specific ranges.
- `representatives`: ranks used to choose representative weak and strong cases.

## Outputs

Operational CSVs are written under `outputs`:

- `outputs/01_dataset`: case index, dataset summary, dataset paths, and Fresnel-time grid.
- `outputs/02_cycle_slips`: representative TVD cycle-slip traces.
- `outputs/03_filter_bank`: score curves, Fresnel-time estimates, downstream metrics, and sensitivity tables.
- `outputs/04_sliding`: VAR-order, practical sliding BP, latency heatmap, phase example, and forecast example tables.

Paper-facing artifacts are written under `results`:

- `results/figures`: PDF figures grouped by result section.
- `results/csv_tables`: generated CSV tables for reporting.
