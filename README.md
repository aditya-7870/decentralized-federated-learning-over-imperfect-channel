# Decentralized learning over lossy links

Reproducible Computer Networks project: local-only training, standard gossip (M1), event-triggered gossip (M2), and ideal spectral analysis (M3).

## Review on another laptop

**No training or GPU is needed to show the completed work.**

1. Open `output/report/Team21_Revised_Decentralized_Learning_Report.docx` in Word.
2. Open the figures in `results/review/figures/` and `results/review/m3/`.
3. Run the saved-results viewer with any compatible Python:

```powershell
python demo.py
```

That command uses only the standard library. For CPU inference, use 64-bit Python 3.14 (the tested version), then:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe demo.py --predict
```

The inference command loads `demo_assets/mnist_demo.pt` onto the CPU and predicts twenty included test digits. It does not download data or retrain. The checkpoint contains client 0 from the seed-42 IID Ring run, not an ensemble or the best model selected from all seeds.

## Reproduce the experiments

The tested training environment is Python 3.14, PyTorch 2.14.1+cu130, TorchVision 0.29.1+cu130, and an RTX 5070 Ti Laptop GPU. The NVIDIA driver must support the selected CUDA wheel. CPU execution is supported but slower.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe experiments.py --suite smoke --output .work/smoke
.\.venv\Scripts\python.exe experiments.py --suite all
.\.venv\Scripts\python.exe loss_topology_experiments.py
.\.venv\Scripts\python.exe original_budget_experiments.py
.\.venv\Scripts\python.exe cifar_experiments.py
.\.venv\Scripts\python.exe summarize_results.py
.\.venv\Scripts\python.exe main_m3.py --results-dir results/review
.\.venv\Scripts\python.exe make_report_figures.py
```

MNIST and CIFAR-10 download into `data/` on first use. Internet access is required for installation and first dataset download only. The main suite uses the full MNIST training and test sets, not a sampled subset.

### Experiment suites

- `smoke`: one-round M1 versus zero-threshold M2, for implementation validation.
- `core`: 60 runs = five seeds × three partitions × four methods (local, Ring, Grid, Random).
- `extended`: core plus fixed event thresholds, delay, Bernoulli loss, and burst loss = 105 runs.
- `all`: extended plus N=20/50 two-class Ring/Grid learning = 125 runs.
- `loss_topology_experiments.py`: ten more MNIST runs completing the 20% loss comparison on Grid and Random.
- `original_budget_experiments.py`: ten matched IID local-only/Ring runs at the original 20-round, two-local-epoch, batch-64 budget.
- `cifar_experiments.py`: ten additional runs = five seeds × local-only/Ring on Dirichlet CIFAR-10.

Core and impairment runs use ten rounds, one local epoch, batch size 128, learning rate .01, and local SGD momentum .9. The separately labeled original-budget control uses twenty rounds, two epochs, and batch size 64. A run-level restart skips complete JSON files. It does not resume an interrupted optimizer state. Delete or move a condition's JSON deliberately if you want to repeat that condition; use a new output folder when changing the protocol.

## Scientific interpretation

- Reported accuracy is the mean of individual client test accuracies. It is not the accuracy of a central average or an ensemble.
- The two-class partition is exactly two labels/client; Dirichlet alpha=.1 is a separate setting. Partitions are disjoint and cover all samples.
- The objective weights clients uniformly. This differs from pooled sample weighting for unequal client data sizes. Optimizer momentum remains local after model mixing.
- A complete float32 model is one application message. Byte totals exclude packet headers, fragmentation, retransmissions, and energy.
- Broadcast suppression uses clients × rounds; actual traffic savings use directed neighbor messages. The original Grid's reported 8% broadcast suppression is 6% message/byte savings (188 versus 200 attempts).
- Erasures with receiver renormalization can change the model mean. Aggregation-induced drift is measured explicitly.
- Gilbert-Elliott has transitions pGB=.05 and pBG=.2, erasure probabilities 0/1, stationary loss .2, and mean bad burst five rounds. It is initialized in stationarity.
- Delayed messages carry source-round versions. Older arrivals cannot overwrite newer accepted models. Arrivals beyond the last training round remain pending.
- M3's `E(t) <= E(0) * rho**(2*t)` is a squared-disagreement bound for fixed symmetric ideal averaging. It is not a convergence guarantee for SGD with losses or delay.
- Confidence intervals use Student t across seed-level results. Paired gains are computed within seeds. Five seeds do not establish universal superiority.

## Files and outputs

| Path | Purpose |
|---|---|
| `experiments.py` | Recommended reproducible MNIST experiment runner |
| `cifar_experiments.py` | Harder-task extension using the same engine |
| `channel.py` | Independent/burst erasures, delay buffer, arrival accounting |
| `gossip.py` / `gossip_event_triggered.py` | Shared receiver path and sender thresholds |
| `dataset.py` | IID, shard, exact two-class, Dirichlet partition utilities |
| `model.py` / `client.py` | CNNs, local SGD, model serialization |
| `main.py` / `main_m2.py` | Original configurable M1/M2 entry points, with fixes |
| `main_m3.py` / `spectral_analysis.py` | Signed spectra and ideal averaging analysis |
| `summarize_results.py` | Seed statistics, paired gains, and new figures |
| `results/review/runs/` | Complete revised run records |
| `results/review/summary.json` | Recomputed statistics used by report |
| `results/review/validation.json` | Validation provenance |
| `results/review/legacy_accounting_audit.json` | Recalculated original message savings |
| `demo_assets/` | Tracked portable checkpoint and prediction illustration |
| `output/report/` | Revised editable report |
| `HANDOVER.md` | Review-day commands and explanation |

Original report PDFs and original `results/*.json` / `results/m2/` remain historical evidence. The corrected legacy CLI defaults to `results/revised/` to avoid overwriting those files. Its results have a different schema from the new review suite.

## Validation

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Tests cover the scientific correctness issues, including the real M1/M2 entry points under loss and delay. `validate_gpu_equivalence.py` additionally repeats five full-budget GPU zero-threshold runs and compares parameter hashes with M1. Run `validate_results.py` to audit saved-run accounting and data-coverage invariants.

## Report source

`tools/build_report.py` creates the Word report from five-seed summary data using `python-docx`. Keep `tools/report_sections.py` and `report_sources/` alongside it: they contain the expanded literature review, research-gap discussion, verified bibliography, and reference audit. The report includes all 30 supplied literature references plus FedAvg and dataset sources, numbered by first citation. It was authored with the Codex bundled document runtime. To regenerate on another machine, install `python-docx`, run the figure/statistics commands first, and then run this script. `tools/render_report_word.ps1` provides a Windows Word PDF export for visual QA; it needs Microsoft Word. QA renders and temporary environments are excluded from Git.

Before pushing, include `results/review/`, `demo_assets/`, `output/report/`, `report_sources/`, `tools/`, tests, requirements, source files, and documentation. Keep the original PDF reports and teacher correction documents too. `data/`, `.venv/`, `.work/`, internal QA renders, bulk legacy checkpoints, and extra local training output folders are intentionally ignored. Do not add the downloaded datasets or environment to Git. Follow `HANDOVER.md` for the review-day sequence and GitHub transfer checklist.
