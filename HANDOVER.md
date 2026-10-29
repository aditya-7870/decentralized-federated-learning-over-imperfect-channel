# Team 21 review handover

## Prepare the laptop before the review

Clone or download the complete repository, then open a terminal in its root folder (the folder containing `demo.py`). Open the revised Word report and `CORRECTIONS.md`. Keep the figures ready in a second window.

Run `python demo.py` once in advance. This needs only Python and reads the saved results. For the live trained-model demonstration, follow the CPU installation commands in README, then run `.\.venv\Scripts\python.exe demo.py --predict`. Complete dependency installation before arriving; neither demonstration command needs a dataset download or a GPU. Keep `demo_assets/mnist_demo.pt`, its prediction image, and `results/review/` in the clone.

## Suggested ten minute presentation

| Time | What to show | What to explain |
|---|---|---|
| 0–1 min | Report abstract and research-gap table in Section 2.4 | The project measures the benefit and communication cost of decentralized learning under heterogeneous data, losses, and delay. It evaluates an implemented gossip procedure; it does not claim to invent decentralized learning. |
| 1–3 min | Architecture figure and source map below | Clients train locally, send model snapshots to graph neighbors, and combine accepted arrivals. M1 always sends; M2 applies a change threshold; M3 analyzes ideal averaging separately. |
| 3–5 min | `python demo.py` and `baseline_accuracy.png` | There are 155 completed runs across 31 conditions, each with five seeds. Explain local-only controls and why two-class/Dirichlet partitions reveal communication benefit more clearly than IID accuracy alone. |
| 5–7 min | `event_tradeoff.png`, `delay_effect.png`, and corresponding report tables | Message reduction has an accuracy cost. Delay experiments distinguish stale arrivals from missing messages. Report confidence intervals and the tested conditions. |
| 7–8 min | `.\.venv\Scripts\python.exe demo.py --predict` and `demo_assets/demo_predictions.png` | This loads an already trained MNIST client model on CPU and predicts twenty bundled digits. It demonstrates inference, not live distributed training or a fresh full-test evaluation. |
| 8–10 min | `CORRECTIONS.md`, tests, and limitations | Walk through the teacher's requested corrections and their evidence. Explain that simulated message loss is not a complete wireless physical layer and that ideal spectral bounds do not prove convergence of impaired CNN training. |

## Source files to explain

| File | Role in the project |
|---|---|
| `dataset.py` | Loads benchmarks and creates IID, exact two-class, and Dirichlet client partitions. |
| `model.py` and `client.py` | Define the CNN and client-local training. |
| `network.py` | Creates graph topologies and Metropolis mixing weights. |
| `channel.py` | Simulates directed-message erasures, burst loss, delayed arrivals, and communication counts. |
| `gossip.py` and `gossip_event_triggered.py` | Implement standard and event-triggered exchange with a common receiver path. |
| `experiments.py` | Runs the controlled experiment suites and saves per-seed evidence. The companion runners add loss/topology, original-budget, and CIFAR comparisons. |
| `metrics.py` and `summarize_results.py` | Compute model disagreement, accuracy, traffic, and aggregate seed-level statistics. |
| `spectral_analysis.py` and `main_m3.py` | Analyze signed spectra and ideal averaging bounds separately from CNN training. |
| `tests/test_correctness.py` | Checks partition coverage, communication accounting, delay freshness, equivalence, and mathematical invariants. |
| `demo.py` | Reads saved results or runs the portable CPU checkpoint. |

Start with the workflow and open only the relevant functions when the teacher asks. Explain inputs, outputs, and the reason for each module rather than reading the source line by line. `main.py`, `main_m2.py`, and `run.py` remain legacy entry points; the revised report's controlled experiment evidence comes from the review-suite runners.

## Likely questions and defensible answers

- **Was the model trained on this laptop?** The saved experiments were trained on the RTX 5070 Ti laptop. This review laptop loads the saved results and performs CPU inference; it does not need to repeat training.
- **Why do you need local-only training?** It measures how much accuracy is attributable to communication under the same partitions, initialization, and training budget.
- **How much communication is saved?** Quote the threshold and accuracy together. On Dirichlet MNIST, Ring at delta 2 saves 79.6% of directed messages, while mean accuracy falls from 89.83% to 62.37%.
- **Is Ring always the best topology?** No. Results depend on client count, graph instance, data allocation, and budget. At 50 clients in the two-class comparison, Grid outperforms Ring.
- **Does the spectral bound prove CNN convergence under loss?** No. It applies to pure averaging with a fixed symmetric doubly stochastic matrix. Local SGD, delayed arrivals, and directed erasures change the update dynamics.
- **Is this the same as a real wireless deployment?** It simulates application-message loss and round-based delay. It does not implement radio waveforms, physical packetization, retransmission protocols, or measured energy consumption.
- **What is the contribution compared with the references?** A reproducible controlled evaluation combining matched baselines, heterogeneous partitions, communication accounting, impairments, uncertainty, and a separate ideal spectral check. The project does not benchmark implementations of every cited algorithm.

If the teacher requests validation, use the prepared environment to run `python -m pytest -q` and `python validate_results.py`. The tests and saved-result audit do not require repeating the training suite. Avoid starting `experiments.py --suite all` during the presentation.

## Show the report and evidence

Open `output/report/Team21_Revised_Decentralized_Learning_Report.docx`. It retains the academic single-column style, author block, numbered sections, tables, equations, and figure captions of the original PDF, with corrected content.

The expanded report has 23 pages, restores the original section structure, and includes a literature review and a research-gap table connecting the literature to the completed experiments. Its 33 cited sources include all 30 supplied references. Bibliographic corrections are recorded in `report_sources/REFERENCE_VERIFICATION.md`. Include `report_sources/` and `tools/report_sections.py` when transferring the report-generation source.

Open `results/review/figures/baseline_accuracy.png` to explain why local-only controls and non-IID data matter. Then use `event_tradeoff.png` for bandwidth versus accuracy and `delay_effect.png` for stale communication. `results/review/m3/` contains ideal graph analysis.

## Demonstrate without training

```powershell
python demo.py
```

This prints saved results without needing PyTorch. For the working trained model, after installing CPU PyTorch as described in README:

```powershell
.\.venv\Scripts\python.exe demo.py --predict
```

The command runs twenty included test digits on CPU. Training is not started, and the dataset is not downloaded. Open `demo_assets/demo_predictions.png` alongside the terminal to show those digits and predictions.

## Explain the important corrections

1. Local-only is the control: high IID accuracy alone cannot prove communication helps.
2. Each condition has five seeds and a 95% Student-t interval. New training uses ten rounds and one local epoch, so do not directly compare its accuracies with the old forty-epoch results.
3. M1 and M2 with threshold zero share the same receiver path and matched initialization. Exact model hashes and per-round accuracy agree across five full ten-round GPU validation runs.
4. The CNN has 206,922 MNIST parameters. Its float32 model payload is 827,688 bytes.
5. Broadcasts and directed messages have different denominators. The original Grid result saves 8/100 broadcasts but 12/200 messages: 6% payload traffic savings.
6. The five-node Grid has a degree-one node. Do not claim that Ring universally beats Grid.
7. Spectral analysis concerns ideal pure averaging. Local SGD, directed losses, and delayed messages are additional effects; renormalization can move the network average.
8. This is an application-message simulator, not a complete wireless physical-layer implementation. CIFAR-10, when included in the results, is a short-budget local-only/Ring extension, not a full sweep.

## What is safe to run during review

Use `demo.py`, the report, and saved figures. `summarize_results.py` regenerates summary plots without training but needs the numerical/plotting dependencies. `main_m3.py` analyzes graphs without CNN training. `experiments.py` and `cifar_experiments.py` start training; they are not needed for the demonstration.

## Files that must reach your laptop

Include the source, README, this guide, `CORRECTIONS.md`, both requirements files, `tests/`, `tools/`, `report_sources/`, `results/review/`, `demo_assets/`, and `output/report/`. Keep the original PDF reports, teacher correction DOCX, and `Review/` documents for comparison. The portable demo checkpoint is small enough for ordinary Git. Downloaded data and virtual environments do not need to be transferred.

## Transfer through GitHub

From the project root, inspect what will be committed:

```powershell
git add -A
git status --short
git diff --cached --stat
```

Confirm that the revised Word report, `demo_assets/mnist_demo.pt`, all 155 files in `results/review/runs/`, the five `validation_runs/` records, report sources, and review guides appear in the staged files or are already tracked. The original report PDFs and correction documents should remain tracked. One-off PDF extraction scripts and extracted text may appear as removals from Git; they are not used by the project or report builder.

Then create a commit and push your current branch:

```powershell
git commit -m "Prepare corrected project and report for review"
git push
```

If this branch has no upstream, use the upstream command Git prints for your branch. On the review laptop, clone the repository or download its ZIP after the push completes. Open the actual project folder before running commands. For an existing clone, commit or preserve any local work before pulling the updated branch.

The ignore rules keep local environments, downloaded datasets, caches, logs, internal QA copies, extraction helpers, and bulk training checkpoints off GitHub. They preserve the portable trained checkpoint, results, validation evidence, report-generation sources, and original/final report documents. Ignoring a file does not delete it from your laptop. Do not use `git add -f` on the ignored folders.

## Completed evidence

All 155 experiment runs are complete, covering 31 conditions with five seeds each. Five extra full-budget GPU runs validate zero-threshold M1/M2 equality. All 15 regression tests and all saved-result accounting checks pass.

| Comparison | Mean client accuracy and 95% interval |
|---|---|
| Original-budget IID, 20 rounds × 2 local epochs | Local-only 98.52 ± 0.07%; Ring 99.17 ± 0.09%; paired gain 0.65 ± 0.02 percentage points |
| Two-class MNIST, 10 rounds × 1 local epoch | Local-only 19.953 ± 0.003%; Ring 49.27 ± 3.13% |
| Dirichlet MNIST, same short budget | Local-only 53.03 ± 4.55%; Ring 89.83 ± 1.05% |
| Dirichlet CIFAR-10, same short budget | Local-only 28.29 ± 2.84%; Ring 38.36 ± 3.15% |

Traffic savings must be shown with their accuracy cost: Ring at delta=2 saves 79.6% of messages on Dirichlet MNIST, but mean accuracy falls to 62.37% from 89.83%. This is a tradeoff, not free communication reduction.
