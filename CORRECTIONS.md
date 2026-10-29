# Teacher correction mapping

The original report and raw results remain unchanged. The revised Word report and new evidence are separate artifacts.

| Requested correction | Implemented change | Evidence |
|---|---|---|
| Local-only baseline | Matched initialization, partitions, local epochs, and minibatch order; no channel calls | `experiments.py`; raw local runs |
| Non-IID evaluation | Exact two-label/client and separate Dirichlet alpha=.1 allocations; complete disjoint coverage | Partition histograms, sizes, hashes; coverage tests |
| Delay evaluation | Source-round metadata, latest-version selection, pending arrivals; max delay 0/1/2/4 | Raw impairment runs; freshness/accounting tests |
| Burst loss | Per-directed-link Gilbert-Elliott chain with stationary loss .2 | Channel parameters in report; silent-round state test |
| Five seeds and intervals | Seeds 42-46; 95% Student-t intervals; paired local-only differences | Summary JSON and figures |
| M1 vs M2 zero threshold | One reference-model initialization; dedicated shuffle generators; common receiver path | Exact GPU smoke hashes and CPU entry-point regression |
| Parameters and bytes | MNIST 206,922 parameters; 827,688 bytes/model | Architecture test; revised tables |
| Spectra and grid degrees | Preserve signed eigenvalues; correct minimum; explicit Grid degree-three node | Spectral tests, graph diagrams, tables |
| Event denominators | Separate client broadcasts from degree-weighted directed messages and bytes | Grid 188/200 attempts = 6% legacy traffic savings; accounting audit |
| Consensus and spectral claims | Pure averaging uses squared-error rho^(2t) bound; no claim it proves impaired SGD convergence | Spectral bound test and revised analysis/discussion |
| Renormalization drift | Measure aggregation-only mean drift; explicitly retain it as a limitation | One-way-reception regression and raw per-round drift |
| N=5 scope | Five-seed neural training at N=20/50, Ring and Grid; spectral scale sweep | Scale run records and tables |
| Harder task | CIFAR-10 runner using same engine; local-only/Ring Dirichlet comparison | Five-seed CIFAR runs completed; restricted comparison scope stated in report |
| Title and literature | Distinct title; expanded review of all 30 supplied works; research-gap questions linked to implemented experiments; corrected bibliographic metadata | 33 cited sources; literature synthesis and research-gap tables; `report_sources/REFERENCE_VERIFICATION.md` |
| Presentation | New Word document in the original academic style; numbered figures/tables/equations; no raw LaTeX or HTML | Visually checked Word export |
| Code availability and handover | Requirements, tests, raw JSON, checkpoint, CPU demo, review guide | README and HANDOVER.md |

The revised ten-round, one-local-epoch experiment is not a numerical continuation of the original twenty-round, two-local-epoch experiment. Historical single-seed values are discussed only as an audit of the old report. No claim of publication acceptance or journal readiness is made.
