# hybrid-talent-rec-benchmarks

Minimal replication package for the paper *Benchmark-Sensitive Hybrid Recommendation for Talent Path Prediction*.

This repository contains the frozen ICT benchmark files, the model and evaluation code, and the derived result artifacts needed to reproduce the manuscript's main tables, figures, and robustness checks.

Dr. Özkan Canay
Dept. of Information Systems and Technologies
Sakarya University
canay@sakarya.edu.tr

## Repository contents

- `data/prepared/`: frozen benchmark files for `jobhop_ict_benchmark_v1` and `karrierewege_ict_benchmark_v1`
- `data/raw/`: retrieval notes for the original public source files
- `src/`: benchmark preparation and core training/evaluation code
- `scripts/`: focused analysis utilities for figure generation and robustness checks
- `output/`: manuscript result files regenerated from the frozen benchmarks
- `manuscript/figs/`: figure PDFs produced from the included result files

## Tested environment

- Python 3.12
- Windows 11

The Python code is platform-agnostic. The included commands should also run on Linux or macOS with the same folder layout.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Quick reproduction

Run the two main evaluations:

```bash
python src/train_eval_talent_hybrid.py --dataset-dir data/prepared/jobhop_ict_benchmark_v1 --output-path output/jobhop_with_gru4rec_results.json
python src/train_eval_talent_hybrid.py --dataset-dir data/prepared/karrierewege_ict_benchmark_v1 --output-path output/karrierewege_with_gru4rec_results.json
```

Regenerate the manuscript-side follow-up analyses:

```bash
python scripts/analyze_proxy_sensitivity.py --dataset-dir data/prepared/jobhop_ict_benchmark_v1 --dataset-dir data/prepared/karrierewege_ict_benchmark_v1 --output-path output/review/proxy_sensitivity_results.json
python scripts/benchmark_model_runtime.py --dataset-dir data/prepared/jobhop_ict_benchmark_v1 --dataset-dir data/prepared/karrierewege_ict_benchmark_v1 --output-path output/review/runtime_benchmark_results.json
python scripts/analyze_effect_sizes.py --input-path output/jobhop_with_gru4rec_results.json --output-path output/review/jobhop_effect_sizes.json
python scripts/analyze_effect_sizes.py --input-path output/karrierewege_with_gru4rec_results.json --output-path output/review/karrierewege_effect_sizes.json
python scripts/analyze_dqn_ablation.py --dataset-dir data/prepared/jobhop_ict_benchmark_v1 --output-path output/review/jobhop_dqn_ablation.json
python scripts/analyze_dqn_ablation.py --dataset-dir data/prepared/karrierewege_ict_benchmark_v1 --output-path output/review/karrierewege_dqn_ablation.json
python scripts/generate_interpretability_case.py --dataset-dir data/prepared/jobhop_ict_benchmark_v1 --user-id 57220 --output-path output/review/jobhop_interpretability_case_57220.json
python scripts/generate_result_figures.py
```

## Output mapping

- `output/jobhop_with_gru4rec_results.json` and `output/karrierewege_with_gru4rec_results.json`: main repeated-split benchmark results
- `output/review/proxy_sensitivity_results.json`: leave-one-proxy-out sensitivity analysis
- `output/review/runtime_benchmark_results.json`: coarse runtime comparison
- `output/review/jobhop_effect_sizes.json` and `output/review/karrierewege_effect_sizes.json`: paired standardized effect sizes
- `output/review/jobhop_dqn_ablation.json` and `output/review/karrierewege_dqn_ablation.json`: family-level DQN robustness check
- `output/review/jobhop_interpretability_case_57220.json`: worked JobHop interpretability example
- `manuscript/figs/fig3_selected_models.pdf` and `manuscript/figs/fig4_fusion_weights.pdf`: figure files regenerated from the included outputs

## Raw public data

The original public source files are not mirrored here. The exact pinned URLs used during benchmark freezing are stored in:

- `data/prepared/jobhop_ict_benchmark_v1/metadata.json`
- `data/prepared/karrierewege_ict_benchmark_v1/metadata.json`

See `data/raw/README.md` for retrieval notes and source entry points.
