# DTSC3602 Group 9 — D3 model evaluation

This folder compares a baseline and the group's two candidate fit models, using one validation protocol and one set of metrics.

| Role | Model | Source |
|---|---|---|
| Baseline | TF-IDF cosine (untrained) | `eval/run_evaluation.py` |
| Candidate A | JobSwipe: gte-small profile ↔ job cosine | github.com/Baguirr1/jobswipe @ `fa50c8b` |
| Candidate B | Job-Tinder: LightGBM on 11 pairwise features | github.com/awalla63/Job-Tinder-Test-build @ `cef3a70` |

The write-up is in **`MODELING_NOTE.md`**. The raw numbers are in `results/evaluation.json` and the figures are in `results/figures/`.

## Reproduce

Requires Python 3.12, Node ≥ 22.18 (to run JobSwipe's TypeScript directly) and git.

```bash
git clone https://github.com/Baguirr1/jobswipe repos/jobswipe
git clone https://github.com/awalla63/Job-Tinder-Test-build repos/job_tinder_test
git -C repos/jobswipe checkout fa50c8b
git -C repos/job_tinder_test checkout cef3a70

uv venv --python 3.12 .venv
uv pip install --python .venv --index-strategy unsafe-best-match -r requirements.txt

.venv/Scripts/python eval/run_evaluation.py     # ~10 min the first time, ~2-3 min after (embeddings cached)
.venv/Scripts/python eval/make_figures.py
```

The labelled dataset (`cnamuangtoun/resume-job-description-fit`, 8,000 pairs) downloads from Hugging Face on the first run. Use `--boot 200` for faster, slightly noisier confidence intervals.

## Files

- `D3_Model_Evaluation.ipynb`: the whole evaluation and the figures in one notebook, step by step. This is the same code as the `eval/` scripts below; run it from the repo root.
- `eval/run_evaluation.py`: the whole protocol (data, both splits, every model, metrics, CIs, error analysis). It imports Job-Tinder's own feature, training and metric code unchanged, so candidate B is exactly the shipped model.
- `eval/jobswipe_texts.mts`: builds the exact strings JobSwipe embeds by importing JobSwipe's own `embeddings.ts` and `resumeExtract.ts`.
- `eval/make_figures.py`: draws the figures from `results/evaluation.json`.
- `results/oof_scores.npz`: out-of-fold scores and predictions for every model (cold-resume split), for any further analysis.
