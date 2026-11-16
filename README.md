# DTSC3602 Group 9 — D3 model evaluation

This repo compares a baseline and the group's two candidate fit models on two tests, using the same metrics throughout.

| Role | Model | Source |
|---|---|---|
| Baseline | TF-IDF cosine (untrained) | `eval/run_evaluation.py` |
| Candidate A | JobSwipe: gte-small profile ↔ job cosine | github.com/Baguirr1/jobswipe @ `fa50c8b` |
| Candidate B | Job-Tinder: LightGBM on 11 pairwise features | github.com/awalla63/Job-Tinder-Test-build @ `cef3a70` |

- **Test 1, labelled fit data:** `cnamuangtoun/resume-job-description-fit`, 8,000 labelled pairs, with cold-resume and cold-JD splits. This is Job-Tinder's training source.
- **Test 2, neutral ground:** 240 Kaggle resumes ranked against real LinkedIn postings. Neither model was trained on this data, and the label is whether a posting's title matches the resume's field.

**Result:** Job-Tinder wins Test 1 clearly but is significantly worse than the baseline on Test 2. JobSwipe is statistically tied with the baseline on both tests. **Selected: JobSwipe's embedding-similarity ranking**, with TF-IDF as a fallback. The reasoning is in **`MODELING_NOTE.md`** and **`D3_Modeling_Note.docx`**.

## Reproduce

Requires Python 3.12, Node ≥ 22.18 (to run JobSwipe's TypeScript directly) and git. Test 2 also needs the D1 project's Kaggle data next to this folder: `../dtsc3602_proj/resumes/Resume/Resume.csv` and `../dtsc3602_proj/postings/postings.csv`.

```bash
git clone https://github.com/Baguirr1/jobswipe repos/jobswipe
git clone https://github.com/awalla63/Job-Tinder-Test-build repos/job_tinder_test
git -C repos/jobswipe checkout fa50c8b
git -C repos/job_tinder_test checkout cef3a70

uv venv --python 3.12 .venv
uv pip install --python .venv --index-strategy unsafe-best-match -r requirements.txt

.venv/Scripts/python eval/run_evaluation.py     # Test 1: ~10 min the first time, ~2-3 min after (embeddings cached)
.venv/Scripts/python eval/make_figures.py
.venv/Scripts/python eval/in_domain_check.py    # Test 2: ~15 min the first time
```

The labelled dataset downloads from Hugging Face on the first run. Use `--boot 200` with `run_evaluation.py` for faster, slightly noisier confidence intervals.

## Files

- `D3_Model_Evaluation.ipynb`: both tests and the figures in one notebook, step by step. This is the same code as the `eval/` scripts below; run it from the repo root.
- `eval/run_evaluation.py`: Test 1, the whole protocol (data, both splits, every model, metrics, CIs, error analysis). It imports Job-Tinder's own feature, training and metric code unchanged.
- `eval/in_domain_check.py`: Test 2. It scores Job-Tinder with its shipped model files, not a retrained model.
- `eval/jobswipe_texts.mts`: builds the exact strings JobSwipe embeds by importing JobSwipe's own `embeddings.ts` and `resumeExtract.ts`.
- `eval/make_figures.py`: draws the figures from `results/evaluation.json`.
- `results/evaluation.json`, `results/in_domain_check.json`: all the numbers.
- `results/oof_scores.npz`: Test 1 out-of-fold scores and predictions for every model.
