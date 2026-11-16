"""D3 model comparison: a baseline and the two candidate models, under one protocol.

Candidates (both are the group's existing models, scored without modification):

    jobswipe_gte_cosine   github.com/Baguirr1/jobswipe
                          gte-small (384-d) cosine between a profile vector and a job
                          vector. Profile = header (title, skills, bio suggested from the
                          resume) weighted 0.5 + up to two ~1,500-char resume chunks; job =
                          first 1,500 chars. The strings are built by JobSwipe's own
                          TypeScript (eval/jobswipe_texts.mts). Untrained / zero-shot.
    jobtinder_lightgbm    github.com/awalla63/Job-Tinder-Test-build
                          LightGBM multiclass on 11 pairwise features (bge-small chunked
                          cosine, TF-IDF cosine, skill overlap, years / seniority gaps),
                          using that repo's own feature code and hyper-parameters, retrained
                          inside every training fold.

Baseline:
    tfidf_cosine          TF-IDF cosine between resume and JD. No training, no embeddings.
                          The question every candidate must answer: does it beat word overlap?

Reference rows (context, not candidates):
    random                uniform random scores -- the floor for every metric
    jobswipe_resume_only  JobSwipe when the seeker accepts none of the resume suggestions
    bge_cosine            Job-Tinder's encoder alone, without the learned layer

Data: cnamuangtoun/resume-job-description-fit, 8,000 human-labelled pairs
(No Fit / Potential Fit / Good Fit), 643 unique resumes x 351 unique JDs.

Validation (identical for every model):
    cold_jd       the dataset's own split; no test JD appears in train (n = 1,759)
    cold_resume   5-fold GroupKFold by resume, out-of-fold predictions (n = 8,000)
Untrained models still use the training fold for one thing: the two score cut-points
that turn a similarity into a 3-class prediction (cut at the training label mix).

Metrics: AUC Good-vs-No, 3-class macro F1, per-class recall, NDCG@10 per resume
(seeker deck) and per JD (employer deck), share of No Fit cards in each seeker's top 5,
95% cluster-bootstrap CIs, and paired bootstrap CIs for the gaps between models.

    .venv/Scripts/python eval/run_evaluation.py              # full run (~15 min first time)
    .venv/Scripts/python eval/run_evaluation.py --boot 200   # quicker
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
JOBTINDER = ROOT / "repos" / "job_tinder_test"
sys.path.insert(0, str(JOBTINDER))

import lightgbm as lgb  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from ml.evaluate import (  # noqa: E402  Job-Tinder's metric code, reused unchanged
    auc_good_vs_no,
    build_features,
    cluster_bootstrap_auc,
    group_columns,
    metrics as base_metrics,
    paired_auc_gap_ci,
)
from ml.features import ALL_FEATURE_NAMES, FEATURE_SETS, LABEL_TO_INT, LABELS, fit_score  # noqa: E402
from ml.train import ensure_dataset, make_params  # noqa: E402

OUT = ROOT / "results"
CACHE = ROOT / "eval" / "cache"
GOOD, POT, NO = LABEL_TO_INT["Good Fit"], LABEL_TO_INT["Potential Fit"], LABEL_TO_INT["No Fit"]

BASELINE = "tfidf_cosine"
CANDIDATES = ["jobswipe_gte_cosine", "jobtinder_lightgbm"]
SHOWN = ["random", BASELINE, *CANDIDATES]  # rows in the error analysis

# JobSwipe's display mapping (supabase/migrations/..._gte_small_embeddings.sql, match_pct)
MATCH_LO, MATCH_HI = 0.80, 0.96
JOB_CHAR_CAP = 1500  # JobSwipe's MAX_INPUT_CHARS
HEADER_WEIGHT = 0.5  # JobSwipe's HEADER_WEIGHT


# ---------------------------------------------------------------------------
# JobSwipe scores
# ---------------------------------------------------------------------------
def jobswipe_scores(all_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """gte-small cosine per pair, for the 'suggested' and 'resume-only' profiles."""
    CACHE.mkdir(parents=True, exist_ok=True)
    resumes = sorted(set(all_df.resume_text))
    jobs = sorted(set(all_df.job_description_text))

    texts_path = CACHE / "jobswipe_texts.json"
    if not texts_path.exists():
        src = CACHE / "jobswipe_input.json"
        src.write_text(json.dumps({"resumes": resumes, "jobs": jobs}))
        subprocess.run(["node", str(ROOT / "eval" / "jobswipe_texts.mts"), str(src), str(texts_path)],
                       check=True, cwd=ROOT)
    texts = json.loads(texts_path.read_text(encoding="utf-8"))

    emb_path = CACHE / "gte_small_embeddings.npz"
    flat = sorted({t for v in ("profiles_suggested", "profiles_resume_only")
                   for p in texts[v] for t in [p["header"], *p["chunks"]] if t} | set(texts["jobs"]))
    if emb_path.exists() and list(np.load(emb_path, allow_pickle=True)["keys"]) == flat:
        vecs = np.load(emb_path, allow_pickle=True)["vectors"]
    else:
        from sentence_transformers import SentenceTransformer
        # thenlper/gte-small is the model Supabase.ai serves as 'gte-small' (mean pooled,
        # L2-normalised, 512-token window).
        model = SentenceTransformer("thenlper/gte-small", device="cpu")
        model.max_seq_length = 512
        print(f"  embedding {len(flat):,} JobSwipe strings with gte-small ...", flush=True)
        vecs = model.encode(flat, batch_size=16, normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=True)
        np.savez_compressed(emb_path, keys=np.array(flat, dtype=object), vectors=vecs)
    lookup = dict(zip(flat, vecs))

    def profile_vec(p: dict) -> np.ndarray:
        # generateProfileEmbedding(): header 0.5 + chunks share 0.5, then re-normalise.
        h, chunks = p["header"], p["chunks"]
        if h and chunks:
            parts = [(h, HEADER_WEIGHT)] + [(c, (1 - HEADER_WEIGHT) / len(chunks)) for c in chunks]
        elif h:
            parts = [(h, 1.0)]
        else:
            parts = [(c, 1 / len(chunks)) for c in chunks]
        v = sum(w * lookup[t] for t, w in parts)
        return v / np.linalg.norm(v)

    job_vec = {j: lookup[t] for j, t in zip(jobs, texts["jobs"])}
    out = {}
    for variant, key in (("jobswipe_gte_cosine", "profiles_suggested"),
                         ("jobswipe_resume_only", "profiles_resume_only")):
        res_vec = {r: profile_vec(p) for r, p in zip(resumes, texts[key])}
        out[variant] = np.array([float(res_vec[r] @ job_vec[j]) for r, j in
                                 zip(all_df.resume_text, all_df.job_description_text)])
    out["_header_has_title"] = {r: bool(p["header"].startswith("Role:"))
                                for r, p in zip(resumes, texts["profiles_suggested"])}
    return out


# ---------------------------------------------------------------------------
# Models: fit(train_idx) -> predict(idx) -> (score, predicted class)
# ---------------------------------------------------------------------------
def prior_cuts(train_scores, y_train):
    shares = np.bincount(y_train, minlength=3) / len(y_train)
    return np.quantile(train_scores, np.cumsum(shares)[:2])


def score_model(scores: np.ndarray):
    """An untrained similarity: rank by the score, classify with training-mix cut-points."""
    def fit(tr, y):
        cuts = prior_cuts(scores[tr], y[tr])
        return lambda idx: (scores[idx], np.searchsorted(cuts, scores[idx], side="right"))
    return fit


def lightgbm_model(X: np.ndarray, rounds: int, seed_offset: list[int]):
    """Job-Tinder's shipped model: LightGBM on the pairwise_plus features, its params."""
    names = FEATURE_SETS["pairwise_plus"]
    cols = [ALL_FEATURE_NAMES.index(n) for n in names]

    def fit(tr, y):
        booster = lgb.train(make_params(seed_offset[0]),
                            lgb.Dataset(X[tr][:, cols], label=y[tr], feature_name=names),
                            num_boost_round=rounds)

        def predict(idx):
            p = booster.predict(X[idx][:, cols])
            return fit_score(p), p.argmax(axis=1)
        return predict
    return fit


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def nofit_top5(y, score, resume_ids) -> float:
    """Mean share of No Fit pairs among each resume's 5 highest-scored jobs (resumes with >=5)."""
    frame = pd.DataFrame({"g": resume_ids, "y": y, "s": score})
    vals = [(c.nlargest(5, "s").y == NO).mean() for _, c in frame.groupby("g") if len(c) >= 5]
    return round(float(np.mean(vals)), 4)


def all_metrics(y, score, pred, jd_ids, resume_ids) -> dict:
    r = base_metrics(y, score, pred, jd_ids, resume_ids)
    r["nofit_share_top5_seeker"] = nofit_top5(y, score, resume_ids)
    r["confusion"] = confusion_matrix(y, pred, labels=[0, 1, 2]).tolist()
    return r


def extra_groups(all_df: pd.DataFrame, X: np.ndarray, has_title: dict) -> dict[str, pd.Series]:
    """Groups specific to how the two candidates read text (truncation, title detection)."""
    jd_len = all_df.job_description_text.str.len()
    res_len = all_df.resume_text.str.len()
    groups = group_columns(X)
    groups["jd_length"] = pd.Series(np.where(jd_len <= JOB_CHAR_CAP, "<=1,500 chars (fully read)",
                                    np.where(jd_len <= 3000, "1,501-3,000", ">3,000")))
    groups["resume_length"] = pd.Series(pd.cut(res_len, [0, 3000, 6000, 1e9],
                                               labels=["<=3,000 chars", "3,001-6,000", ">6,000"])
                                        .astype(str).to_numpy())
    groups["resume_title_detected"] = pd.Series(
        ["title found" if has_title[r] else "no title found" for r in all_df.resume_text])
    return groups


def error_breakdown(groups, y, oof) -> dict:
    out = {}
    for condition, g in groups.items():
        rows = []
        for value in sorted(g.unique(), key=str):
            m = (g == value).to_numpy()
            row = {"group": str(value), "n": int(m.sum()),
                   "good_fit_share": round(float((y[m] == GOOD).mean()), 3)}
            for name, (score, pred) in oof.items():
                severe = ((y[m] == NO) & (pred[m] == GOOD)) | ((y[m] == GOOD) & (pred[m] == NO))
                row[name] = {"auc": round(auc_good_vs_no(y[m], score[m]), 4),
                             "macro_f1": round(float(f1_score(y[m], pred[m], average="macro")), 4),
                             "severe_error_rate": round(float(severe.mean()), 4)}
            rows.append(row)
        out[condition] = rows
    return out


def match_pct(cos: np.ndarray) -> np.ndarray:
    return np.clip((cos - MATCH_LO) / (MATCH_HI - MATCH_LO), 0, 1) * 100


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    t0 = time.time()

    train_df, test_df = ensure_dataset()
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    y = all_df.label.map(LABEL_TO_INT).to_numpy()
    is_test = np.r_[np.zeros(len(train_df), bool), np.ones(len(test_df), bool)]
    jd_ids, res_ids = all_df.job_description_text.to_numpy(), all_df.resume_text.to_numpy()
    print(f"{len(all_df):,} pairs | {len(set(res_ids))} resumes | {len(set(jd_ids))} JDs")

    print("Job-Tinder features (TF-IDF, bge-small, skills, years, seniority) ...")
    X = build_features(all_df)
    print("JobSwipe gte-small scores ...")
    js = jobswipe_scores(all_df)
    rng = np.random.default_rng(args.seed)

    seed_box = [args.seed]
    models = {
        "random": ("reference", score_model(rng.random(len(y)))),
        "tfidf_cosine": ("baseline", score_model(X[:, ALL_FEATURE_NAMES.index("tfidf_cosine")])),
        "jobswipe_gte_cosine": ("candidate", score_model(js["jobswipe_gte_cosine"])),
        "jobtinder_lightgbm": ("candidate", lightgbm_model(X, args.rounds, seed_box)),
        "jobswipe_resume_only": ("ablation", score_model(js["jobswipe_resume_only"])),
        "bge_cosine": ("ablation", score_model(X[:, ALL_FEATURE_NAMES.index("emb_cosine")])),
    }

    # --- Split 1: cold-JD -------------------------------------------------------
    print("Cold-JD split ...")
    tr, te = np.flatnonzero(~is_test), np.flatnonzero(is_test)
    cold_jd, cj_scores = {}, {}
    for name, (_, fit) in models.items():
        seed_box[0] = args.seed
        score, pred = fit(tr, y)(te)
        r = all_metrics(y[te], score, pred, jd_ids[te], res_ids[te])
        r["auc_ci"] = cluster_bootstrap_auc(y[te], score, jd_ids[te], args.boot, args.seed)
        cold_jd[name], cj_scores[name] = r, score

    # --- Split 2: cold-resume ---------------------------------------------------
    print(f"Cold-resume split ({args.folds}-fold GroupKFold) ...")
    folds = list(GroupKFold(n_splits=args.folds).split(X, y, res_ids))
    cold_res, oof = {}, {}
    for name, (_, fit) in models.items():
        score, pred = np.zeros(len(y)), np.zeros(len(y), int)
        for k, (ftr, fva) in enumerate(folds, 1):
            seed_box[0] = args.seed + k  # matches Job-Tinder's train.py fold seeds
            score[fva], pred[fva] = fit(ftr, y)(fva)
        r = all_metrics(y, score, pred, jd_ids, res_ids)
        r["auc_ci"] = cluster_bootstrap_auc(y, score, res_ids, args.boot, args.seed)
        cold_res[name], oof[name] = r, (score, pred)

    # --- Paired comparisons -----------------------------------------------------
    pairs = [("jobtinder_lightgbm", BASELINE), ("jobswipe_gte_cosine", BASELINE),
             ("jobtinder_lightgbm", "jobswipe_gte_cosine"),
             ("jobswipe_gte_cosine", "jobswipe_resume_only"), ("jobswipe_gte_cosine", "bge_cosine")]
    gaps = {}
    for a, b in pairs:
        gaps[f"{a} - {b}"] = {
            "cold_jd": {"point": round(cold_jd[a]["auc_good_vs_no"] - cold_jd[b]["auc_good_vs_no"], 4),
                        "ci": paired_auc_gap_ci(y[te], cj_scores[a], cj_scores[b], jd_ids[te],
                                                args.boot, args.seed)},
            "cold_resume": {"point": round(cold_res[a]["auc_good_vs_no"] - cold_res[b]["auc_good_vs_no"], 4),
                            "ci": paired_auc_gap_ci(y, oof[a][0], oof[b][0], res_ids,
                                                    args.boot, args.seed)},
        }

    # --- Error analysis (cold-resume OOF: every pair scored by a model blind to its resume)
    groups = extra_groups(all_df, X, js["_header_has_title"])
    errors = error_breakdown(groups, y, {n: oof[n] for n in SHOWN})

    # Most confidently wrong pairs for the two candidates (for the write-up)
    worst = {}
    for name in CANDIDATES:
        s, p = oof[name]
        pct = pd.Series(s).rank(pct=True).to_numpy()
        fp = np.flatnonzero((y == NO))[np.argsort(-pct[y == NO])[:5]]
        fn = np.flatnonzero((y == GOOD))[np.argsort(pct[y == GOOD])[:5]]
        worst[name] = {kind: [{"score_percentile": round(float(pct[i]), 3),
                               "resume_start": all_df.resume_text[i][:160],
                               "jd_start": all_df.job_description_text[i][:160]} for i in idx]
                       for kind, idx in (("no_fit_ranked_highest", fp), ("good_fit_ranked_lowest", fn))}

    # JobSwipe's user-facing "% match" badge, by true label
    mp = match_pct(js["jobswipe_gte_cosine"])
    badge = {lab: {"mean_match_pct": round(float(mp[y == i].mean()), 1),
                   "share_shown_>=70%": round(float((mp[y == i] >= 70).mean()), 3),
                   "share_shown_<30%": round(float((mp[y == i] < 30).mean()), 3)}
             for i, lab in enumerate(LABELS)}
    cos = js["jobswipe_gte_cosine"]
    badge["raw_cosine_range"] = [round(float(cos.min()), 4), round(float(cos.max()), 4)]

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "duration_seconds": round(time.time() - t0, 1),
        "data": {"n_pairs": int(len(y)), "n_resumes": int(len(set(res_ids))),
                 "n_jds": int(len(set(jd_ids))), "n_cold_jd_test": int(is_test.sum()),
                 "label_share": {lab: round(float((y == i).mean()), 3) for i, lab in enumerate(LABELS)}},
        "roles": {n: role for n, (role, _) in models.items()},
        "cold_jd": cold_jd, "cold_resume": cold_res,
        "paired_auc_gaps": gaps,
        "error_analysis_cold_resume": errors,
        "jobswipe_match_badge_by_label": badge,
        "most_confident_errors_cold_resume": worst,
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(OUT / "oof_scores.npz", y=y, is_test=is_test,
                        **{f"{n}_score": oof[n][0] for n in oof},
                        **{f"{n}_pred": oof[n][1] for n in oof})
    print_report(report)
    print(f"\nSaved results/evaluation.json ({report['duration_seconds']}s)")


def print_report(rep: dict) -> None:
    for split in ("cold_jd", "cold_resume"):
        print(f"\n{split.upper()}")
        print(f"{'model':<22}{'AUC':>7}{'95% CI':>17}{'macroF1':>9}{'NDCGseek':>10}"
              f"{'NDCGemp':>9}{'NoFit@5':>9}{'recGood':>9}")
        for n, r in rep[split].items():
            ci = f"[{r['auc_ci'][0]:.3f},{r['auc_ci'][1]:.3f}]"
            print(f"{n:<22}{r['auc_good_vs_no']:>7.3f}{ci:>17}{r['macro_f1']:>9.3f}"
                  f"{r['ndcg_seeker_at_10']:>10.3f}{r['ndcg_employer_at_10']:>9.3f}"
                  f"{r['nofit_share_top5_seeker']:>9.3f}{r['recall_per_class']['Good Fit']:>9.3f}")
    print("\nPAIRED AUC GAPS (95% cluster-bootstrap CI)")
    for k, v in rep["paired_auc_gaps"].items():
        print(f"  {k:<46} cold_jd {v['cold_jd']['point']:+.3f} {v['cold_jd']['ci']}  "
              f"cold_resume {v['cold_resume']['point']:+.3f} {v['cold_resume']['ci']}")
    print("\nERROR ANALYSIS (cold-resume OOF) AUC / severe-error rate")
    for cond, rows in rep["error_analysis_cold_resume"].items():
        print(f"  {cond}")
        for row in rows:
            cells = "  ".join(f"{n[:14]} {row[n]['auc']:.3f}/{row[n]['severe_error_rate']:.3f}"
                              for n in SHOWN)
            print(f"    {row['group']:<28}{row['n']:>6}  {cells}")
    print("\nJobSwipe badge by label:", json.dumps(rep["jobswipe_match_badge_by_label"]))


if __name__ == "__main__":
    main()
