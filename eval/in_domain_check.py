"""Neutral-ground check: rank real LinkedIn postings for real Kaggle resumes.

The labelled fit dataset is Job-Tinder's training source, which gives it a home-field
advantage. This check uses data neither model was trained on:

    resumes   Kaggle Resume.csv (livecareer, 24 categories); 10 per category = 240
              (Job-Tinder's README verifies these are disjoint from its training data)
    postings  Kaggle LinkedIn postings.csv, description >= 700 chars

Proxy label: a posting is ON-TARGET for a resume when its title matches the resume's
category (the CATEGORY_TITLES patterns from Job-Tinder's evaluate_in_domain.py, used as a
label definition only). Every resume gets the same kind of candidate list: 10 on-target
postings + 40 off-target postings, and each model ranks those 50.

Each model runs exactly as in production:
    jobswipe     gte-small; job text = JobSwipe's buildJobText(title, location/remote,
                 description); profile = header suggested from the resume + 2 chunks
    jobtinder    the SHIPPED models/fit_model.txt + fit_artifacts.joblib (not retrained);
                 job text = title + company + description[:4000], bge-small chunked
    tfidf        baseline: TF-IDF cosine fitted on this sample's text

Metrics (mean over resumes, 95% bootstrap CI over resumes):
    auc            on-target vs off-target, per resume
    precision@10   share of the top 10 that are on-target (10 exist, so 1.0 is possible)
    ndcg@10
    seniority_mismatch@10   share of the top 10 that is a severe level mismatch: a resume
                 with <= 2 years shown a Director/Executive posting, or >= 10 years shown
                 Internship/Entry level (LinkedIn's own level field)

    .venv/Scripts/python eval/in_domain_check.py      (~10 min, mostly embedding)
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
JT = ROOT / "repos" / "job_tinder_test"
sys.path.insert(0, str(JT))
D1 = ROOT.parent / "dtsc3602_proj"
CACHE = ROOT / "eval" / "cache"
OUT = ROOT / "results"

import joblib  # noqa: E402
import lightgbm as lgb  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.metrics import ndcg_score, roc_auc_score  # noqa: E402
from sklearn.preprocessing import normalize  # noqa: E402

from ml.encoder import get_encoder, job_text  # noqa: E402
from ml.evaluate_in_domain import CATEGORY_TITLES  # noqa: E402
from ml.features import JobSide, ResumeSide, build_matrix, fit_score, resume_years  # noqa: E402

SEED = 42
PER_CATEGORY = 10
N_ON, N_OFF = 10, 40
POOL_ON, POOL_RANDOM = 25, 600
TOP = 10
BOOT = 1000
MODELS = ["random", "tfidf_cosine", "jobswipe_gte_cosine", "jobtinder_lightgbm"]


def load_sample():
    rng = np.random.default_rng(SEED)
    res = pd.read_csv(D1 / "resumes" / "Resume" / "Resume.csv", usecols=["ID", "Resume_str", "Category"])
    res = res.groupby("Category").sample(PER_CATEGORY, random_state=SEED).reset_index(drop=True)
    res["Resume_str"] = res.Resume_str.str.strip()

    cols = ["job_id", "title", "company_name", "description", "location", "remote_allowed",
            "formatted_experience_level"]
    jobs = pd.read_csv(D1 / "postings" / "postings.csv", usecols=cols)
    jobs = jobs[jobs.title.notna() & (jobs.description.fillna("").str.len() >= 700)]
    jobs = jobs.drop_duplicates(["company_name", "title"]).reset_index(drop=True)
    pats = {c: re.compile(p) for c, p in CATEGORY_TITLES.items()}
    low = jobs.title.str.lower()
    match = pd.DataFrame({c: low.str.contains(p.pattern.replace("(", "(?:")) for c, p in pats.items()})

    # Shared pool (~1,200 postings): 25 on-target per category + 600 random postings.
    pool = set()
    for c in CATEGORY_TITLES:
        pool |= set(rng.choice(np.flatnonzero(match[c].to_numpy()), POOL_ON, replace=False))
    pool |= set(rng.choice(len(jobs), POOL_RANDOM, replace=False))
    pool = np.array(sorted(pool))
    in_pool = np.zeros(len(jobs), bool)
    in_pool[pool] = True

    # Candidate lists: same recipe for every resume, fixed by the seed. Off-target postings
    # come from the whole pool, so some are other categories' on-target jobs (harder negatives).
    rows = []
    for i, r in res.iterrows():
        m = match[r.Category].to_numpy()
        on = rng.choice(np.flatnonzero(m & in_pool), N_ON, replace=False)
        off = rng.choice(np.flatnonzero(~m & in_pool), N_OFF, replace=False)
        rows += [(i, j, 1) for j in on] + [(i, j, 0) for j in off]
    pairs = pd.DataFrame(rows, columns=["r", "j", "on_target"])
    used = np.sort(pairs.j.unique())
    jobs = jobs.iloc[used].reset_index(drop=True)
    pairs["j"] = pairs.j.map({old: new for new, old in enumerate(used)})
    return res, jobs, pairs


def jobswipe_vectors(res, jobs):
    key = CACHE / "in_domain_jobswipe.npz"
    if key.exists():
        z = np.load(key)
        return z["R"], z["J"]
    src, dst = CACHE / "in_domain_js_in.json", CACHE / "in_domain_js_out.json"
    src.write_text(json.dumps({
        "resumes": res.Resume_str.tolist(),
        "jobs": [{"title": t, "description": d, "location": l if isinstance(l, str) else None,
                  "is_remote": bool(rm == 1.0)}
                 for t, d, l, rm in zip(jobs.title, jobs.description, jobs.location, jobs.remote_allowed)],
    }))
    subprocess.run(["node", str(ROOT / "eval" / "jobswipe_texts.mts"), str(src), str(dst)], check=True, cwd=ROOT)
    texts = json.loads(dst.read_text(encoding="utf-8"))
    profiles = texts["profiles_suggested"]
    print(f"  JobSwipe found a title for {sum(p['header'].startswith('Role:') for p in profiles)}"
          f"/{len(profiles)} resumes")

    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("thenlper/gte-small", device="cpu")
    m.max_seq_length = 512
    flat = sorted({t for p in profiles for t in [p["header"], *p["chunks"]] if t} | set(texts["jobs"]))
    print(f"  embedding {len(flat):,} strings with gte-small ...", flush=True)
    vec = dict(zip(flat, m.encode(flat, batch_size=16, normalize_embeddings=True, show_progress_bar=True)))

    def prof(p):
        h, ch = p["header"], p["chunks"]
        parts = ([(h, .5)] + [(c, .5 / len(ch)) for c in ch]) if h and ch else [(h, 1.)] if h else \
                [(c, 1 / len(ch)) for c in ch]
        v = sum(w * vec[t] for t, w in parts)
        return v / np.linalg.norm(v)
    R = np.stack([prof(p) for p in profiles])
    J = np.stack([vec[t] for t in texts["jobs"]])
    np.savez_compressed(key, R=R, J=J)
    return R, J


def jobtinder_scores(res, jobs, pairs):
    key = CACHE / "in_domain_jobtinder_vecs.npz"
    jt_texts = [job_text(t, c if isinstance(c, str) else None, d[:4000])
                for t, c, d in zip(jobs.title, jobs.company_name, jobs.description)]
    if key.exists():
        z = np.load(key)
        R, J = z["R"], z["J"]
    else:
        enc = get_encoder()
        print(f"  embedding {len(res)} resumes + {len(jobs):,} postings with bge-small ...", flush=True)
        R, J = enc.encode_resumes(res.Resume_str.tolist()), enc.encode_jobs(jt_texts)
        np.savez_compressed(key, R=R, J=J)
    art = joblib.load(JT / "models" / "fit_artifacts.joblib")
    booster = lgb.Booster(model_file=str(JT / "models" / "fit_model.txt"))
    rt = [res.Resume_str[i] for i in pairs.r]
    jt = [jt_texts[j] for j in pairs.j]
    tf = np.asarray(normalize(art["vectorizer"].transform(rt)).multiply(
        normalize(art["vectorizer"].transform(jt))).sum(axis=1)).ravel()
    X = build_matrix([ResumeSide(text=t, embedding=R[i]) for t, i in zip(rt, pairs.r)],
                     [JobSide(text=t, embedding=J[j]) for t, j in zip(jt, pairs.j)],
                     tfidf_cosines=tf, skill_idf=art["skill_idf"])
    from ml.features import ALL_FEATURE_NAMES
    cols = [ALL_FEATURE_NAMES.index(n) for n in art["feature_names"]]
    return fit_score(booster.predict(X[:, cols]))


def per_resume(pairs, score, mismatch):
    out = []
    for _, g in pairs.assign(s=score, mm=mismatch).groupby("r"):
        top = g.nlargest(TOP, "s")
        out.append((roc_auc_score(g.on_target, g.s), top.on_target.mean(),
                    ndcg_score([g.on_target.to_numpy()], [g.s.to_numpy()], k=TOP), top.mm.mean()))
    return np.array(out)


def main():
    t0 = time.time()
    print("Sampling resumes and postings ...")
    res, jobs, pairs = load_sample()
    print(f"  {len(res)} resumes x {N_ON + N_OFF} candidates = {len(pairs):,} pairs, "
          f"{len(jobs):,} unique postings")

    scores = {"random": np.random.default_rng(SEED).random(len(pairs))}
    tv = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit(
        pd.concat([res.Resume_str, jobs.title + " " + jobs.description]))
    Rt = normalize(tv.transform(res.Resume_str))
    Jt = normalize(tv.transform(jobs.title + " " + jobs.description))
    scores["tfidf_cosine"] = np.asarray(Rt[pairs.r.to_numpy()].multiply(Jt[pairs.j.to_numpy()]).sum(axis=1)).ravel()
    print("JobSwipe ...")
    R, J = jobswipe_vectors(res, jobs)
    scores["jobswipe_gte_cosine"] = np.einsum("ij,ij->i", R[pairs.r], J[pairs.j])
    print("Job-Tinder (shipped model) ...")
    scores["jobtinder_lightgbm"] = jobtinder_scores(res, jobs, pairs)

    # seniority mismatch flag per pair
    yrs = np.array([resume_years(t) for t in res.Resume_str])
    lvl = jobs.formatted_experience_level.fillna("").to_numpy()
    ry, jl = yrs[pairs.r], lvl[pairs.j]
    mismatch = (((ry <= 2) & np.isin(jl, ["Director", "Executive"])) |
                ((ry >= 10) & np.isin(jl, ["Internship", "Entry level"]))).astype(float)

    rng = np.random.default_rng(SEED)
    boot = rng.integers(0, len(res), (BOOT, len(res)))
    per = {m: per_resume(pairs, scores[m], mismatch) for m in MODELS}
    names = ["auc", "precision_at_10", "ndcg_at_10", "seniority_mismatch_at_10"]
    report = {"n_resumes": len(res), "n_pairs": int(len(pairs)), "n_postings": int(len(jobs)),
              "candidates_per_resume": {"on_target": N_ON, "off_target": N_OFF}, "models": {}, "gaps_vs_baseline": {},
              "auc_by_category": {}}
    for m in MODELS:
        report["models"][m] = {n: {"mean": round(float(per[m][:, k].mean()), 4),
                                   "ci": [round(float(x), 4) for x in np.percentile(per[m][boot, k].mean(1), [2.5, 97.5])]}
                               for k, n in enumerate(names)}
    for m in ("jobswipe_gte_cosine", "jobtinder_lightgbm"):
        report["gaps_vs_baseline"][m] = {n: {"mean": round(float((per[m][:, k] - per["tfidf_cosine"][:, k]).mean()), 4),
                                             "ci": [round(float(x), 4) for x in np.percentile(
                                                 (per[m][boot, k] - per["tfidf_cosine"][boot, k]).mean(1), [2.5, 97.5])]}
                                         for k, n in enumerate(names)}
    d = per["jobtinder_lightgbm"] - per["jobswipe_gte_cosine"]
    report["jobtinder_minus_jobswipe"] = {n: {"mean": round(float(d[:, k].mean()), 4),
                                              "ci": [round(float(x), 4) for x in np.percentile(d[boot, k].mean(1), [2.5, 97.5])]}
                                          for k, n in enumerate(names)}
    for cat, idx in res.groupby("Category").groups.items():
        report["auc_by_category"][cat] = {m: round(float(per[m][list(idx), 0].mean()), 3) for m in MODELS}
    report["duration_seconds"] = round(time.time() - t0, 1)
    OUT.mkdir(exist_ok=True)
    (OUT / "in_domain_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n{'model':<22}" + "".join(f"{n:>30}" for n in names))
    for m in MODELS:
        print(f"{m:<22}" + "".join(f"{report['models'][m][n]['mean']:>10.3f} [{report['models'][m][n]['ci'][0]:.3f},{report['models'][m][n]['ci'][1]:.3f}]"
                                   for n in names))
    for k, v in {**report["gaps_vs_baseline"], "jobtinder - jobswipe": report["jobtinder_minus_jobswipe"]}.items():
        print(f"gap {k:<26}" + "".join(f"{v[n]['mean']:>+10.3f} [{v[n]['ci'][0]:+.3f},{v[n]['ci'][1]:+.3f}]" for n in names))
    print("\nAUC by category:")
    for cat, v in report["auc_by_category"].items():
        print(f"  {cat:<24}" + "  ".join(f"{m[:10]} {v[m]:.3f}" for m in MODELS[1:]))
    print(f"\nSaved results/in_domain_check.json ({report['duration_seconds']}s)")


if __name__ == "__main__":
    main()
