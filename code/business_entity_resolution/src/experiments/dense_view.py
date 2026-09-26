"""Dense multilingual retrieval view ("dn"): multilingual-e5-small embeddings, reverse
target -> S1 top-k on the GPU. A second, learned answer to the cross-script blocking gap
(the xs view is the dictionary-based answer). Both write the same file format, so
xscript.py `measure`/`merge --views xs,dn` handle either or both.

Evidence:
- [M] ours: 53% of India blocking misses have an Indic-script target name.
- [S] Karthikatstuffmama (3% dev slice): e5-small dense top-15 unioned with key blocking raised
  recall from 0.974 to 0.993 and holdout F0.5 by +0.0065 (India +0.0128); denser non-Latin retrieval
  (rank 50) added a little more. Dev-slice absolutes are optimistic; only deltas are comparable.

Model: intfloat/multilingual-e5-small (MIT licence, 118M parameters; the challenge allows
MIT/Apache ≤8B). Pretrained weights are downloaded once from Hugging Face. No business data
is looked up anywhere. Text: "query: <raw name>, <raw address>" (raw text keeps the native
script, which the model understands); mean pooling, L2-normalized, fp16.

Scope: by default only targets whose raw name has a non-Latin letter (the measured gap),
against ALL S1 of the country. Use `--scope all` for every target (≈20M texts per split, hours).

  python src/experiments/dense_view.py build --split train [--k 15] [--scope nonlatin|all]
      -> work/dense/<split>/{s1,tg}_<country>.npy (+ _ids.npy; embeddings, resumable)
         work/block/<split>/dn_<country>.parquet  (rid, gid, r_dn, s_dn)
  python src/experiments/dense_view.py features --split train     (after run.py features)
      -> work/featdn/<split>/<part>.parquet: dn_cos, row-aligned extra set for BER_EXTRA=featdn
Requirements: a CUDA GPU (CPU works but is slow) and `pip install -r requirements-dense.txt`
in a separate venv (keeps the CPU pipeline's environment unchanged).
Cost (RTX 3050 laptop 4 GB): not yet measured on real data (synthetic: 1.6–4.8k texts/s incl. warm-up);
nonlatin train ≈1.8M texts (India S1 + non-Latin targets). GPU search is seconds to minutes.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ber import config  # noqa: E402
from ber.utils import done, log, mark_done, stage  # noqa: E402

MODEL = os.environ.get("BER_DENSE_MODEL", "intfloat/multilingual-e5-small")
NONLATIN = r"[\p{L}--\p{Latin}]"
MAX_LEN = 64


def load_model(device):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, dtype=torch.float16 if device == "cuda" else torch.float32)
    return tok, model.to(device).eval()


def encode(texts, tok, model, device, batch=512):
    """Mean-pooled, L2-normalized fp16 embeddings; texts are processed in length order."""
    import torch
    order = np.argsort([len(t) for t in texts], kind="stable")
    out = np.empty((len(texts), model.config.hidden_size), dtype=np.float16)
    t0 = time.time()
    with torch.inference_mode():
        for s in range(0, len(texts), batch):
            idx = order[s:s + batch]
            enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=MAX_LEN,
                      return_tensors="pt").to(device)
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            e = (h * m).sum(1) / m.sum(1).clamp(min=1)
            e = torch.nn.functional.normalize(e.float(), dim=-1)
            out[idx] = e.to(torch.float16).cpu().numpy()
            if s // batch % 200 == 0 and s:
                log(f"  encoded {s:,}/{len(texts):,} ({s / (time.time() - t0):,.0f} texts/s)")
    return out


def cached_encode(path, texts, tok_model, device):
    if done(path):
        return np.load(path, mmap_mode="r")
    tok, model = tok_model()
    t0 = time.time()
    emb = encode(texts, tok, model, device)
    np.save(path, emb)
    mark_done(path)
    log(f"encoded {len(texts):,} texts in {time.time() - t0:.0f}s ({len(texts) / max(1e-9, time.time() - t0):,.0f}/s) -> {path.name}")
    return emb


def search(Q, S, k, device, q_batch=2048, s_chunk=131072):
    """Exact top-k inner product of each row of Q among rows of S (both normalized)."""
    import torch
    k = min(k, S.shape[0])
    qs, its, scs = [], [], []
    S_chunks = [(s0, torch.from_numpy(np.ascontiguousarray(S[s0:s0 + s_chunk])).to(device))
                for s0 in range(0, S.shape[0], s_chunk)]
    for q0 in range(0, Q.shape[0], q_batch):
        q = torch.from_numpy(np.ascontiguousarray(Q[q0:q0 + q_batch])).to(device)
        best_v, best_i = None, None
        for s0, Sc in S_chunks:
            v, i = torch.topk(q @ Sc.T, min(k, Sc.shape[0]), dim=1)
            i = i + s0
            if best_v is None:
                best_v, best_i = v, i
            else:
                v2, j = torch.topk(torch.cat([best_v, v], 1), k, dim=1)
                best_i = torch.gather(torch.cat([best_i, i], 1), 1, j)
                best_v = v2
        n = best_v.shape[0]
        qs.append(np.repeat(np.arange(q0, q0 + n), best_v.shape[1]))
        its.append(best_i.cpu().numpy().ravel())
        scs.append(best_v.float().cpu().numpy().ravel())
    q, it, sc = np.concatenate(qs), np.concatenate(its), np.concatenate(scs)
    rank = np.tile(np.arange(k), len(q) // k) if len(q) % k == 0 else None
    return q, it, sc, rank


def build(split, k, scope):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"dense view: model {MODEL}, device {device}, scope {scope}, k {k}")
    W = config.WORK_DIR
    ddir, bdir = W / "dense" / split, W / "block" / split
    ddir.mkdir(parents=True, exist_ok=True)
    bdir.mkdir(parents=True, exist_ok=True)
    s1 = pl.read_parquet(W / "prep" / f"s1_{split}.parquet", columns=["rid", "country", "fold", "name", "addr"])
    tg = pl.scan_parquet(W / "prep" / f"tg_{split}.parquet").select("gid", "country", "name", "addr")
    if scope == "nonlatin":
        tg = tg.filter(pl.col("name").str.contains(NONLATIN))
    tg = tg.collect()
    text = lambda df: ("query: " + df.get_column("name") + ", " + df.get_column("addr")).to_list()
    cache = {}

    def tok_model():
        if "m" not in cache:
            cache["m"] = load_model(device)
        return cache["m"]

    for country in sorted(tg.get_column("country").unique()):
        out = bdir / f"dn_{country}.parquet"
        if done(out):
            log(f"dn {split}/{country}: cached")
            continue
        s1c, tgc = s1.filter(pl.col("country") == country), tg.filter(pl.col("country") == country)
        if s1c.height == 0 or tgc.height == 0:
            continue
        with stage(f"dn_build_{split}_{country}", {"targets": tgc.height, "s1": s1c.height, "scope": scope}):
            S = cached_encode(ddir / f"s1_{country}.npy", text(s1c), tok_model, device)
            T = cached_encode(ddir / f"tg_{scope}_{country}.npy", text(tgc), tok_model, device)
            np.save(ddir / f"s1_{country}_ids.npy", s1c.get_column("rid").to_numpy())
            np.save(ddir / f"tg_{scope}_{country}_ids.npy", tgc.get_column("gid").to_numpy())
            tq, sq, sc, rk = search(T, S, k, device)
            if rk is None:
                rk = (pl.DataFrame({"q": tq, "s": sc}).with_columns(
                    r=pl.col("s").rank("ordinal", descending=True).over("q") - 1).get_column("r").to_numpy())
            qmask = (s1c.get_column("fold").is_in(config.QUERY_FOLDS).to_numpy() if split == "train"
                     else np.ones(s1c.height, bool))
            keep = qmask[sq]
            res = pl.DataFrame({
                "rid": s1c.get_column("rid").to_numpy()[sq[keep]].astype(np.int32),
                "gid": tgc.get_column("gid").to_numpy()[tq[keep]].astype(np.int32),
                "r_dn": rk[keep].astype(np.int16), "s_dn": sc[keep].astype(np.float32)})
            res.write_parquet(out)
            mark_done(out)
            log(f"dn {split}/{country}: {res.height:,} pairs")


OUT_SET = "featdn"


def features(split, scope):
    """Extra feature set: dn_cos = e5 cosine of the pair (NaN when the target has no embedding,
    i.e. outside --scope). Row-aligned with work/feat/<split>/<part>.parquet, so any matcher can
    use it with BER_EXTRA=featdn (combine: BER_EXTRA=featx,featdn). CPU only, ~1–2 GB RAM."""
    W = config.WORK_DIR
    ddir, src, out = W / "dense" / split, W / "feat" / split, W / OUT_SET / split
    out.mkdir(parents=True, exist_ok=True)
    s_emb, s_ids, t_emb, t_ids = [], [], [], []
    for f in sorted(ddir.glob("s1_*_ids.npy")):
        c = f.name[3:-8]
        tf = ddir / f"tg_{scope}_{c}.npy"
        if not tf.exists():
            continue
        s_ids.append(np.load(f)); s_emb.append(np.load(ddir / f"s1_{c}.npy", mmap_mode="r"))
        t_ids.append(np.load(ddir / f"tg_{scope}_{c}_ids.npy")); t_emb.append(np.load(tf, mmap_mode="r"))
    if not s_ids:
        sys.exit(f"no embeddings in {ddir}; run `build --split {split}` first")
    # global id -> (block, row) lookup via sorted ids
    def index(ids_list):
        ids = np.concatenate(ids_list)
        blk = np.concatenate([np.full(len(x), i, np.int32) for i, x in enumerate(ids_list)])
        row = np.concatenate([np.arange(len(x)) for x in ids_list])
        o = np.argsort(ids)
        return ids[o], blk[o], row[o]
    si, sb, sr = index(s_ids)
    ti, tb, tr = index(t_ids)

    def lookup(sorted_ids, q):
        pos = np.clip(np.searchsorted(sorted_ids, q), 0, len(sorted_ids) - 1)
        return pos, sorted_ids[pos] == q

    for p in sorted(src.glob("*.parquet")):
        dst = out / p.name
        if done(dst):
            continue
        base = pl.read_parquet(p, columns=["rid", "gid", "fold"])
        rid, gid = base.get_column("rid").to_numpy(), base.get_column("gid").to_numpy()
        ps, hs = lookup(si, rid)
        pt, ht = lookup(ti, gid)
        cos = np.full(len(rid), np.nan, np.float32)
        ok = hs & ht & (sb[ps] == tb[pt])
        for b in np.unique(sb[ps][ok]):
            m = ok & (sb[ps] == b)
            a = np.asarray(s_emb[b][np.sort(np.unique(sr[ps][m]))])
            ua = np.sort(np.unique(sr[ps][m]))
            ub = np.sort(np.unique(tr[pt][m]))
            B = np.asarray(t_emb[b][ub])
            ia, ib = np.searchsorted(ua, sr[ps][m]), np.searchsorted(ub, tr[pt][m])
            cos[m] = np.einsum("ij,ij->i", a[ia].astype(np.float32), B[ib].astype(np.float32))
        base.with_columns(dn_cos=pl.Series(cos)).write_parquet(dst)
        mark_done(dst)
        log(f"{OUT_SET} {split}/{p.name}: {len(rid):,} pairs, {int(np.isfinite(cos).sum()):,} with dn_cos")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "features"])
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--scope", choices=["nonlatin", "all"], default="nonlatin")
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.split, a.k, a.scope)
    else:
        features(a.split, a.scope)


if __name__ == "__main__":
    main()
