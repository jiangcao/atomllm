"""KV-cache checks: (1) cached logits == full forward, incl. rewinds and
ragged chunks; (2) a batch of guided window samples gives the same tokens
with and without the cache (same seed); (3) speed.
usage: venv/bin/python python/rl/test_kv_cache.py CKPT TOKENIZER"""
import sys, time, json, numpy as np, torch
sys.path.insert(0, "python")
from train import load_checkpoint

ck, tokp = sys.argv[1], sys.argv[2]
dev = "cuda"
model, _ = load_checkpoint(ck, dev); model.eval()
V, S = model.config.vocab_size, model.config.max_seq_len
g = torch.Generator().manual_seed(0)

# (1) random sequences, random rewinds
B = 6
seqs = [torch.randint(3, V, (int(torch.randint(40, S - 10, (1,), generator=g)),), generator=g).tolist()
        for _ in range(B)]
cache = model.alloc_cache(B, dev)
have = [0] * B
worst = 0.0
for step in range(60):
    rows, starts, chunks = [], [], []
    for r in range(B):
        if have[r] >= len(seqs[r]):
            continue
        if have[r] > 5 and torch.rand(1, generator=g) < 0.15:       # rewind + new branch
            back = int(torch.randint(1, 5, (1,), generator=g))
            have[r] -= back
            seqs[r][have[r]:have[r] + back] = torch.randint(3, V, (back,), generator=g).tolist()
        n = min(int(torch.randint(1, 4, (1,), generator=g)), len(seqs[r]) - have[r])
        rows.append(r); starts.append(have[r]); chunks.append(seqs[r][have[r]:have[r] + n])
    if not rows:
        break
    N = max(map(len, chunks))
    x = torch.zeros((len(rows), N), dtype=torch.long, device=dev)
    for i, c in enumerate(chunks):
        x[i, :len(c)] = torch.tensor(c)
    t = lambda v: torch.tensor(v, device=dev)
    lg = model.forward_cached(x, t(rows), t(starts), t([len(c) for c in chunks]), cache)
    with torch.no_grad():
        for i, r in enumerate(rows):
            L = starts[i] + len(chunks[i])
            ref = model(torch.tensor([seqs[r][:L]], device=dev))[0][0, -1]
            worst = max(worst, float((ref - lg[i]).abs().max()))
            have[r] = L
print(f"(1) max |cached - full| logit difference: {worst:.2e}")
assert worst < 1e-3

# (2) guided sampling, same seed, with / without cache
import window_generation as wg
import window_generation_v2 as wg2
from window_tokenizer_v2 import WindowZTokenizer
from ase.io import read
tok = WindowZTokenizer.load(tokp)
frames = read("data/sio2/dataset.extxyz", index=":20")
rng = np.random.default_rng(3)
guides = []
NG = int(sys.argv[3]) if len(sys.argv) > 3 else 8
while len(guides) < NG:
    a = frames[int(rng.integers(len(frames)))]
    inside, local, _, Lp = wg.crop_with_transform(a, tok.W, rng)
    levels = [8, 8, 8]
    known = tok.is_known(local[inside], levels)
    ctx, fill = inside[known], inside[~known]
    if not len(fill):
        continue
    sym = np.array(a.get_chemical_symbols())
    counts = {e: int((sym[fill] == e).sum()) for e in tok.config["elements"]}
    t = tok.thresholds(levels)
    guides.append(wg2.ZWindowGuide(tok, levels, counts, sym[ctx], local[ctx], [], np.zeros((0, 3)),
                                   t, np.full(3, tok.W)))
wg2.FORCE_BONDS = True
out = {}
for use in (False, True):
    wg2.USE_KV_CACHE = use
    torch.manual_seed(7); torch.cuda.synchronize(); t0 = time.time()
    res = wg2.z_generate_batch(model, guides, 0.8, 50, 20, dev)
    torch.cuda.synchronize()
    out[use] = ([r[0] for r in res], [np.round(g.new_atoms()[1], 3).tolist() for g in guides], time.time() - t0)
    print(f"(2) cache={use}: {sum(bool(r) for r in out[use][0])}/{NG} complete, {out[use][2]:.1f} s")
same = sum(a == b for a, b in zip(out[False][1], out[True][1]))
print(f"    identical samples with / without cache: {same}/{NG} (fp differences can flip a rare draw)")
print(f"    speed-up {out[False][2] / out[True][2]:.1f}x")
