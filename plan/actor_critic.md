# Actor-critic fine-tuning of the v2 window model

Status (2026-11-20): plan approved in general; step 0 (rejection-sampling
fine-tuning) and the KV cache are being implemented first.

## 1. Setup

**Episode** = completion of one window (100-250 tokens). The defect judge
already exists for this and it is exactly the step surface growth repeats.

Prompt pools:
- **MD crops**: a window cut from a real MD frame with the fill region emptied
  (as in `data/surface/fair_complete.py`); the surroundings are real, so the
  judge is exact. Training runs only (validation runs stay for evaluation).
- **Growth states**: context + env atoms dumped from real surface-growth runs
  (substrate, interface, rejected cases), so training targets where growth
  actually fails. Needs a `--dump_states` option in the growth driver.

**Actor**: the v2 model, always sampled through the guide masks used in
generation (clash, overcoordination, gap/near-miss, force-bonds, env completion).
The policy is the softmax over allowed tokens only. Sample at temperature 1, no
top-k, so log-probs are exact.

**Critic**: a small head on the last hidden state. At every token it predicts a
vector of final-structure quantities q̂:
- defect counts by type: dangling Si, non-bridging O, over-coordinated Si / O,
  dead ends;
- window statistics: mean deviation of Si-O length, O-Si-O and Si-O-Si angles
  from the MD distributions; optionally MACE energy per atom after a short relax.

Value V = w · q̂. Separate quantities are easier to learn and inspect, and the
weights can change without retraining the critic.

## 2. Reward

| term | how | default weight |
|---|---|---|
| defects | `find_defects` on atoms whose coordination is final (new atoms + ctx atoms next to the fill) | -1.0 per defect |
| dead ends | `dead_ends` on open atoms (0.1 % false alarms in bulk) | -0.5 |
| geometry | mean log-likelihood of new bond lengths / angles under MD histograms, centred so MD scores 0 | +0.3 |
| energy (optional) | batched MACE energy/atom after 20-step relax, relative to MD mean | -0.2 per eV/atom |
| KL to the start model | per token, frozen reference | beta 0.02-0.05, adaptive |

**Credit assignment** (uses the v2 grammar): `[NEXT]` is the model's decision
that the focus atom is finished.
- an under-coordinated atom is penalised at its own `[NEXT]`;
- an over-coordinated atom or near miss is penalised at the position tokens of
  the atom that caused it;
- everything else gets the terminal reward; the critic covers the rest.

**Reward hacking**: atom counts are fixed by the guide, so the model cannot
place fewer atoms. The risk is drift towards ordered / crystal-like motifs.
Counter: KL penalty, MD-matched geometry term, and ring statistics + RDF/ADF in
evaluation.

## 3. Algorithm

PPO with GAE, token-level advantages.
- **Rollout**: P prompts x K = 8 samples via the lock-step batched sampler
  (`z_generate_batch`, with KV cache). Record per step: token, allowed mask,
  old log-prob, value.
- **Update**: one teacher-forced forward per sequence recomputes all log-probs
  under the stored masks (masks are deterministic given the sequence). Clipped
  policy loss + per-quantity value MSE + entropy bonus; 2-4 epochs of
  minibatches.
- **Baseline**: GAE gamma = 1, lambda = 0.95. Ablation: group-mean baseline
  over the K samples (GRPO-style, critic-free) to check the critic earns its keep.

## 4. Implementation steps (`python/rl/`)

0. **Control: rejection-sampling fine-tuning (RFT)** - generate many windows,
   keep the defect-free ones, fine-tune on them (+ replay of the original data).
   Shows how much headroom there is; PPO must beat it.
   - `rl/prompts.py` (MD-crop prompts), `rl/judge.py`, `rl/rft.py`.
1. **Infrastructure**
   - KV cache in `AtomicGPT` (`forward_cached`), used by `z_generate_batch`;
     sampling no longer recomputes the whole sequence every token.
   - `rl/rollout.py`: log-prob-recording version of `z_generate_co`. Tests:
     recomputed log-prob == sampled log-prob; masked probabilities sum to 1.
   - `rl/reward.py`: reward vector, MD reference histograms (built once from
     `data/structures/SiO2` and the interface frames), per-atom attribution to
     `[NEXT]` / position tokens.
   - growth-state prompts (`--dump_states` in the growth driver).
2. **Critic warm-up** with the actor frozen: first offline on the training
   sequences (defect labels and statistics known -> every prefix has a target),
   then on rollouts.
3. **PPO loop** (`rl/ppo.py`): adaptive KL, checkpoints, per-iteration log
   (reward components, KL, entropy, clean-window rate). MD crops first, then
   50 % growth states.
4. **Evaluation** every N iterations:
   - `fair_complete.py` defect rates on its 4 window shapes;
   - RDF, bond-angle distributions, ring statistics vs MD;
   - sample diversity;
   - 2x2 growth benchmark at fixed seed: reject rate and backtracks
     (reference: 34 % clean samples, 23 backtracks).

## 5. Cost

KV cache should make rollouts ~10x faster. Batch of 64 prompts x 8 samples
~1-2 min + reward (fast without MACE). A few hundred PPO iterations: ~6-12 GPU
hours. Steps 0-2 fit in a day.

## Decisions

1. MACE energy in the reward: off at first, on later (3x slower reward).
2. Prompt mix: MD crops first, then growth states.
3. Start model: the chain model trained on all 128 interface runs
   (`checkpoints/sio2_iface_win8_z/checkpoint_best_step088000_loss2.3591.pt`,
   finished 2026-11-20 09:17).

## 6. Self-training with MACE (added 2026-11-20)

Evidence that guides the data choices (reference 2x2 growth, `surface_v2_2x2_new`):
- Rejection culprits: earlier window 59, this window 56, substrate 13, 1x1 patch 3;
  failure types dead end 60, Si dangling 41, O non-bridging 26 -> the oxide network
  fails, not mainly the interface. More interface MD is not the first lever.
- 4x more interface data (32 -> 128 runs): val loss 2.391 -> 2.359, but the fair
  completion test did not improve (~24-28 % defective core atoms vs 0 % for MD).
  The limit is exposure bias, not data volume -> train on the model's own outputs.

**(a) Relaxed RFT** (volume): in `rl/rft.py`, relax each clean window with MACE
(ctx + env fixed, ~50 FIRE steps, batched), re-judge, re-encode with
`encode_window` on the relaxed coordinates and train on those. The model learns
MACE-corrected geometry instead of its own raw samples. Consistent with the MD
data (training frames are MACE-relaxed). Compare against plain RFT (step 0).

**(b) Anneal grown structures** (validation + surface-specific data):
`data/surface/anneal_grown.py` - frozen lateral rim + substrate bottom, relax ->
300 K 2 ps -> ramp -> 1100 K 5 ps -> cool 5 ps -> relax; defects and Si-O bond
changes counted in the core (interface + oxide interior). A structure that stays
defect-free through the anneal is evidence of a really trap-free structure. The
annealed frames are surface/interface data of the kind actually generated: crop
windows from them (core only) and add with extra weight.

Guard rails for both: >= 50 % original MD data in every training mix, cap per
prompt, ring statistics / RDF / ADF and diversity checks (reward-hacking section).

Order: step-0 result -> (b) on the reference 2x2 now -> (a) if step 0 helps.

## 7. Physicality report (added 2026-11-20)

`python/physicality.py FILE[@INDEX] ... [--mace]`, compared with 60 bulk + 60
interface MD validation frames (cached `data/physicality_ref.npz`):
level 1 coordination / defects / Si-O, O-Si-O, Si-O-Si distributions / edge
sharing; level 2 shortest-path rings, density, g(r); interface Si1+..Si3+ per
nm^2; level 3 MACE per-atom energies (SiO4 Si, O) vs reference. Used as the
evaluation in step 4 and as the critic's "statistical quantities". First result
(reference 2x2 growth): coordination-clean but strained as grown (edge-sharing
12.7 vs 4.0 /100 Si, 30 % Si-O-Si < 120 deg vs 12 %, SiO4 Si +0.87 eV); MD anneal
brings it close to MD (7.3 /100 Si, 13.5 %, +0.16 eV). -> add strain terms
(edge sharing, small Si-O-Si, 2/3-rings) to the reward, and relax/anneal before
using generated structures as data.

## 8. Implemented: actor-critic on real growth episodes (2026-11-20, run ppo1)

Decision (user): skip DPO, use real growth runs, go to actor-critic directly.
Why (context test, data/surface/context_test.py): with complete surroundings the
model fills growth-shaped windows well (72 % clean MD context, 84 % grown, 94 %
annealed; 0 % of defects involve atoms outside the window), so window size is not
the problem. What it lacks is the future: training never shows a partial
structure that cannot be completed, but in growth the space beyond the fill box
is empty and is filled later by the model itself (dead ends = most rejects).
Density-based fill counts cost another ~9 points (72 -> 63 %).

- Episode: one 1x1 (10.86 Å) growth run with the full settings (relax-judge, dead
  ends, gap mask, repair, 8 candidates, backtracking), substrate = last frame of a
  random interface MD training run. ~2-3 min, ~290 candidates.
- Action: every sampled token; policy = masked softmax(logits / 0.8), no top-k
  (recorded allowed set + log-prob; recomputed = sampled to 1.4e-5).
- Reward per candidate: raw defects -1, dead ends -0.5; accepted: -blame of a later
  backtrack it caused (1 / 0.5) - interior defects left at the end; rejected: 0
  future; clean but not picked: critic bootstrap.
- Critic: value head on detached hidden states, (immediate, future) per token.
- PPO: clip 0.2, 2 epochs, lr 1e-5 (value 1e-3), KL to start model beta 0.02,
  entropy 0.001; iteration 0 critic only.
- Code: python/rl/recorder.py, python/rl/ppo.py; hooks in window_generation.py
  (RECORDER) and window_generation_v2.py (--record, --top_k 0, RECORD_ACTIONS).
Next: evaluate checkpoints on the val window test, the context test and the 2x2
benchmark; then dense per-token credit ([NEXT] / placement attribution), strain
terms (edge sharing, Si-O-Si < 120 deg, small rings), learned fill counts.
