# Final Implementation Guide — From Image Classification to Image Captioning

**Group 5 · members Nuno, Francisco and Artur · written Fri 27 Nov 2026 · 1.5-week plan: work Sat 28 Nov – Mon 7 Dec, submit Mon 7 Dec · official deadline Fri 11 Dec (8–11 Dec = contingency only) · oral in the week of 14 Dec**

> **Timeline at a glance.** S0 contracts Sat 28 Nov → S1 independent work Sun 29 Nov – Tue 1 Dec (gate G1 + bonus decision Tue 1 Dec) → S2 integration Wed 2 – Fri 4 Dec (**experiment freeze Fri 4 Dec**) → S3 report Sat 5 – Mon 7 Dec (**submit Mon 7 Dec**). Items marked **[if ahead]** are cut from the core plan and done only if their owner is ahead of schedule.

This is the one plan all three of us follow until submission. For every step it states **who** owns it, **when** it runs, **why** we chose the approach, the **theory** it relies on, the **hypotheses** it must confirm or reject, and exactly **what goes into** the matching `\paragraph{Qx.y.}` of `report/Deep_Learning_2026_27__Course_Project_Report_Group_5/main.tex`.

---

## 0. Ground rules (apply to every step)

| Rule | Concretely |
|---|---|
| Allowed stack (Parts 2–3) | Python ≥ 3.10, `torch`, `torchvision.transforms` (resize/augment only), `matplotlib` (plots only), `tqdm`, the skeleton. **No** `numpy` (not even `.numpy()`), `sklearn`, Lightning/Keras/HF Trainer, `torchvision.models`, `timm`, `transformers`, pretrained weights or tokenizers. Stdlib (`json`, `argparse`, `pathlib`) is fine — the skeleton uses it. |
| Seeds | Always `0, 1, 2` (`set_seed(s)` before building the model **and** the loaders). Search runs may use seed 17, as Nuno did in Q2.2 — say so. |
| Splits | Train fits; **validation decides everything** (hyperparameters, early stopping, architecture, threshold choices); **test is read once**, for the final configuration only. Never justify a choice with a test number. |
| Statistics | Report `mean ± sample std` over 3 seeds (`torch.tensor(v).std(unbiased=True)`, already in `mean_std()`). |
| Comparing two conditions | Same seeds ⇒ **paired differences** $\Delta_s = m_{B,s}-m_{A,s}$. Report $\bar\Delta \pm s_\Delta$. Call an effect **consistent** only if all three $\Delta_s$ share a sign and $\lvert\bar\Delta\rvert > s_\Delta$; otherwise write "within seed noise". This is our single decision rule for every hypothesis below. |
| Reproducibility | Every number/figure in the report is produced by `src/run_qX_Y.py`, which writes `results/qX_Y/summary.json` (+ figures). Report figures are exported as PDF to `report/Deep_Learning_2026_27__Course_Project_Report_Group_5/figures/`. Log `device` in every JSON. |
| Skeleton changes | Allowed, but each one goes in `CHANGES.md` and in the report. |
| Hardware | Part 2 on laptop CPUs (MLP epochs take ~1 s). Part 3 debugged on laptops with `--subset`, **reported** runs on the free Colab GPU. All seeds of one experiment on the same device. |
| AI use | This guide was written with Claude ⇒ declare it in checklist items 1–4 ("Brainstorming ideas", "Designing or analyzing experiments", "Explaining concepts"). Each member keeps notes of their own AI use for items 4–5. |

**Hypothesis protocol.** Every experiment is written as: *H (claim) → prediction (a measurable number and its direction) → test (script + metric) → decision (by the paired rule above) → one sentence in the report.* A rejected hypothesis is a result, not a failure — report it with the explanation.

---

## 1. Where we stand (all work so far by Nuno)

| Question | Status | Result (test, 3 seeds) | Location |
|---|---|---|---|
| Q2.1 linear baseline | Done | 26.73 ± 0.91 % (SGD, lr 0.001, 25 epochs, 122,890 params) | `src/run_q2_1.py`, `results/q2_1/` |
| Q2.2 MLP search | Done | 28.33 ± 0.15 % (512→256, ReLU, Adam 1e-3, 6,425,866 params) | `src/run_q2_2.py`, `results/q2_2/` |
| Q2.3 controlled study | Done | Dropout 0.3 selected → 29.90 ± 1.54 %; per-class accuracies + confusion matrix | `src/run_q2_3.py`, `results/q2_3/` |
| Report | Template only (Q2.1 draft text lives in `results/q2_1/README.md`) | — | `report/…Group_5/main.tex` |

**Follow-ups on Nuno's work (owned by Nuno, Sprint 1, ~1 effort point):**

1. **Q2.1 — selection by validation only.** The README justifies lr 0.001 with test gains. Rewrite the justification with validation numbers; optionally sweep `{0.1, 0.01, 0.001, 0.0001}` on validation (cheap) and read test once for the winner. Delete the duplicate `results/q2_1_lr0001/`.
2. **Q2.3 — equal budget.** Early stopping makes the number of updates differ between conditions. Rerun with `patience=None` (all conditions train exactly 18 epochs; `train()` still restores the best validation epoch). Recompute with paired differences.
3. **Q2.3 — show underfitting too.** Add one condition expected to underfit (e.g. Adam `weight_decay=1e-2`, or hidden width 32) so the discussion contrasts both regimes. *H: train and validation accuracy both drop and the train–val gap closes.*
4. **Q2.3 — the "→ cat" bias.** Many classes are predicted as cat. Check the predicted-class distribution (column sums of the confusion matrix) and the class templates (Q2.4). *H: cat acts as a default class for cluttered close-up scenes; its column sum is well above 100 per model.*
5. **Checkpoints for downstream work.** Save the final MLP (dropout 0.3) for seeds 0–2 and the Q2.1 linear model to `checkpoints/` (gitignored, mirrored to the shared Drive folder). Add `ConfigurableMLP.features(x)` returning the 256-d penultimate activations (`self.network[:-1]`), so Part 3 can use it as an encoder.

---

## 2. Team split — equal load, minimal waiting

**Principle: one theme per member, theory and experiment owned by the same person.** Whoever derives a property also tests it, which makes the report coherent and the oral easy.

| Member | Theme | Owns |
|---|---|---|
| **Nuno** | MLPs and recurrence theory | Q2.1–Q2.3 (done) + follow-ups, **Q2.4**, **Q1.2**, **Q3.4(b)**; runs the MLP-encoder cells of Q3.3; bonus **Q4.2** (if go) |
| **Francisco** | Convolutions and encoders | **Q1.1**, **Q2.5**, **Q3.3(a)**, **Q3.3(b)**, report assembly in LaTeX |
| **Artur** | Language pipeline | Shared infrastructure, **Q3.1**, **Q3.2**, **Q3.4(a)**, README + reproducibility check + submission |

**Effort balance** (1 point ≈ half a day of focused work; estimates):

| Task | Pts | Nuno | Francisco | Artur |
|---|---:|---:|---:|---:|
| Q2.1 + Q2.2 + Q2.3 (done) | 8 | 8 | | |
| Q2.1–Q2.3 follow-ups | 1 | 1 | | |
| Q2.4 MLP inductive biases | 3 | 3 | | |
| Q1.2 gradients through time | 3 | 3 | | |
| Q3.4(b) decoding theory | 1 | 1 | | |
| Q1.1 convolutions | 3 | | 3 | |
| Q2.5 CNN | 4 | | 4 | |
| Q3.3(a) encoder study | 4 | | 4 | |
| Q3.3(b) class mention | 2 | | 2 | |
| Report assembly and page budget | 2 | | 2 | |
| Shared infrastructure (Sprint 0) | 2 | | | 2 |
| Q3.1 tokenizer | 2 | | | 2 |
| Q3.2 RNN captioner | 6 | | | 6 |
| Q3.4(a) beam search | 3 | | | 3 |
| README, fresh-clone reproduction, submission | 2 | | | 2 |
| **Required total (46)** | | **16** | **15** | **15** |
| Bonus Q4.2, encoder swap only (if go on Tue 1 Dec) | 3 | 3 | | |
| ~~Bonus Q4.1~~ | — | dropped in the 1.5-week plan | | |

Points cover the core scope; **[if ahead]** items are not counted. Required work stays balanced (16 / 15 / 15, with Nuno's 8 completed points included). Inside the 1.5-week window, though, Nuno has the least *remaining* work (8 points against 15), so Nuno is also the **floater**: Nuno takes the optional bonus and runs the 6 MLP-encoder runs of Francisco's Q3.3 grid (execution only — Francisco keeps the design and analysis). This uses Nuno's slack to shorten the critical path, which runs through Artur (Q3.2 → Q3.4) and Francisco (Q2.5 → Q3.3). If the group prefers strict equality including the bonus, skip the bonus.

Everyone, without points: writes their own report paragraphs, reviews their partner's pull requests, fills the AI checklist, and joins the oral mock.

**Review pairs** (a PR to `main` needs the partner's approval, and the partner must be able to explain it):
Nuno reviews Francisco (Q1.1 ↔ Q2.4 shift logic) · Francisco reviews Artur (encoders inside the captioner) · Artur reviews Nuno (Q1.2 ↔ LSTM choice, Q3.4(b) ↔ beam results).

---

## 3. Sprint plan (divide and conquer)

| Sprint | Dates | Nuno | Francisco | Artur | Gate (end of sprint) |
|---|---|---|---|---|---|
| **S0 — Contracts** | Sat 28 Nov (1 day) | `inductive_bias.py`; `features()` + MLP/linear checkpoints | `encoders.py` (CNN class + `build_encoder`); LaTeX `figures/` + page-budget comments | `experiment.py`; `tokenizer.py` API; `captioning.py` API stubs; `CHANGES.md` | **G0 (Sat 28 Nov, night):** all interfaces merged on `main` |
| **S1 — Independent work** | Sun 29 Nov – Tue 1 Dec (3 days) | Q2.1–Q2.3 follow-ups; **Q2.4** complete; Q2.1–Q2.4 paragraphs written as results land; Q1.2 draft | **Q2.5** CNN trained (3 seeds, main + zero-padding variant) + checkpoints; shift/permutation on CNN; Q1.1 draft | **Q3.1** complete; **Q3.2** end-to-end with the MLP encoder (Colab, `--subset` first) | **G1 (Tue 1 Dec, night):** CNN checkpoints on Drive; `run_q3_2.py --encoder {mlp,cnn} --freeze` works; **bonus go/no-go** |
| **S2 — Integration** | Wed 2 – Fri 4 Dec (3 days) | Q1.2 + Q3.4(b) final; runs the 6 MLP-encoder Q3.3 runs (frozen + joint) on his Colab; bonus Q4.2 if go | **Q3.3(a)(b)**: runs the 3 CNN-joint runs, owns the design and all analysis; Q1.1 + Q2.5 paragraphs | Q3.2 final 3-seed runs (= the CNN-frozen cell of the Q3.3 grid, shared with Francisco); **Q3.4(a)** beam search | **G2 (Fri 4 Dec, night): experiment freeze**, bonus included |
| **S3 — Report** | Sat 5 – Mon 7 Dec | Paragraphs done Sat 5; review Artur's | Part 3 paragraphs Sat 5; assemble LaTeX and trim to page limit Sun 6 | Q3.1/Q3.2/Q3.4 paragraphs Sat 5; fresh-clone reproduction + README Sun 6 | All review + AI checklist Mon 7; **submit Mon 7 Dec** |
| **Contingency** | Tue 8 – Fri 11 Dec | Not planned. Only for fixes found in review; the official deadline stays Fri 11 Dec | | | |
| **Oral prep** | Tue 8 – Mon 14 Dec | Mock: each member explains a part they did not write | | | |

**What changed to fit 1.5 weeks.**

1. **Shorter sprints, earlier gates.** S0 is one day, S1 and S2 are three days each; G1 moves to Tue 1 Dec and the freeze to Fri 4 Dec.
2. **Rolling write-up.** Each paragraph is written within 24 h of its result becoming final, so S3 only assembles, trims and reviews — no section waits for "writing week".
3. **No duplicated runs.** The reported Q3.2 model (CNN encoder, pretrained, frozen) *is* one cell of the Q3.3 grid: Artur runs it once and Francisco reuses it.
4. **Parallel compute.** The 12 Q3.3 runs are spread over three Colab accounts at once — Artur (CNN-frozen, which is Q3.2), Nuno (MLP frozen + joint), Francisco (CNN-joint) — so the grid takes one or two days instead of four. Part 2 stays on laptop CPUs, so it never competes for a GPU.
5. **Scope cut to the core.** **[if ahead]**: the Q2.5 flatten-head variant, the Q3.3 fifth condition (fine-tuned), the Q3.4 length-penalty sweep and BLEU-vs-length plot, the per-epoch BLEU curve in Q3.2, the Part 1 numerical appendices. The Q3.2 error taxonomy uses 30 captions instead of 50. Nothing the statement requires was cut.
6. **Bonus reduced.** Q4.1 (Transformer from scratch) is dropped — too large for the window. Q4.2 is reduced to an encoder swap, owned by Nuno, decided at G1 and frozen with everything else on Fri 4 Dec.
7. **Daily async check-in at 21:00** (3 lines each in the group chat: merged today, next, blocked by). A blocker on a gate gets fixed the next morning, not at the next sprint.

**Why nobody waits.** There are only four cross-member dependencies, and Sprint 0 turns each into an interface instead of a result:

1. Nuno's `inductive_bias.py` → Francisco's CNN shift/permutation tests (needed S1, available G0).
2. Nuno's MLP checkpoints + `features()` → Artur's first captioner (needed S1, available G0).
3. Francisco's CNN checkpoints → Artur's Q3.2 final runs and Francisco's Q3.3 (needed S2, available G1).
4. Artur's `run_q3_2.py` flags → Francisco's Q3.3 grid (needed S2, available G1).

Part 1 (Q1.1, Q1.2) and Q3.4(b) depend on nothing and fill any waiting time.

---

## 4. Sprint 0 — shared contracts

**Target layout** (skeleton files stay in `src/` because they import each other by module name):

```
src/
├── load_data.py  data.py  train.py  metrics.py   # skeleton (changes → CHANGES.md)
├── experiment.py        # Artur: seeds, mean_std, paired_diff, save_summary, checkpoint paths
├── inductive_bias.py    # Nuno: shift, make_permutation, permute_pixels, unpermute_weights
├── encoders.py          # Francisco: CNN, MLP-encoder wrapper, build_encoder
├── tokenizer.py         # Artur: tokenize, Vocab, encode, decode, special ids
├── captioning.py        # Artur: Captioner, greedy/beam generate, caption_metrics
└── run_qX_Y.py          # one entry point per question
results/qX_Y/            # summary.json, figures (png for README), README.md
checkpoints/             # gitignored; mirrored to Drive "DL-Group5/checkpoints"
report/…Group_5/figures/ # PDF figures used by main.tex
```

**Interfaces to merge at G0** (bodies can be stubs that raise `NotImplementedError`; signatures may not change afterwards without telling the others):

```python
# experiment.py (Artur)
SEEDS = (0, 1, 2)
def mean_std(values) -> dict                    # {"mean", "std"} with unbiased std
def paired_diff(a, b) -> dict                   # per-seed b-a, mean, std, consistent: bool
def save_summary(path, summary: dict) -> None   # adds device, git commit, timestamp
def ckpt_path(name: str, seed: int) -> Path     # checkpoints/<name>_seed<seed>.pt

# inductive_bias.py (Nuno)
def shift(images, dx: int, dy: int, mode="circular") -> Tensor   # "circular" = torch.roll, "fill" = 0 (= train-mean colour after normalisation)
def make_permutation(seed=1234, n=64 * 64) -> Tensor              # fixed, from a seeded torch.Generator
def permute_pixels(images, perm) -> Tensor                        # same permutation on all 3 channels
def unpermute_weights(w, perm) -> Tensor                          # back to original pixel order for plotting

# encoders.py (Francisco)
class CNN(nn.Module): out_dim = 256; def features(self, x) -> (B, 256); def grid(self, x) -> (B, 64, 128); def forward(self, x) -> (B, 10)
def build_encoder(kind: str, ckpt: str | None, freeze: bool) -> nn.Module   # exposes .out_dim, returns (B, D)

# tokenizer.py (Artur)
PAD, BOS, EOS, UNK = 0, 1, 2, 3
def tokenize(caption: str) -> list[str]
class Vocab: build(captions, min_freq), encode(caption) -> list[int], decode(ids) -> list[str], __len__

# captioning.py (Artur)
class Captioner(nn.Module): forward(images, inputs) -> logits (B, T, V)
def generate(model, image_loader, method="greedy", beam_size=1, length_norm=False, max_len=20) -> list[list[str]]
def caption_metrics(hyps, refs, labels) -> dict  # bleu4, class_mention, avg_len, distinct1, distinct2
```

**Git workflow.** Branch per task (`n/q2-4`, `f/q2-5`, `a/q3-2`); small PRs into `main`, approved by the review partner; never commit `data/` or `checkpoints/` (add `checkpoints/` to `.gitignore`); rebase before merging.

---

## 5. Step-by-step plan per question

Each step lists **Owner · Sprint · Depends on · Report slot**, then *Theory*, *Approach and why*, *How*, *Hypotheses*, *Report output* and *Done when*.

### Part 1 — Theory (20 pts)

#### Q1.1 Convolutions as constrained linear layers — Francisco · S1 (draft) / S2 (final) · no dependency · `\paragraph{Q1.1.}` ≈ 0.7 page

*Theory you must master.*
- **Cross-correlation with circular padding.** $y_i=\sum_{j=-r}^{r} w_j\,x_{(i+j)\bmod n}$. Every output is a dot product of the same $k=2r+1$ weights with a window around position $i$ (**locality**), and the same weights are reused at every $i$ (**weight sharing**).
- **Writing it as $y=Wx$.** Read off the coefficient of $x_m$ in $y_i$: it is non-zero only if $m\equiv i+j \pmod n$ for some $\lvert j\rvert\le r$, and then equals $w_j$. So $W_{i,m}$ depends only on $(m-i)\bmod n$ — each row is the previous one rotated right by one. That is exactly the **circulant** definition, and the matrix is also **banded** (only $k$ non-zeros per row, wrapping at the corners).
- **Free parameters.** A circulant $W$ is fixed by its first row, here by $k$ numbers, against $n^2$ for a dense $A$. This parameter tying is the whole "constraint".
- **Shift operator and equivariance.** $(Sx)_i=x_{(i-1)\bmod n}$. Equivariance means $f(Sx)=Sf(x)$: shifting the input shifts the output. Proof strategy: substitute $Sx$ into Eq. (1) and re-index the sum. An elementwise $\sigma$ commutes with any permutation of coordinates, and $S$ is one, so $\sigma(\mathrm{conv}(Sx)) = S\,\sigma(\mathrm{conv}(x))$. Averaging all coordinates is unchanged by permuting them, so global average pooling turns equivariance into **invariance**.
- **"Equivariant iff circulant".** (⇐) is the computation above for a general circulant $A$. (⇒) apply $ASx=SAx$ to basis vectors: $Se_m=e_{m+1}$, so column $m+1$ of $A$ equals column $m$ shifted down by one; chaining this shows $A_{i,m}$ depends only on $(m-i)\bmod n$. Both directions must be written.
- **Q1.1(c) counts.** Dense: $(3\cdot64\cdot64)\times(16\cdot64\cdot64)$ weights $+\,16\cdot64\cdot64$ biases. Conv: $16\times(3\cdot3\cdot3)$ weights $+\,16$ biases. Compute both and their ratio.

*Approach and why.* Write the gains (parameters, sample efficiency, built-in translation equivariance, locality prior suited to images) and the losses (no global interaction in one layer, no use of absolute position, a wrong prior if pixels are not spatially ordered). **End with explicit predictions for Q2.5**: (P1) a CNN with far fewer parameters matches or beats the MLP; (P2) a CNN is robust to shifts, exactly so for circular shifts by multiples of its total stride with circular padding and global pooling; (P3) a pixel permutation hurts the CNN but not the MLP. Francisco then confirms or rejects them in Q2.5 — the report gains a clear theory → experiment story.

**[if ahead]** *Appendix (doesn't count toward the page limit, 20 lines of torch).* Build $W$ for $n=8$ and compare $Wx$ with `F.conv1d(F.pad(x, (r, r), mode="circular"), w)`; check $\mathrm{conv}(\mathrm{roll}(x))=\mathrm{roll}(\mathrm{conv}(x))$ numerically.

*Done when:* all three parts derived, Nuno (reviewer) can redo (b) unaided, predictions P1–P3 written.

#### Q1.2 Gradients through time — Nuno · S1 (draft) / S2 (final) · no dependency · `\paragraph{Q1.2.}` ≈ 0.7 page

*Theory you must master.*
- **One-step Jacobian.** $h_t=\tanh(a_t)$, $a_t=Wh_{t-1}+Ux_t+b$. Chain rule: $\frac{\partial h_t}{\partial h_{t-1}}=\frac{\partial h_t}{\partial a_t}\frac{\partial a_t}{\partial h_{t-1}}$, with $\tanh'(a)=1-\tanh^2(a)$ applied elementwise, i.e. a diagonal matrix. Then $\frac{\partial h_T}{\partial h_k}=\prod_{t=k+1}^{T}\frac{\partial h_t}{\partial h_{t-1}}$, **ordered** with $t=T$ on the left.
- **Spectral norm bound.** $\lVert AB\rVert_2\le\lVert A\rVert_2\lVert B\rVert_2$ (sub-multiplicativity); the norm of a diagonal matrix is its largest absolute entry, and $0<1-h^2\le1$. Multiply $T-k$ factors.
- **Vanishing.** If $\sigma_{\max}(W)<1$ the bound decays geometrically, so contributions of early steps to $\nabla L$ become negligible: the model cannot learn long-range dependencies (Bengio et al., 1994).
- **Exploding (linear symmetric case).** $W=Q\Lambda Q^\top$ ⇒ $W^{T-k}=Q\Lambda^{T-k}Q^\top$, and for a symmetric matrix the spectral norm is $\max_i\lvert\lambda_i\rvert$, so the norm is $\lvert\lambda_{\max}\rvert^{T-k}$, exponential when $\lvert\lambda_{\max}\rvert>1$.
- **Clipping.** $g\leftarrow\tau\,g/\lVert g\rVert$ when $\lVert g\rVert>\tau$ (Pascanu et al., 2013). It bounds the step size, which fixes explosions, but it rescales the *whole* gradient: the long-range terms stay negligible *relative* to short-range ones, so the lost signal is not recovered.
- **LSTM.** With $c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t$ and the gates treated as constants, $\partial c_t/\partial c_{t-1}=\mathrm{diag}(f_t)$, so the product is a product of forget gates — no weight matrix and no squashing derivative. With $f_t\approx1$ the gradient passes almost unchanged (the "constant error carousel"). A positive forget-gate bias (e.g. 1) makes $f_t=\sigma(\cdot)$ close to 1 at initialisation, so the cell **remembers by default** until it learns to forget (Gers et al., 2000; Jozefowicz et al., 2015).

*Approach and why.* Derivations first, then one paragraph connecting them to Part 3: **Artur uses an LSTM with forget bias 1 and gradient clipping at 5 in Q3.2 because of exactly these results.** **[if ahead]** Appendix: an autograd plot of $\lVert\partial h_T/\partial h_k\rVert$ against $T-k$ for $W$ scaled to $\sigma_{\max}=0.9$ and $1.1$ — a direct numerical confirmation.

*Done when:* (a)–(c) derived; Artur (reviewer) agrees the Q3.2 design paragraph cites them correctly.

---

### Part 2 — Image classification (40 pts)

**Shared protocol for every Part 2 model from now on** (so Q2.4/Q2.5 comparisons are fair): Adam, lr 1e-3, batch 128, at most 35 epochs, patience 8 on validation accuracy, best epoch restored, flip off, seeds 0–2, channel statistics from train. If the CNN needs a different lr, choose it on validation and say so.

#### Q2.1–Q2.3 — Nuno · done + follow-ups in S1 · `\paragraph{Q2.1.}` 0.35 p · `\paragraph{Q2.2.}` 0.45 p · `\paragraph{Q2.3.}` 0.8 p

*Theory Nuno must be able to explain in the oral.*
- **Softmax regression.** $p(y=c\mid x)=\mathrm{softmax}(Wx+b)_c$, trained with cross-entropy $-\log p(y\mid x)$; it is an MLP with no hidden layer, so its decision boundaries are linear in pixel space. Each row $w_c$ is a **class template**: the logit is the match between the image and that template.
- **MLP capacity and generalisation.** Hidden layers with ReLU give non-linear boundaries; 6.4 M parameters for 6,500 images means overfitting is the expected failure mode — which the train–validation gap in Q2.3 shows.
- **Regularisers.** Dropout (Srivastava et al., 2014) randomly zeroes hidden units at training time and approximates an ensemble; `Adam(weight_decay=…)` adds an L2 term to the gradient, which interacts with Adam's per-parameter scaling (unlike decoupled AdamW, Loshchilov & Hutter, 2019) — mention it as a known limitation of that condition. Horizontal flip is data augmentation that encodes a symmetry prior (a flipped bus is still a bus).
- **Early stopping and best-epoch restore** select a model on validation, so validation accuracy at the chosen epoch is optimistically biased — which is why test is reported.

*Report output.* Q2.1: data table (splits, class counts, channel statistics) + learning-curve figure + one-line test accuracy. Q2.2: search table (5 candidates, validation accuracy, parameters) + selected configuration + learning curves + test mean ± std + comparison with the linear model. Q2.3: one table with train accuracy, validation accuracy, gap and paired $\Delta$ per condition (including the new underfitting condition); per-class accuracy table; confusion matrix figure; 3–4 sentences on which differences are consistent by our rule, and on over-/underfitting.

*Done when:* the five follow-ups in §1 are merged and the three paragraphs are in `main.tex`.

#### Q2.4 Inductive biases of MLPs — Nuno · S1 · depends on Nuno's checkpoints · `\paragraph{Q2.4.}` ≈ 0.6 page

*Theory.*
- **Weights as templates.** A first-layer unit computes $\phi(w_j^\top x+b_j)$ with $w_j\in\mathbb R^{12288}$: it fires when the image correlates with $w_j$. Reshaped to $3\times64\times64$, $w_j$ shows which pixel pattern drives it. Expect blurry, position-specific colour layouts (sky-blue tops for airplane, green or brown lower halves for giraffe and elephant) rather than edges, because nothing ties a weight to its neighbours.
- **No weight sharing ⇒ no translation equivariance.** A pattern learned at position $p$ has no counterpart at $p+\delta$; shifting the image moves the evidence onto weights that never saw it. Prediction: accuracy falls as the shift grows.
- **Why a pixel permutation does not matter to an MLP.** Let $P$ be the fixed permutation matrix. Then $W_1(Px)=(W_1P)\,x$, and $\{W_1P\}=\{W_1\}$ as $W_1$ ranges over all matrices, so the hypothesis class is unchanged. PyTorch initialises entries i.i.d., so the initial distribution is invariant to permuting columns, and SGD/Adam act coordinate-wise, so the whole training run on permuted data is a relabelled copy of a run on the original data. **Expected test accuracy is identical**; per-seed numbers differ only because the same seed now produces a differently aligned initialisation. The MLP simply never assumed pixels had neighbours.

*Approach and why.*
- **Two shift modes:** `circular` (`torch.roll`), matching the circular padding of Q1.1 so the CNN comparison in Q2.5 is clean, and `fill` (vacated pixels set to 0, i.e. the training-mean colour after normalisation), closer to a real camera shift. Shifts $\{0,1,2,3,4,6,8,12,16\}$, horizontal and vertical.
- **Un-permuting the weights** of the retrained permuted MLP turns the invariance argument into a picture: they should look like ordinary templates. This is our creative addition — cheap and very convincing.
- **Flip stays off** (it is off in the best configuration). Flipping a permuted image is not the permutation of a flipped image, so flips would break the equivalence argument.

*How.*
1. Retrain or load the Q2.1 linear model and plot its 10 class templates; plot 16 first-layer MLP units (min–max normalise each to [0, 1]; `imshow` gets CPU tensors permuted to H×W×C).
2. `run_q2_4.py --part shift`: evaluate the 3 MLP checkpoints on shifted test images; plot accuracy against shift (mean ± std band; one line per direction and mode).
3. `run_q2_4.py --part permute`: `perm = make_permutation(1234)`; permute all splits; retrain the best MLP for seeds 0–2 under the shared protocol; compare with the unpermuted MLP by paired differences; plot `unpermute_weights` for the same 16 units.

*Hypotheses.*
- **H2.4a** Test accuracy decreases monotonically with |shift| in both modes; the first few pixels already cost a consistent drop.
- **H2.4b** Permuted and unpermuted MLP accuracies are within seed noise (paired rule).
- **H2.4c** Un-permuted weights of the permuted model show the same kind of colour-layout templates as the original.

*Report output.* One figure with two panels (templates and weights; un-permuted weights) or move one to the appendix; one accuracy-vs-shift figure (the CNN curve is added later by Francisco — share the plotting function); one sentence per hypothesis. The code stays parametrised by model, because Francisco reruns it.

*Done when:* `inductive_bias.py` merged at G0, figures and JSON in `results/q2_4/`, paragraph written.

#### Q2.5 From MLPs to CNNs — Francisco · S1 · depends on `inductive_bias.py` (G0) · `\paragraph{Q2.5.}` ≈ 0.7 page

*Theory.*
- **Conv layer parameters:** $C_{\text{out}}(C_{\text{in}}k^2)+C_{\text{out}}$, independent of image size. **Receptive field** grows with depth and pooling, so later layers see whole objects while each weight stays local.
- **Equivariance and where it breaks** (from Q1.1): conv + BatchNorm (per-channel affine in eval mode) + ReLU are equivariant; **zero padding** breaks it at the borders; **stride and pooling** make the network equivariant only to shifts that are multiples of the total stride — other shifts change which pixels fall in each pooling window (aliasing; Zhang, 2019; Azulay & Weiss, 2019). A **global-average-pooling (GAP) head** gives invariance; a **flatten + linear head** keeps position information and loses it.
- **Why a permutation hurts a CNN.** After permuting, neighbouring pixels are unrelated, so the locality prior is wrong; and $W_1P$ is not a convolution, so the CNN cannot undo the permutation the way the MLP can. Its hypothesis class is **not** closed under permutations.

*Approach and why.*
- **Architecture:** four stages of `Conv3×3 → BatchNorm → ReLU → MaxPool2`, channels 32 → 64 → 128 → 256 (64 → 32 → 16 → 8 → 4 spatially, total stride 16), then GAP → `Linear(256, 10)`. That is about 0.39 M parameters (896 + 18,496 + 73,856 + 295,168 conv, 960 BatchNorm, 2,570 head) — **16× fewer** than the MLP, which satisfies "comparable or smaller" with room and makes P1 a strong test. Add dropout before the head only if validation shows overfitting.
- **Controlled variants (our creative axis, each confirms a theory claim):** core — padding `circular` (main model) vs `zeros`, 3 seeds each; **[if ahead]** — head `GAP` vs `flatten`.
- **Same protocol** as the MLP (shared protocol above); same seeds; same shift and permutation code.
- `features()` returns the 256-d GAP output and `grid()` the stage-3 map (8×8 positions × 128 channels) — used later by Part 3 (and by Q4.1 if it is ever revived). Save checkpoints `cnn_seed{s}.pt` to Drive by **G1**.

*How.* `run_q2_5.py --padding circular --head gap` (+ variants) → `results/q2_5/`; reuse `shift`, `permute_pixels`; plot the CNN curves on Nuno's accuracy-vs-shift figure; build the comparison table.

*Calculations.* Parameter counts printed by `count_parameters` and checked by hand with the formula above (put the check in the paragraph). For the exactness test, compare predicted classes on the original and 16-pixel-shifted test sets: with circular padding + GAP they must be **identical on all 1,000 images** (up to floating point).

*Hypotheses.*
- **H2.5a (P1)** CNN test accuracy > MLP test accuracy (paired rule) with 16× fewer parameters.
- **H2.5b (P2)** With circular padding + GAP, circular shifts by 16 and 32 leave every prediction unchanged; shifts of 1–8 px change accuracy far less than for the MLP; with zero padding (or a flatten head, if run) the invariance is visibly weaker.
- **H2.5c (P3)** Permuting pixels lowers CNN accuracy consistently, while the MLP is unaffected (H2.4b), so the CNN's advantage shrinks or reverses.

*Report output.* Main table: linear / MLP / CNN — parameters, validation and test accuracy (mean ± std), plus a row each for permuted MLP and permuted CNN (this is the template's Table 1 extended). The shared shift figure. Three or four sentences explaining H2.5a–c through Q1.1, closing the P1–P3 loop.

*Done when:* G1 checkpoints exist, table + figure in `results/q2_5/`, paragraph written.

---

### Part 3 — Image captioning (40 pts)

**Captioning protocol** (fixed by Artur at G1, used by Francisco in Q3.3): LSTM decoder (embedding 256, hidden 512, 1 layer, dropout 0.3), image → initial $(h_0,c_0)$, Adam lr 1e-3, batch 64, gradient clipping 5, at most 20 epochs, patience 3 on token-weighted validation NLL, greedy decoding with `max_len=20`, seeds 0–2 (captioner seed $s$ uses classifier checkpoint seed $s$, so pretraining variance is included).

#### Q3.1 Text preprocessing — Artur · S1 · no dependency · `\paragraph{Q3.1.}` ≈ 0.35 page

*Theory.*
- **Tokenisation and vocabulary.** Word frequencies are Zipfian: a few thousand types cover almost all tokens, and the long tail is mostly typos and rare names. Mapping words with count below a threshold to `<unk>` keeps the softmax small and gives `<unk>` enough training examples. The vocabulary comes from **training captions only** — validation words absent from training must become `<unk>`, otherwise we leak information.
- **Special tokens.** `<bos>` starts generation, `<eos>` is a predicted token that lets the model decide the length, `<pad>` fills batches.
- **Padding and masking.** `PadCollate` right-pads each batch to its longest caption. Pad positions are not data: counting them in the loss would reward predicting `<pad>` and make perplexity look better than it is. With input `y[:, :-1]` and target `y[:, 1:]`, `cross_entropy(…, ignore_index=PAD)` removes them exactly; `<eos>` stays a target, `<bos>` never is one.

*Approach and why.* Tokeniser exactly as the statement suggests (lowercase, separate punctuation, split on whitespace — string methods only), **keeping punctuation as tokens** because the references contain it and BLEU must see the same tokenisation on both sides. Choose the threshold from a **coverage curve** (vocabulary size and validation `<unk>` rate for thresholds 1–10): pick where the `<unk>` rate stops improving much; 5 is the usual starting point. Report all three requested statistics.

*How.* `tokenizer.py` + `run_q3_1.py` → `results/q3_1/` (`summary.json`, coverage curve, caption-length histogram, the saved vocabulary as JSON so every later run uses the same ids).

*Calculations.* `<unk>` fraction = (# `<unk>` among validation word tokens) / (# validation word tokens), special tokens excluded — state this definition.

*Hypothesis.* **H3.1** A threshold around 5 removes most of the vocabulary while keeping the validation `<unk>` rate low; captions are short enough that `max_len=20` covers almost all of them (check the 95th/99th percentiles).

*Report output.* Vocabulary size, `<unk>` rate, length histogram (small figure or numbers), and two sentences on batching and masking.

#### Q3.2 An RNN captioner — Artur · S1 (pipeline) / S2 (final) · depends on Nuno's MLP checkpoint (G0), Francisco's CNN (G1) · `\paragraph{Q3.2.}` ≈ 1.0 page

*Theory.*
- **Conditional language model.** $p(y\mid x)=\prod_{t=1}^{L}p(y_t\mid y_{<t},x)$, trained by maximum likelihood, i.e. token-level cross-entropy summed over the caption.
- **Teacher forcing.** At training time the decoder is fed the **ground-truth** previous word; at test time it is fed its **own** prediction. The mismatch (exposure bias; Bengio et al., 2015) explains why early mistakes snowball into wrong captions.
- **Perplexity.** $\mathrm{PPL}=\exp\big(\frac{1}{N}\sum_{\text{non-pad targets}}-\log p(y_t\mid y_{<t},x)\big)$ — the effective number of equally likely next words. It must be **token-weighted**: sum the NLL with `reduction="sum"` and divide by the number of non-pad targets, not average per-batch means.
- **LSTM decoder** (Hochreiter & Schmidhuber, 1997): gates $i_t,f_t,o_t$ and cell $c_t$ (equations in Q1.2). Chosen over a GRU because of the additive cell path derived by Nuno in Q1.2(c); forget-gate bias set to 1 (`bias_ih_l0[H:2H] = 1`, PyTorch gate order i, f, g, o).
- **Image conditioning.** Vinyals et al. (2015) feed the image once, at the start, and report that feeding it at every step overfits more easily. We follow that: $h_0=\tanh(W_h v)$, $c_0=\tanh(W_c v)$ where $v$ is the encoder output.
- **BLEU-4** (Papineni et al., 2002): clipped n-gram precisions $p_1..p_4$ over the corpus, geometric mean, times the brevity penalty $\mathrm{BP}=\min(1,e^{1-r/c})$. It rewards n-gram overlap with any of the 5 references and penalises captions shorter than the references.

*Approach and why.*
- **Reported Q3.2 model:** CNN encoder pretrained on classification (Francisco's checkpoint) and **frozen**. This is the cheapest, most stable baseline, and it becomes one cell of Francisco's Q3.3 grid, so no run is wasted. During S1, develop with Nuno's MLP encoder.
- **Early stopping on validation NLL** (cheap and smooth); BLEU is computed once after training, on validation and test (a per-epoch BLEU curve is **[if ahead]**).
- **Frozen encoder pitfalls:** pass only decoder parameters to the optimizer, set `requires_grad=False`, and call `model.encoder.eval()` inside `loss_fn` (the skeleton calls `model.train()` on everything, which would update BatchNorm statistics). Report total and trainable parameters, since `count_parameters` counts only the trainable ones.
- **Qualitative examples chosen by a rule, not by hand:** the 4 test images with the highest and the 4 with the lowest sentence-level BLEU against their references (a few lines on top of `bleu4` with one hypothesis), plus a counted error taxonomy on 30 random test captions annotated by Artur and checked by Francisco: wrong class, generic caption, hallucinated object, wrong count, repetition, ungrammatical. Also count unique captions and those copied verbatim from training — a direct measure of how generic the model is.

*How.* `captioning.py` (`Captioner`, `generate`, `caption_metrics`) + `run_q3_2.py --encoder {mlp,cnn} --pretrained --freeze/--joint --seed s [--subset N]` → `results/q3_2/`. The CLI flags are the G1 contract with F.

*Hypotheses.*
- **H3.2a** Validation perplexity falls quickly and then rises while training perplexity keeps falling (overfitting on 6,500 images); the best epoch is early.
- **H3.2b** Greedy captions are fluent but generic: far fewer unique captions than test images, frequent templates ("a … parked on the side of a street").
- **H3.2c** The dominant error is a wrong or generic object word, concentrated on classes the encoder confuses (link to the Q2.3/Q2.5 confusion matrices).

*Report output.* Model description (5 lines, cite Vinyals and Q1.2); train/validation loss and perplexity figure; BLEU-4 validation/test mean ± std; a figure with the 8 test images, generated caption and one reference; the error-taxonomy counts in two sentences.

*Done when:* the 3-seed baseline is in `results/q3_2/summary.json` and the CLI contract works for F.

#### Q3.3(a) Encoder comparison — Francisco · S2 · depends on G1 · `\paragraph{Q3.3.}` (a)+(b) ≈ 0.9 page

*Theory.*
- **Transfer learning.** A frozen classifier encoder provides features trained to separate 10 classes; they keep what the label needs and discard what captions also mention (counts, background, actions). Joint training adapts the features to the language objective but, with 6,500 images, may overfit or learn weaker visual features than label supervision gives.
- **MLP vs CNN features.** MLP features are position-specific templates (Q2.4); CNN features are translation-robust and more parameter-efficient (Q2.5). Better features should help most on the object word.
- **"Does a better classifier make a better captioner?"** Classification accuracy measures one bit of the caption (the class); BLEU scores ~11 words of n-gram overlap. The relationship is expected but need not be monotone.

*Approach and why.* A **2 × 2 grid** — encoder {MLP, CNN} × {pretrained + frozen, trained jointly from scratch} — with identical decoder, protocol and greedy decoding, 3 seeds each — 12 runs: the 3 CNN-frozen runs are Artur's Q3.2 runs, Nuno runs the 6 MLP-encoder runs on his Colab account (Francisco hands him the exact commands), and Francisco runs the 3 CNN-joint runs. **[if ahead]** Our creative addition is a **fifth condition, CNN pretrained + fine-tuned** (encoder lr 1e-4, decoder 1e-3). It separates the effect of *pretraining* (frozen vs from-scratch differ in both) from the effect of *freezing* (frozen vs fine-tuned differ only in that). State in the report that "jointly trained" means "from scratch" in the 2 × 2 grid.

*Calculations.*
- Parameters per model: total and trainable (MLP encoder ≈ 6.4 M dominates; CNN ≈ 0.39 M).
- **Linear probe** for encoders trained without labels (joint/from scratch): freeze the trained encoder, fit a `Linear(D, 10)` on training features (a few epochs, validation-selected), report test accuracy. Every encoder then has a "classification accuracy", which Q3.3(b) needs.

*Hypotheses.*
- **H3.3a** CNN encoders beat MLP encoders on BLEU-4 and on class mention accuracy.
- **H3.3b** On this small dataset, pretrained + frozen beats joint from scratch for both encoders; fine-tuning (5th condition) beats frozen by a smaller margin.
- **H3.3c** Across conditions, encoder classification accuracy ranks the models the same way as class mention accuracy, but less clearly as BLEU-4.

*Report output.* One table, rows = the 4 grid conditions (+ fine-tuned if run): total/trainable parameters, encoder classification accuracy (classifier or probe), validation perplexity, BLEU-4 validation/test, class mention accuracy (test), all mean ± std. Two or three sentences answering the question with paired differences.

#### Q3.3(b) Class mention accuracy — Francisco · S2 · same runs · shares `\paragraph{Q3.3.}`

*Theory.* `class_mention_accuracy` checks whether any generated token names the image's class (or a synonym, plural added). It measures **semantic grounding of the main object** and nothing else. BLEU measures surface n-gram overlap (fluency and content, but the class word is ~1 of ~11 tokens and generic n-grams like "a", "of the" score well). Classification accuracy measures visual discriminability of the label without any language.

*Approach and why.*
- **Human ceiling:** score each of the 5 reference captions of every test image as if it were a hypothesis and average — the statement says ~94 % on training captions. This puts our numbers on a meaningful scale.
- **Conditional analysis** (frozen encoders, which still have a classifier head): mention accuracy when the classifier is right vs wrong. If captions mention the right class far more often when the classifier is right, the decoder relies on the same evidence as the classifier.
- **Synonyms:** extend `CLASS_SYNONYMS` only from evidence — list the most frequent nouns in reference captions that the current list misses (e.g. compounds) — and report the change in `CHANGES.md`.
- **Scatter plot:** BLEU-4 against class mention accuracy, one point per (condition, seed).

*Hypotheses.* **H3.3d** Mention accuracy is well below the human ceiling and tracks encoder classification accuracy; **H3.3e** some conditions with similar BLEU differ clearly in mention accuracy — showing BLEU's blindness to the key noun.

*Report output.* Mention accuracy column in the Q3.3 table, ceiling in the text, the conditional numbers in one sentence, and 2–3 sentences on what each metric captures that the others miss.

#### Q3.4(a) Beam search — Artur · S2 · depends on the Q3.2 checkpoints · `\paragraph{Q3.4.}` (a)+(b) ≈ 0.75 page

*Theory.*
- Exact $\arg\max_y p(y\mid x)$ is intractable ($V^L$ sequences). **Greedy** keeps one prefix; **beam search** keeps the $B$ best prefixes by cumulative log-probability, expands each by every word ($B\times V$ candidates), keeps the top $B$, moves a hypothesis to the finished set when it emits `<eos>`, and stops when $B$ hypotheses are finished or `max_len` is reached.
- **Scoring.** Raw: $\log p(y\mid x)=\sum_t\log p(y_t\mid y_{<t},x)$ — a sum of negative terms, so longer captions always score lower (proved by Nuno in Q3.4(b)). Length-normalised: $\frac{1}{L}\log p(y\mid x)$ (Wu et al., 2016 divide by a tunable power of a smoothed length instead); $L$ counts tokens including `<eos>` — state it.
- **Search error vs model error.** A bigger beam finds higher-probability captions; if BLEU falls anyway, the model's probabilities are miscalibrated (it prefers short, generic captions), not the search.

*Approach and why.* Implement per image (1,000 test images × beam 5 is cheap on GPU), batched over the beam. **Unit test:** beam size 1 must reproduce greedy output exactly. Evaluate 6 settings — beam {1, 3, 5} × {raw, normalised} — on the 3 Q3.2 seeds. **[if ahead]** Creative extras: a sweep $\alpha\in\{0,0.5,0.7,1\}$, and a plot of BLEU-4 against average length with the mean reference length marked.

*Hypotheses.*
- **H3.4a** With raw scoring, larger beams give shorter captions and fewer distinct words and bigrams.
- **H3.4b** Length normalisation gives longer, slightly more diverse captions; BLEU improves while captions are shorter than the references (brevity penalty) and stops improving once they are not.
- **H3.4c** The best beam setting beats greedy on BLEU-4 by a consistent but small margin.

*Report output.* One table: 6 rows × BLEU-4, average length, distinct-1, distinct-2 (mean ± std over the 3 models); optional BLEU-vs-length figure; three sentences on the trade-offs. Q3.4(b) follows in the same paragraph.

#### Q3.4(b) Why exact decoding favours short captions — Nuno · S2 · no dependency · inside `\paragraph{Q3.4.}`

*Theory and approach.*
- **(i) Counterexample with vocabulary {a, b, `<eos>`}.** Construction principle: make the first-step winner lead to a **flat** next-step distribution and the runner-up lead to a **confident** `<eos>`. Greedy commits to the winner and can only reach a low total; the runner-up path ends with a higher product. Write the two-level probability table, compute every complete caption's probability, and show greedy's output is not the argmax.
- **(ii) Prefix monotonicity.** $p(y_{1:t+1}\mid x)=p(y_{1:t}\mid x)\,p(y_{t+1}\mid y_{1:t},x)\le p(y_{1:t}\mid x)$ because the last factor is at most 1. Every extra word can only lower the score, so exact MAP prefers short captions whenever a short one has a reasonably likely `<eos>`. Length normalisation compares the **average** per-token log-probability, removing the systematic penalty per word.
- Close by pointing at Artur's Q3.4(a) numbers (H3.4a) as the empirical confirmation.

---

### Part 4 — Bonus (up to +20; in this window only Q4.2 as an encoder swap, decided at G1 on Tue 1 Dec)

Go criteria at G1: Q2.4 and Q2.5 final, Q3.2 running end-to-end, Part 1 drafted, and Nuno on schedule. Otherwise skip Part 4 and delete its section from `main.tex` (the limit then stays 8 pages). With only Q4.2, it adds about one page.

**Q4.1 (Transformer captioner) is dropped from the 1.5-week plan.** Writing the decoder blocks by hand, tuning warmup (which needs a per-step scheduler in `train.py`) and producing attention maps is roughly a member-week of work, and its lecture is only on Mon 30 Nov. Keep `CNN.grid()` in the encoder contract anyway: it costs nothing and keeps the option open if the group is unexpectedly far ahead after the freeze.

#### Q4.2 Pretrained encoder — Nuno · S2 (Wed 2 – Fri 4 Dec) · reuses Artur's decoder and Francisco's evaluation table · `\paragraph{Q4.2.}` ≈ 1.0 page

*Theory.* Large-scale pretraining (a ResNet trained on ImageNet's 1.28 M labelled images, or CLIP's contrastive image–text training) learns features that already encode objects, scenes and some relations — far more than our 6,500 images can teach a network from scratch. This is transfer learning at its extreme: in Q3.3 the encoder was pretrained on our 10 labels; here it was pretrained on orders of magnitude more data and classes.

*Approach and why.* **Swap only the encoder:** a pretrained ResNet-50 from `torchvision.models` (allowed in Q4.2), frozen, inputs upsampled from 64×64 to 224×224 with `torchvision.transforms.Resize`, ImageNet normalisation. Precompute the 2,048-d features once for all 8,500 images (a few minutes on Colab) and train **Artur's LSTM decoder unchanged** on them with the captioning protocol, 3 seeds. Because nothing else changes, the gain is attributable to the encoder alone — the cleanest possible answer to "where do the improvements come from". A GPT-2 decoder (ClipCap, Mokady et al., 2021) is out of scope for this window. Record compute: encoder parameters, feature-extraction time, decoder training time.

*Hypotheses.* **H4.2a** The pretrained encoder raises BLEU-4 and, even more, class mention accuracy above our best from-scratch encoder (consistent by the paired rule); **H4.2b** the gain costs about two orders of magnitude more encoder parameters (≈ 25 M against ≈ 0.39 M) but little extra training time, since the features are precomputed. Note that upsampling 64×64 images limits what the pretrained network can see — mention it as a caveat.

*Report output.* Rows added to the Q3.3 table format: pretrained ResNet-50 frozen vs our best encoder — BLEU-4, class mention accuracy, encoder parameters, feature-extraction and training time; three sentences on where the gain comes from and what it costs.

---

## 6. Report assembly (Francisco assembles; every owner writes their own paragraphs)

**Page budget** (8 pages from title to the last answer; references and appendices excluded):

| Slot | Pages | Writer | Main exhibits |
|---|---:|---|---|
| Title + Contributions | 0.4 | Francisco | — |
| Q1.1 | 0.7 | Francisco | derivations |
| Q1.2 | 0.7 | Nuno | derivations |
| Q2.1 | 0.35 | Nuno | data table, curves |
| Q2.2 | 0.45 | Nuno | search table, curves |
| Q2.3 | 0.8 | Nuno | ablation table, per-class table, confusion matrix |
| Q2.4 | 0.6 | Nuno | weights figure, shift figure (shared) |
| Q2.5 | 0.7 | Francisco | linear/MLP/CNN table |
| Q3.1 | 0.35 | Artur | stats, length histogram |
| Q3.2 | 1.0 | Artur | curves, 8 examples, BLEU |
| Q3.3 | 0.9 | Francisco | 4-condition table (+ fine-tuned if run), scatter |
| Q3.4 | 0.75 | Artur + Nuno | 6-setting table, derivation |
| Slack | 0.3 | | |
| **Bonus Q4.2 (if go)** | +1.0 | Nuno | encoder-swap table |

**Paragraph shape** (every `\paragraph{Qx.y.}`): one-sentence answer first → setup in 1–2 sentences (model, protocol, seeds) → table or figure → 2–4 sentences of interpretation, each tied to a numbered hypothesis and to the Part 1 result it confirms. Results the template's example table already anticipates (logistic regression and MLP rows) are extended with the CNN and permuted rows in Q2.5.

**LaTeX conventions.** `booktabs` tables with `$m \pm s$` cells; figures as PDF in `figures/`, width `0.8\linewidth` or two panels side by side; labels `tab:q2-5`, `fig:q2-4-shift`; cite with `\citet{…}`. Add to `references.bib` (only `vinyals2015show` is there now): Papineni 2002, Hochreiter 1997, Bengio 1994, Pascanu 2013, Gers 2000, Srivastava 2014, Zhang 2019, Bengio 2015 (scheduled sampling), Wu 2016, and He 2016 (ResNet) if Q4.2 is done. Move overflow (extra figures, numerical checks) to an appendix rather than shrinking fonts or margins — the template forbids that.

**Contributions paragraph** (draft, fill in at the end):
> Nuno implemented the linear, MLP and inductive-bias experiments (Q2.1–Q2.4), derived Q1.2 and Q3.4(b), and wrote those answers. Francisco derived Q1.1, designed and evaluated the CNN (Q2.5), ran the encoder study (Q3.3) and assembled the report. Artur built the shared infrastructure, the tokenizer, the RNN captioner and the decoding algorithms (Q3.1, Q3.2, Q3.4(a)), and ensured reproducibility. All members reviewed each other's code and derivations.

**AI checklist.** Fill in together on Mon 7 Dec from everyone's notes; include this guide and any code help.

---

## 7. Definition of done and submission (Artur leads, all check)

- [ ] Every table and figure regenerates from `src/run_*.py` → `results/*/summary.json`; seeds and device are in each JSON.
- [ ] `grep -rnE "numpy|sklearn|torchvision.models|timm|transformers" src/` finds nothing outside Part 4 scripts.
- [ ] `CHANGES.md` lists every skeleton change; the report mentions them.
- [ ] A fresh clone + `pip install -r src/requirements.txt` + the README commands reproduces one result per part (Artur, Sun 6 Dec).
- [ ] README: environment, one command per table/figure, expected runtimes (CPU/GPU), where checkpoints go.
- [ ] Report: ≤ 8 pages (≤ 10 with Part 4), question order, contributions paragraph, completed checklist — each missing item costs 10 points.
- [ ] Zip = report PDF + code, **no** `data/`, **no** `checkpoints/`; uploaded to Fénix under the group name on **Mon 7 Dec** (official deadline Fri 11 Dec).
- [ ] Oral mock (8–14 Dec): each member explains one part owned by someone else — derivation, code path and experiment.
