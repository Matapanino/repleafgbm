# Related-work bibliography additions (verified 2026-07-02)

## Question
Verify exact bibliographic data for six candidate Related Work citations for
`docs/paper/repleafgbm-algorithm.tex` / `docs/paper/references.bib`, in the house
bibtex style (verified 2026-06-26 against DBLP/publisher/arXiv), and map each to a
Related Work paragraph (GBDTs; model trees/PWL leaves; tabular DL & numerical
embeddings; hybrids) and to what the paper should *say* about it relative to
RepLeafGBM's thesis (raw-feature routing + representation-conditioned leaf, frozen
encoder, honest mid-pack Grinsztajn result).

## Existing coverage (checked first, to avoid duplicates)
`references.bib` already cites, and no proposed addition duplicates them: TabNet
(`arik2021tabnet`), NODE (`popov2020node`), FT-Transformer
(`gorishniy2021revisiting`), numerical-embedding PLR/periodic
(`gorishniy2022embeddings`), DeepGBM (`ke2019deepgbm`), Deep Neural Decision
Forests (`kontschieder2015deep`), Grinsztajn et al. (`grinsztajn2022why`), and the
LightGBM/CatBoost/XGBoost trio. None of TabPFN, McElfresh 2023, TabArena, NGBoost,
Lou et al. GA2M/EBM, or a linear-tree citation appeared anywhere in the `.tex`, so
all six were real gaps, not re-citations.

## 1. TabPFN — cite the **v2 Nature 2025** paper
- Hollmann, N., Müller, S., Purucker, L., Krishnakumar, A., Körfer, M., Hoo, S. B.,
  Schirrmeister, R. T., Hutter, F. "Accurate Predictions on Small Data with a
  Tabular Foundation Model." *Nature*, 637(8045):319–326, 2025
  (DOI 10.1038/s41586-024-08328-6; volume/issue/pages confirmed).
- v2 is the right citation now: the in-context prior-fitted transformer handling
  both classification and regression, the one every 2025–26 tabular benchmark
  (including TabArena) compares against; the ICLR 2023 classification-only TabPFN
  is superseded for citation purposes.
- **Maps to:** tabular-DL paragraph, as a second axis — not "embed-then-MLP" but
  "in-context, no gradient training at all". Framed as context for the honest
  mid-pack result: TabPFN wins on small-n by amortizing training into a pretrained
  prior, without explicit routing.

## 2. McElfresh et al. 2023 — NN vs. GBDT meta-study
- McElfresh, D. C., Khandagale, S., Valverde, J., Prasad C., V., Ramakrishnan, G.,
  Goldblum, M., White, C. "When Do Neural Nets Outperform Boosted Trees on Tabular
  Data?" *NeurIPS 2023, Datasets and Benchmarks Track* (176 datasets, 19
  algorithms). Published 7-author camera-ready list used (the arXiv revision adds
  two authors — cite the venue form, cf. the `wang1997m5prime` note).
- **Maps to:** tabular-DL paragraph next to `grinsztajn2022why`: the NN-vs-GBDT gap
  is often within noise/HPO-budget effects — the justification for the paper's own
  fair-budget, significance-tested comparison bar.

## 3. TabArena — living tabular benchmark
- Erickson, N., Purucker, L., Tschalzev, A., Holzmüller, D., Mutalik Desai, P.,
  Salinas, D., Hutter, F. "TabArena: A Living Benchmark for Machine Learning on
  Tabular Data." *NeurIPS 2025, Datasets and Benchmarks Track* (spotlight);
  confirmed from the proceedings PDF title page (also arXiv:2506.16791).
- **Maps to:** tabular-DL paragraph / leaderboard framing: GBDTs remain strong
  defaults; DL catches up only with heavy tuning/ensembling; foundation models
  excel on small data — the 2025 update of the Grinsztajn thesis, independently
  corroborating why mid-pack-among-tuned-baselines is meaningful.

## 4. NGBoost — probabilistic/natural-gradient boosting
- Duan, T., Avati, A., Ding, D. Y., Thai, K. K., Basu, S., Ng, A. Y., Schuler, A.
  "NGBoost: Natural Gradient Boosting for Probabilistic Prediction." *ICML 2020*,
  PMLR vol. 119, pp. 2690–2700 (confirmed via PMLR/DBLP).
- **Maps to:** GBDTs paragraph as a *leaf-target* variant vs RepLeafGBM's
  *leaf-basis* variant: NGBoost changes *what* the leaf predicts (distribution
  parameters), RepLeafGBM changes what the leaf predicts *on*; compatible in
  principle, not combined in v0 (future-work pointer).

## 5. EBM / GA2M — cite **both** Lou et al. KDD'12 and KDD'13
- Lou, Caruana, Gehrke. "Intelligible Models for Classification and Regression."
  *KDD 2012*, pp. 150–158 (DOI 10.1145/2339530.2339556).
- Lou, Caruana, Gehrke, Hooker. "Accurate Intelligible Models with Pairwise
  Interactions." *KDD 2013*, pp. 623–631 (DOI 10.1145/2487575.2487579) — the GA2M
  paper InterpretML's EBM implements.
- Citing both is standard EBM practice; skip the InterpretML software paper.
- **Maps to:** model-trees/PWL-leaves paragraph as the sibling design point:
  boosting with a non-constant leaf where routing and leaf basis are the *same*
  raw features (per-feature/pair shape functions); RepLeafGBM instead conditions
  the leaf on a learned, frozen, decoupled representation.

## 6. LightGBM `linear_tree` — already covered, no new entry needed
- LightGBM's docs cite Shi/Li/Li ("Gradient Boosting with Piece-Wise Linear
  Regression Trees") for `linear_tree`; **`shi2019gbdtpl` is already in
  `references.bib` and already cited** in the model-trees paragraph. Action taken:
  one clause "(implemented as LightGBM's `linear_tree` option)" on the existing
  citation; no new bib entry.

## Sanity pass
- TabNet/NODE/FT-Transformer/DeepGBM/DNDF/Grinsztajn already cited — no duplicates.
- Flagged for a future verification pass (not added): **GRANDE** (Marton et al.,
  ICLR 2024, arXiv:2309.17130) — differentiable raw-feature routing via a
  straight-through estimator; would freshen the Hybrids paragraph's recency.

## Guardrail check
All six are citation-only additions describing *other* methods (no code changes;
routing/freezing invariants untouched). Framing guardrail respected in the edits:
TabPFN/TabArena/McElfresh are presented as *context for the mid-pack claim*, not as
competitors RepLeafGBM claims to beat.
