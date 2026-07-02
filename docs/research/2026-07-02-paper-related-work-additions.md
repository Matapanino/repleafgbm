# Related-work bibliography additions (verified 2026-07-02)

## Question
Verify exact bibliographic data for six candidate Related Work citations for
`docs/paper/repleafgbm-algorithm.tex` / `docs/paper/references.bib`, in the house
bibtex style (verified 2026-06-26 against DBLP/publisher/arXiv), and map each to a
Related Work paragraph (`\S\ref{sec:related}`: GBDTs; model trees/PWL leaves;
tabular DL & numerical embeddings; hybrids) and to what the paper should *say*
about it relative to RepLeafGBM's thesis (raw-feature routing + representation-
conditioned leaf, frozen encoder, honest mid-pack Grinsztajn result).

## Existing coverage (checked first, to avoid duplicates)
`references.bib` already cites, and no proposed addition duplicates them: TabNet
(`arik2021tabnet`), NODE (`popov2020node`), FT-Transformer
(`gorishniy2021revisiting`), numerical-embedding PLR/periodic
(`gorishniy2022embeddings`), DeepGBM (`ke2019deepgbm`), Deep Neural Decision
Forests (`kontschieder2015deep`), Grinsztajn et al. (`grinsztajn2022why`), and the
LightGBM/CatBoost/XGBoost trio. None of TabPFN, McElfresh 2023, TabArena, NGBoost,
Lou et al. GA2M/EBM, or a linear-tree citation appear anywhere in the `.tex` (grepped
for `TabPFN|NGBoost|EBM|GA2M|McElfresh|TabArena` — zero hits), so all six are real
gaps, not re-citations.

## 1. TabPFN — cite the **v2 Nature 2025** paper
- Hollmann, N., Müller, S., Purucker, L., Krishnakumar, A., Körfer, M., Hoo, S. B.,
  Schirrmeister, R. T., Hutter, F. "Accurate Predictions on Small Data with a
  Tabular Foundation Model." *Nature*, 637(8045):319–326, 2025.
  [nature.com/articles/s41586-024-08328-6](https://www.nature.com/articles/s41586-024-08328-6)
  (confirmed volume/issue/pages/DOI via [ideas.repec.org mirror](https://ideas.repec.org/a/nat/nature/v637y2025i8045d10.1038_s41586-024-08328-6.html)).
- This is the right citation now: v2 is the in-context, prior-fitted transformer
  that handles both classification and regression and is the one every 2025–26
  tabular benchmark (including TabArena) compares against; the original ICLR 2023
  TabPFN (classification-only) is superseded for citation purposes.
- **Maps to:** Tabular deep learning paragraph (alongside TabNet/NODE/FT-Transformer),
  as a second axis — not "embed-then-MLP" but "in-context, no gradient training at
  all." Frame it as the sharpest counterpoint to RepLeafGBM's honest mid-pack
  result: TabPFN wins on small-$n$ tabular data by amortizing training into a
  pretrained prior, at the cost of (a) no explicit routing/interpretability and (b)
  weak scaling to larger $n$/high-cardinality categoricals — where GBDT-family
  routing (RepLeafGBM included) keeps its niche.

```bibtex
@article{hollmann2025tabpfn,
  author  = {Hollmann, Noah and M{\"u}ller, Samuel and Purucker, Lennart and
             Krishnakumar, Arjun and K{\"o}rfer, Max and Hoo, Shi Bin and
             Schirrmeister, Robin Tibor and Hutter, Frank},
  title   = {Accurate Predictions on Small Data with a Tabular Foundation
             Model},
  journal = {Nature},
  volume  = {637},
  number  = {8045},
  pages   = {319--326},
  year    = {2025}
}
```

## 2. McElfresh et al. 2023 — NN vs. GBDT meta-study
- Confirmed via [DBLP/NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f06d5ebd4ff40b40dd97e30cee632123-Abstract-Datasets_and_Benchmarks.html):
  McElfresh, D. C., Khandagale, S., Valverde, J., Prasad C., V., Ramakrishnan, G.,
  Goldblum, M., White, C. "When Do Neural Nets Outperform Boosted Trees on Tabular
  Data?" *NeurIPS 2023, Datasets and Benchmarks Track.* (176 datasets, 19 algorithms.)
- Note: the **arXiv** version (2305.02997) lists two extra authors (Feuer,
  Hegde) added in a later revision; the published NeurIPS camera-ready has the
  7-author list above — cite the venue authors, matching how the rest of the bib
  cites the published form (cf. the `wang1997m5prime` note on circulating titles).
- **Maps to:** Tabular deep learning paragraph, right next to `grinsztajn2022why`.
  This is the evidentiary complement to Grinsztajn: it says the NN-vs-GBDT gap is
  often within noise/HPO-budget effects, which is exactly the caution the paper's
  own "shared-HPO-budget, mid-pack, not SOTA" framing already takes seriously —
  cite it to justify *why* a fair-budget, significance-tested comparison (rather
  than a leaderboard-topping claim) is the right bar for RepLeafGBM's own results.

```bibtex
@inproceedings{mcelfresh2023when,
  author    = {McElfresh, Duncan C. and Khandagale, Sujay and Valverde,
               Jonathan and {Prasad C.}, Vishak and Ramakrishnan, Ganesh and
               Goldblum, Micah and White, Colin},
  title     = {When Do Neural Nets Outperform Boosted Trees on Tabular Data?},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS),
               Datasets and Benchmarks Track},
  year      = {2023}
}
```

## 3. TabArena — living tabular benchmark
- Confirmed from the paper PDF first page ([papers.neurips.cc](https://papers.neurips.cc/paper_files/paper/2025/file/1697e3fb412da11dc9488249f9e7bbc9-Paper-Datasets_and_Benchmarks_Track.pdf),
  also [arXiv:2506.16791](https://arxiv.org/abs/2506.16791)): Erickson, N.,
  Purucker, L., Tschalzev, A., Holzmüller, D., Mutalik Desai, P., Salinas, D.,
  Hutter, F. "TabArena: A Living Benchmark for Machine Learning on Tabular Data."
  *39th Conference on Neural Information Processing Systems (NeurIPS 2025), Track
  on Datasets and Benchmarks* (spotlight).
- **Maps to:** either the tabular-DL paragraph (as the up-to-date leaderboard
  context) or the intro/experiments framing near the Grinsztajn discussion. Its
  headline finding — GBDTs remain strong defaults, DL catches up only with heavy
  tuning/ensembling, foundation models (TabPFNv2/TabICL) excel on small data — is
  the 2025 update of exactly the Grinsztajn thesis the paper already leans on, and
  independently corroborates why "mid-pack among tuned tree/DL baselines" is a
  meaningful, non-trivial result rather than a modest one.

```bibtex
@inproceedings{erickson2025tabarena,
  author    = {Erickson, Nick and Purucker, Lennart and Tschalzev, Andrej and
               Holzm{\"u}ller, David and {Mutalik Desai}, Prateek and Salinas,
               David and Hutter, Frank},
  title     = {{TabArena}: A Living Benchmark for Machine Learning on Tabular
               Data},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS),
               Datasets and Benchmarks Track},
  year      = {2025}
}
```

## 4. NGBoost — probabilistic/natural-gradient boosting
- Confirmed via PMLR: Duan, T., Avati, A., Ding, D. Y., Thai, K. K., Basu, S.,
  Ng, A. Y., Schuler, A. "NGBoost: Natural Gradient Boosting for Probabilistic
  Prediction." *ICML 2020* (PMLR vol. 119, pp. 2690–2700).
  [proceedings.mlr.press/v119/duan20a](http://proceedings.mlr.press/v119/duan20a.pdf).
- **Maps to:** GBDTs paragraph, as a *leaf-target* variant rather than a
  *leaf-basis* variant: NGBoost keeps constant/shallow leaves but boosts
  distributional parameters jointly via a natural-gradient correction. Cite it as
  a contrast, not a competitor: it changes *what the leaf predicts* (distribution
  params) while RepLeafGBM changes *what the leaf predicts on* (a learned
  representation); both are compatible in principle (representation-conditioned
  leaves could in future feed NGBoost-style multi-parameter targets), but v0 does
  not do this — flag as a possible future-work pointer, not an implemented overlap.

```bibtex
@inproceedings{duan2020ngboost,
  author    = {Duan, Tony and Avati, Anand and Ding, Daisy Yi and Thai,
               Khanh K. and Basu, Sanjay and Ng, Andrew Y. and Schuler,
               Alejandro},
  title     = {{NGBoost}: Natural Gradient Boosting for Probabilistic
               Prediction},
  booktitle = {Proceedings of the 37th International Conference on Machine
               Learning (ICML)},
  pages     = {2690--2700},
  year      = {2020}
}
```

## 5. EBM / GA2M — cite **both** Lou et al. KDD'12 and KDD'13
- Lou, Y., Caruana, R., Gehrke, J. "Intelligible Models for Classification and
  Regression." *KDD 2012*, pp. 150–158.
  [DOI 10.1145/2339530.2339556](https://dl.acm.org/doi/10.1145/2339530.2339556).
- Lou, Y., Caruana, R., Gehrke, J., Hooker, G. "Accurate Intelligible Models with
  Pairwise Interactions." *KDD 2013*, pp. 623–631. (This is the paper that
  introduces **GA2M** — pairwise shape functions — which InterpretML's EBM
  implements.) [DOI 10.1145/2487575.2487579](https://dl.acm.org/doi/10.1145/2487575.2487579).
- Recommendation: cite **both** (2012 for the single-feature GAM/shape-function
  boosting formulation, 2013 for GA2M's pairwise extension) — this is standard
  practice in the EBM literature and DBLP confirms both records cleanly. Skip the
  InterpretML software paper (Nori et al. 2019, arXiv) as a third citation unless
  the paper wants to point at the open-source package specifically.
- **Maps to:** Model trees / piecewise-linear leaves paragraph, as the sibling
  design point: EBM/GA2M is also "boosting with a non-constant leaf," but the leaf
  there is a *per-feature (or per-feature-pair) shape function fit on raw
  features*, and — critically — **routing and leaf basis are the same raw
  features** (each weak learner is a shallow tree/shape function over 1–2 raw
  dims). RepLeafGBM's leaf basis is instead a learned, frozen, possibly
  high-dimensional $\zt(x)$ decoupled from the routing features. Good contrast
  sentence: EBM raises leaf expressivity by composing many raw-feature shape
  functions; RepLeafGBM raises it by conditioning on a representation instead.

```bibtex
@inproceedings{lou2012intelligible,
  author    = {Lou, Yin and Caruana, Rich and Gehrke, Johannes},
  title     = {Intelligible Models for Classification and Regression},
  booktitle = {Proceedings of the 18th ACM SIGKDD International Conference on
               Knowledge Discovery and Data Mining (KDD)},
  pages     = {150--158},
  year      = {2012}
}

@inproceedings{lou2013accurate,
  author    = {Lou, Yin and Caruana, Rich and Gehrke, Johannes and Hooker,
               Giles},
  title     = {Accurate Intelligible Models with Pairwise Interactions},
  booktitle = {Proceedings of the 19th ACM SIGKDD International Conference on
               Knowledge Discovery and Data Mining (KDD)},
  pages     = {623--631},
  year      = {2013}
}
```

## 6. LightGBM `linear_tree` — already covered, no new entry needed
- LightGBM's own docs cite Shi/Li/Li directly for the `linear_tree` parameter's
  regularization (confirmed via the Parameters page,
  [lightgbm.readthedocs.io/.../Parameters.html](https://lightgbm.readthedocs.io/en/stable/Parameters.html),
  which links "Gradient Boosting with Piece-Wise Linear Regression Trees" as the
  source). **`shi2019gbdtpl` is already in `references.bib` and is already cited**
  in the "Model trees and piecewise-linear leaves" paragraph — this is the correct
  and sufficient citation; do not add a second entry for the LightGBM feature
  itself (it is an engineering artifact of the paper, not an independent
  citable work; a footnote `\url{}` to the docs page is the right vehicle *if* the
  paper wants to name-drop "as implemented in LightGBM," not a bib entry).
- **Action:** none required for the bibliography; optionally add one clause in the
  existing paragraph — "...also implemented as LightGBM's `linear_tree` option" —
  with a footnote URL, no new bibtex key.

## Sanity pass: any other obviously-expected "trees + learned representation" citation?
- TabNet/NODE/FT-Transformer/DeepGBM/DNDF/Grinsztajn: already cited, confirmed above.
- One gap worth flagging for `research-proposer` to consider (not verified in full
  bibtex detail here, out of scope of the six requested): **GRANDE** (Marton et
  al., "Gradient-Based Decision Tree Ensembles for Tabular Data," ICLR 2024,
  [arXiv:2309.17130](https://arxiv.org/abs/2309.17130)) makes *routing itself*
  differentiable/soft over raw features via a straight-through estimator — a 2024
  entry in the same "Hybrids of trees and representations" paragraph as
  DNDF/NODE/DeepGBM. It would freshen that paragraph's recency but is not required
  by this task; needs its own verification pass before citing.

## Guardrail check
None of the six violate the routing/freezing invariants (I1/I2) — they are all
citation-only additions describing *other* methods, not code changes. Framing
guardrail: TabPFN/TabArena/McElfresh must be presented as *context for the
mid-pack claim*, not as competitors RepLeafGBM claims to beat — the paper's own
honesty (mid-pack, not SOTA) must survive the addition of these citations, not be
undercut by implying we outperform foundation models we haven't benchmarked
against.

## Next steps
1. Hand this note to `research-proposer` (or directly to the paper editor) to
   insert the six bibtex entries into `references.bib` and thread one sentence per
   item into the four Related Work paragraphs as mapped above.
2. Decide whether to spend a verification pass on GRANDE (ICLR 2024) before adding
   it to the Hybrids paragraph.
3. If NGBoost is cited, consider adding the one-line future-work pointer
   (representation-conditioned leaves feeding multi-parameter/NGBoost-style
   targets) to `docs/roadmap.md` so the paper's forward-looking claim has a
   tracked home.
