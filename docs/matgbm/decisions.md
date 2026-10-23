# MatGBM の判断記録

このページは `notes/pretrained-gbdt/decisions.md` の転記である。**「仮決定（Masaya 裁定待ち）」は Masaya の確定裁定ではない。**

| 日付 | 区分 | 問い | 決定 | 決めた人 | 根拠・出典 |
|---|---|---|---|---|---|
| 2026-10-22 | 中 | MatGBM の器をどう統一するか | 空間を `leaf_experts=[...]` と葉ごとの selector で列挙し、事前学習埋め込み、TE、original teacher を同じ expert とする。ICL 葉は将来の `ICLExpert`。 | Masaya | `answer-C1b-sol.md`、`answer-C1-sol.md` |
| 2026-10-22 | 中 | 第 1 実験をどれにするか | identity／TE 要約／original teacher × weak/strong router、連結 ridge・global selector・行置換偽空間を対照にする CPU 因子実験。noise bar は実測 0.000125。 | Masaya | `answer-C1b-sol.md`、Kaggle quota（2026-10-22） |
| 2026-10-22 | 軽 | 実験の計算資源 | Kaggle private CPU kernel。ローカル Mac で fit しない。 | Masaya | `competition/docs/HARNESS.md`、ADR-0013 |
| 2026-10-22 | 中 | 北極星 | 効いていた強い部分を一つのシンプルな GBM に落とし込む。採否はこの目的への寄与で判断する。 | Masaya | 本人の発言 |
| 2026-10-22 | 重（事前承認） | 新リポジトリ | 切り分けが必要なら `Matapanino/matgbm` を作ってよい。当面は repleafgbm 拡張で保留し、wrapper が製品中心になる判断時は事前に一行通知する。 | Masaya | 本人の発言、C1/C1b/C4 |
| 2026-10-22 | 中（仮決定・Masaya 裁定待ち） | C4 統合の 5 件 | ①3 機構の中核、②第 1 波 I/T/O、③select は既定化せず実験 arm、④raw-only 既定で factorial、⑤内部 ensemble は wrapper。 | Fable | `answer-C4int-sol.md` |
| 2026-10-22 | 中 | 成功基準 | 単体 AUC が最強単体に bar 内以上、かつ強いモデルと異なる OOF シグナルと bar 超の stack 増分。両方必要。 | Masaya | 本人の発言 |
| 2026-10-22 | 中（仮決定・Masaya 裁定待ち） | C4int-astra／C5 の決めること | 3 機構を中核、内部 ensemble と multi-pass は別モード、select は同時比較、dynamic は第 1 波外、reference/pseudo 分離、multi-pass は 2 pass・inner 停止、original 行は第 2 波 4 対照。 | Fable | `answer-C4int-astra.md`、`answer-C5-{sol,astra}.md` |
| 2026-10-23 | 重 | leaf select（`leaf_model="select"`）と Rust `leaf_select_stats` を削除するか | 波 4 の事前登録判定に従う（Masaya「2A」）。KS1: select − 固定 S8+O = −0.000722（0/5・bar 0.000128）で reject → 1.14.0 で削除（PR #71・ADR 0012）。空間 encoder 6 種は KE で候補ゼロ、削除範囲は別裁定（1B）。 | Masaya | `notes/pretrained-gbdt/ideas.md`「波 4 確定」、`docs/adr/0012-remove-leaf-select.md`、ハーネス `matgbm_decision/wave4-decision.md` |
| 2026-10-24 | 重 | 空間 encoder 6 種（`pca`・`quantile`・`spline`・`rff`・`row_stats`・`prototype`）をどこまで registry から外すか | 1B（2026-10-23「KE で負けた encoder は 1.14.0 の registry から外す」）と 1A（2026-10-24「6 種全部」）。KE は候補ゼロ（spline −0.000364・0/5、他 5 種は bar 0.000128 内、restack add-one は 6 種とも Δ ≤ 0）→ 1.14.0 で 6 種とも削除（ADR 0013）。 | Masaya | `notes/pretrained-gbdt/ideas.md`「波 4 確定」、`docs/adr/0013-remove-space-encoders.md`、ハーネス `matgbm_decision/wave4-decision.md` |

判断源: `notes/pretrained-gbdt/decisions.md`（2026-10-22）。
