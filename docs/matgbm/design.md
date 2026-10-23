# MatGBM 設計契約

## 空間バンク

空間 `T_k` は単なる関数ではなく、状態を持つ写像である。各空間は次の lifecycle を明示する。

1. `fit_reference(X_reference, y_reference=None)` — 許可された参照行だけで状態を fit する。
2. `fit_transform_train(X_train, y_train, sample_weight=None)` — 学習行に渡す値を返す。competition label を読む空間では outer-train 内 OOF 値でなければならない。
3. `transform(X_query)` — 凍結した状態だけで評価・予測行を変換し、状態を変えない。

各 artifact の provenance は最低限 `fit_row_hash`、`label_row_hash`、`fold_hash`、`reference_scope`、`test_rows_used`、`generator/model_hash`、schema、列順を持つ。hash は対応確認であり、リークが無いことの証明ではない。

判断源: `ideas.md` の前提訂正、`answer-C4int-sol.md` の第 1 波。

## leaf-only 境界

`leaf_only_features` に指定した有限数値列は、raw routing、split candidate、feature sampling、importance から除外する一方、encoder の ordered numerical block には加える。カテゴリ、NaN、inf は受け付けない。`leaf_experts` は各 block を凍結し、学習行には bundle 経路の `fit_transform_train`、評価・予測行には通常の `transform` を用いる。

この境界は raw-only を信条ではなく検証可能な既定にする。派生列を router にも渡す場合は、同じ空間の factorial 対照でだけ覆す。

判断源: ADR 0011、`order-M12-leaf-experts-select.md`。

## select と winner's curse

> `select` は 1.14.0 で削除した（ADR 0012）。波 4 で固定 2 ブロック S8+O に 0/5 で負けた（KS1 −0.000722、bar 0.000128）。以下は削除前（1.13.0）の契約の記録である。

各葉で constant と eligible expert block を比較する。block `k` は `n_leaf >= d_k + 2` のとき h 重み付き ridge を fit し、weighted LOO 誤差 `E_k` を求める。constant の誤差を `E_const` とし、

`k* = argmin_k E_k`、`E_k* < (1 - leaf_gate_margin) * E_const`

をともに満たすときだけ `k*` を選ぶ。strict な不等号なので同点は constant である。選ばれない block の係数は 0 とする。

K 個の noisy LOO 最小値には winner's curse がある。v1 は `leaf_gate_margin` と行置換した false block の回帰テストだけで抑え、複雑度罰則や null-bank 較正は未実装である。よって `select` は第 1 波の研究 arm であり、fixed、concat、constant、permuted selector に採否幅超で勝つまで既定化しない。

判断源: ADR 0011、`answer-C4int-sol.md` の裁定②。

## リーク契約

outer fold の学習集合を `T_o`、評価集合を `V_o` とする。competition label を使う TE、self-teacher、空間選択、early stopping、stack 重みは、`T_o` 内の inner fold だけで作る。`V_o` の label は全祖先から封印し、`V_o` 用の列は `T_o` 全体で fit した凍結写像から作る。global OOF を後から切り出して nested OOF の代わりにはできない。

multi-pass の停止、候補選択、stack 重みも inner のみで決める。outer-valid の改善で停止しない。artifact は少なくとも `(outer, inner, pass, role=train_oof|eval_full)` と、親、fit/query/reference/label ID、row order、schema、code/config/state hash を保存する。

判断源: `answer-C5-{sol,astra}.md`。

## multi-pass と内部 ensemble

multi-pass は、学習集合・空間・行源・重み・木数を確定してから一本の GBM を最初から fit する、最大 2 pass の**ハーネス実験モード**である。前 pass の木や encoder を上書きしない。self-teacher は新しいラベル情報を増やさず、再表現または正則化としてのみ仮説を持つ。

内部 ensemble は複数学習系列と OOF 重み学習を含むため、単体 MatGBM の中核ではなく明示 wrapper とする。pass 間の補完性は固定 logit 平均で先に測り、通過後だけ nested stack を比較する。

判断源: `answer-C4int-{sol,astra}.md`、`answer-C5-{sol,astra}.md`。

## 未決と変心条件

| 論点 | 現状 | 変心条件 |
|---|---|---|
| dynamic space | 第 1 波外 | 独立 pilot で動的 PLS が静的 PLS・偽 PLSに +0.00025。 |
| pseudo | 第 2 波 | `C−B > bar`、4/5 正、別 partition 再現。 |
| logreg / IRLS 葉 | 第 2 波 | ridge 比で平均 `> bar`、4/5 正、logloss 非悪化、時間 2 倍以内。 |
| original 行 | 第 2 波の 4 対照 | 選択的追加が無し・一律・同量 random を `bar` と 4/5 で超える。 |

上の数値は `+0.00025` を除き原本の判定則であり、`+0.00025` は置いた数である。判断源: `answer-C4int-sol.md`、`answer-C5-sol.md`。
