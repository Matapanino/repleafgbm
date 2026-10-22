# MatGBM 実験プロトコル

根拠等級は **実測**（保存済み報告の数値）、**目安**（既存 lane からの見積り）、**置いた数**（未検証の運用値）で表す。回をまたぐ数値を合算しない。

## 第 1 波: 確定案

### データと fold

s6e10 train は 699,635 行、`id` を除く predictor は 21 列。`StratifiedKFold(5, shuffle=True, random_state=42)` を固定する。original 129,880 行は teacher/origTE 生成だけに使い、competition 行との exact match は除外する。outer-valid の label は TE、teacher、空間選択、early stopping から封印する。

fold 0 の 100k train / 40k valid / 200 木は整合性・時間測定用のみで、採否には使わない。

根拠: `answer-C4int-sol.md`（実測のデータ数・置いた運用分割）。

### 空間 I/T/O と交換 arm

| 空間 | 定義 | 由来・契約 |
|---|---|---|
| `I_identity` | 標準化した数値 17 列 | fit rows、列順、標準化状態、列 hash を保存。 |
| `T_oof_te` | exact、round ladder、quantile bin を固定した 8〜16 列 | outer-train 内 OOF。valid は outer-train full-fit。competition label を使う。 |
| `O_teacher` | original-only teacher の**確率** 1 列 | teacher model hash、original row hash、label source、重複除外件数を保存。 |

`O_origTE`、`T_genLLR`、`T_route_profile` は `O` と一対一で交換する arm とし、K を増やさない。BPE prefix は既存 artifact の hash が通る場合だけ交換 arm にできる。

全 artifact に `fit_row_hash`、`label_row_hash`、`fold_hash`、`reference_scope`、`test_rows_used`、`generator/model_hash`、schema、列順を必須とする。

根拠: `answer-C4int-sol.md`（実測の候補・契約）。teacher は logit ではなく確率列だったことに注意する。

### arm と対照

各 `S ∈ {T, O}` に 4 条件 factorial を置く。

| arm | router | leaf |
|---|---|---|
| `B` | raw | constant |
| `L` | raw | fixed `S` |
| `R` | raw + `S` | constant |
| `RL` | raw + `S` | fixed `S` |

raw routing の leaf 比較は constant、I-only、T-only、O-only、I+T+O concat、global selector、leaf select とする。router は weak prefix と inner stopping 済み strong を別因子にし、浅さは `num_leaves/max_depth`、木数は `n_estimators` として別に記録する。

対照は T/O の行置換 block、constant の seed-repeat、同じ route の leaf replay。採用候補だけ native grower で再確認する。

根拠: `answer-C4int-sol.md`（確定案）。

### 採否

`bar = max(0.000125, 今回の seed-repeat で観測した最大絶対 per-fold 差)` とする。0.000125 は原本が採った既存 strong XGB seed 差の丸め値であり、MatGBM の信頼区間ではない。

- fixed 空間は `L − B > bar`、4/5 fold 正、pooled OOF 正。
- raw-only 変更は `R − B` または `RL − L` が `> bar`、4/5 fold 正。
- selector の既定化は constant、best-single、concat、permuted selector のすべてに平均 `> bar`、各 4/5 fold 正。
- 未達なら空間と selector の採用を分け、境界値は hold とする。

根拠: `answer-C4int-sol.md`。`bar` は判定用の置いた規則であり、元となる 0.00012503 は既存実測である。

### 診断と CPU 分割

各葉について `Var_h(T_k)`、特異値からの有効 rank、サイズ不足/数値失敗別 fallback 率、葉数・行数・Σh 別の選択率を保存する。併せて `n_eff`、最大 leverage、勝者対 constant/次点の余裕、fit 秒、peak RSS を保存する。

所要は 6〜12 CPU 時間、壁時計 4〜10 時間、peak 5〜15GB の**目安**であり、M12 実測ではない。最初の 15〜30 分 smoke で再見積りする。5 outer fold を 5 private CPU kernel へ分け、さらに factorial / selector / permuted+seed の 3 波に分ける。1 単位が 8 時間超見込みなら 1 arm 単位へ細分化し、arm 完了ごとに receipt を作る。

根拠: `answer-C4int-sol.md`（目安と運用案）。

## 成功基準

1. **単体強度**: MatGBM の単体 AUC が最強単体に対して bar 内以上であること。s6e10 の比較値は TabPFN 0.9616、RealMLP 0.9612、CatBoost 0.9610（実測として記録された参照値）。
2. **補完性**: 強いモデルとの OOF 誤差相関を測り、stack 増分が bar を超えること。低相関だけでは採用しない。

根拠: Masaya の成功基準（2026-10-22）、`decisions.md`。

## 第 2 波

### multi-pass: 7 本比較

最大 2 pass のハーネス実験で、pass 停止は inner だけで決める。比較対象は次の 7 本である。

1. `P1-base`
2. 同一空間・同一 seed 再 fit（決定性/no-op 対照）
3. seed 変更
4. colsample 変更
5. `P2-self`: P1 OOF logit 1 列を追加
6. `P2-self-perm`: 同じ周辺分布の行置換列を追加
7. `P2-external`: provenance-distinct block を一つ追加（既定は genLLR、未使用なら original teacher）

flat OOF の Spearman と logit Pearson を保存する。`ρ ≥ 0.999` なら多様性仮説を棄却、`ρ ≤ 0.985` は目標域だが採用条件ではない。cross-family 級を名乗るには +0.00025（**置いた数**）も必要とする。

根拠: `answer-C5-sol.md`。初回は固定 route replay で共有木の寄与を切り分け、有望な self/external だけ native grow へ進める。

### original 行: 4 対照

competition 行だけで route を作り、original 行は同じ葉の leaf fit の `g,h` にだけ加える。比較は **無し／全 original を一律 0.3／選択的追加／選択的追加と同じ行数・総 Hessian の random**。係数 0.3 や局所 gate の閾値はすべて未検証の置いた数である。

選択的追加が無追加・一律追加・同量 random を `bar` と 4/5 fold で超えた場合だけ進める。pseudo は teacher 由来または fit 安定化目的に限り、この波で別契約として扱う。

根拠: `answer-C5-sol.md`、`ideas.md`。s6e6 の一律 original 追加 +0.0012 と s6e7 の −0.0004 は別回の実測であり、平均化しない。

### IRLS

分類葉では 1-step ridge と 3-step IRLS/full logreg を同じ葉で比較する。IRLS は ridge 比で平均 `> bar`、4/5 fold 正、logloss 非悪化、総時間 2 倍以内を通るまで中核に入れない。

根拠: `answer-C4int-sol.md`（判定則、IRLS は第 2 波）。
