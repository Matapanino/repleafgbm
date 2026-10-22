# MatGBM

## 北極星

> 「ここ数ヶ月で効いていた強い部分を全て、一つのシンプルな GBM に落とし込む。」 — Masaya, 2026-10-22

MatGBM はその器である。ただし、単一 fitted GBM に移せない cross-family stack や seed bag の独立誤差は、限界として残す。

**成功基準**は二つをともに満たすこと。第一に、単体 AUC が最強単体（s6e10 では TabPFN 0.9616、RealMLP 0.9612、CatBoost 0.9610）に対して採否幅内以上であること。第二に、強いモデルと異なる OOF シグナルを持ち、stack 増分が採否幅を超えること。これは Masaya の発言（2026-10-22）を判断源とする。

## 統合設計（10 行）

1. MatGBM は `RepLeafClassifier` による一本の逐次 boosting 系列である。
2. 分岐は既定で raw feature のみを使い、`leaf_only_features` が派生列を隠す。
3. 葉には `leaf_experts` で凍結済みの小さな空間を渡す。
4. 中核の葉解法は h 重み付き ridge、分類では 1 回 Newton 更新である。
5. `leaf_model="select"` は constant を含む候補から weighted LOO と `leaf_gate_margin` で一つを選ぶ。
6. v1 の bank は identity、fold-safe TE、original-only teacher の最大 3 空間で始める。
7. competition label を使う空間は `fit_transform_train` で OOF 化し、評価・予測行は凍結写像で変換する。
8. 未ラベル参照は将来の `X_unlabeled`、pseudo は別引数・別損失として混ぜない。
9. 動的 snapshot、複数反復 IRLS、pseudo 自動生成、内部 ensemble は中核から外す。
10. 巨大 TE bank、mutable encoder、同一 train MLP の量産、provenance のない空間は採らない。

判断源: `answer-C4int-sol.md` の統合設計（2026-10-22）。

## 中核の 3 機構

- **raw routing**: 木の分岐は raw 列で行う。派生列は `leaf_only_features` により分岐、列サンプリング、importance から除外する。
- **凍結空間バンク**: `leaf_experts` の named encoder block を boosting 前に fit して固定する。教師あり空間は学習行だけ `fit_transform_train` の OOF 値を使う。
- **ridge と選択**: 固定空間は `embedded_linear` の連結 ridge、局所比較は `select` の per-leaf weighted LOO で扱う。constant は常に候補である。

判断源: `answer-C4int-{sol,astra}.md` と ADR 0011。

## 最小 API 例

```python
from repleafgbm import RepLeafClassifier, RepLeafRegressor

clf = RepLeafClassifier(
    leaf_model="select",
    leaf_only_features=["te_feature", "teacher_probability"],
    leaf_experts=[
        {"name": "te", "encoder": "identity", "columns": ["te_feature"]},
        {"name": "teacher", "encoder": "identity", "columns": ["teacher_probability"]},
    ],
    leaf_gate_margin=0.01,
).fit(X_train, y_train)

reg = RepLeafRegressor(leaf_model="embedded_linear").fit(X_train, y_train)
```

この API は現物の公開クラス名・引数名に基づく。上の特徴量 producer と OOF/provenance の実装は、現時点では別ハーネスの責務である。

## 中核から外したもの

| 対象 | 現状 | 理由 |
|---|---|---|
| 動的空間生成 | 第 1 波外 | 静的空間との増分をまだ測っておらず、状態管理を増やす。 |
| pseudo | 第 2 波 | soft pseudo は約 −0.000021、hard は −0.000300 の既測がある。 |
| 複数反復 IRLS / logreg 葉 | 第 2 波 | 新情報源より先に solver を複雑化する根拠がない。 |
| 内部 ensemble | 明示 wrapper | 複数学習系列であり、「一つの GBM」の中核ではない。 |
| 巨大 TE bank / mutable encoder | 不採用 | K 腕の過選択とリーク面を増やす。 |
| 同一 train の MLP 量産 | 不採用 | learned MLP 葉 .960104715 は constant .960107425 と同値域だった。 |

判断源: `answer-C4int-sol.md` の裁定表、`answer-C5-{sol,astra}.md`。

## ノート索引

ノートは作業履歴であり、設計の正本はこのディレクトリである。

| ノート | 内容 |
|---|---|
| `ideas.md` | 北極星、候補空間、実測の出発点、C2〜C5 の取り込み。 |
| `decisions.md` | Masaya の決定と裁定待ちの仮決定の時系列。 |
| `answer-C4int-sol.md` | 10 行設計、裁定表、第 1 波、実装分割。 |
| `answer-C4int-astra.md` | C4 統合の別評価、契約と反論。 |
| `answer-C5-sol.md` | multi-pass、nested OOF、diversity、original 行の設計。 |
| `answer-C5-astra.md` | multi-pass の独立批評と最小実験案。 |
| `answer-C3-sol.md` | 効いたレバー 38 件の棚卸しと MatGBM への写像。 |
| `order-M12-leaf-experts-select.md` | leaf-only、expert bundle、select の実装仕様。 |
| `log-M12-progress.md` | M12 実装・テスト・コミットの経過記録。 |
