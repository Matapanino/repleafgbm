# MatGBM review profile

この profile は MatGBM 変更の review-loop 用受入境界である。判断源: ADR 0011、ADR 0012、ADR 0013、`order-M12-leaf-experts-select.md`、`answer-C4int-sol.md`。

## 危険クラス

- **bitwise 退行**: `leaf_only_features=None` と expert 未指定の既存経路は予測 bitwise 同一を保つ。
- **routing 列添字**: leaf-only を raw routing、sampled features、importance から除きつつ、ordered numerical block の末尾に指定順で置く。
- **native 隔離**: bundle、leaf model、data 層に `torch`、`lightgbm`、`cupy` を import しない。histogram/splitter/Rust/CUDA の契約を変更しない。
- **直列化の版**: leaf-only/expert 使用時だけ format v9、既存組合せは旧版、reader は 1〜9 を扱う。
- **K 腕 LOO 過選択**: 葉ごとの block 選択（`select`）は 1.14.0 で削除した（ADR 0012）。再導入する変更では constant を候補に含め、strict margin と置換 false block を検査し、未較正の winner's curse を性能主張で隠さない。
- **OOF リーク**: 学習行の教師あり表現は `fit_transform_train`、eval/predict は凍結 `transform`。outer-valid label は producer、選択、停止、重み学習に渡さない。

## 機械検査可能な受入境界

- leaf-only 指定列が split feature、`get_raw_features()`、importance、列サンプリング母集団に入らない。
- leaf-only 未指定かつ expert 未指定で固定 fixture の予測が `np.array_equal`。
- `leaf_model="select"` は fit・factory・router extraction で `ValueError`、`select` で保存された 1.13.x のモデルの load も `ValueError`（版によらず config の `leaf_model` で判定）。
- 削除した空間 encoder 名（`pca`・`quantile`・`spline`・`rff`・`row_stats`・`prototype`）は `make_encoder`・`encoder=`・`leaf_experts` で `ValueError`、それらで保存された 1.13.x のモデルの load も `ValueError`（top-level・wrapper の base・bundle の child のどこにあっても名前で判定）。
- `fit_transform_train != transform` の dummy encoder で train は前者、eval/predict は後者を受ける。
- `leaf_experts` と `encoder` の同時指定、カテゴリ・NaN・inf の leaf-only を拒否する。
- 2 ブロック `leaf_experts` の `embedded_linear` は回帰・二値分類とも fit/predict/save-load が通り、format v9 を書く。
- provenance artifact は禁止 label ID との祖先交差、行順、重複 ID、schema、親 hash 欠落を失敗とする。

## sibling 契約

`adaptive` は constant 対ひとつの線形表現の LOO gate である。`embedded_linear` は expert block を連結した固定 ridge であり、block 選択を行わない（block 選択の `select` は 1.14.0 で削除）。両者とも raw routing と凍結 encoder lifecycle を保つ。

## 既知の失敗カタログ

- [ADR 0011](../adr/0011-leaf-only-features-and-expert-selection.md): K 腕 winner's curse、scalar-only select（select 部分は ADR 0012 で Superseded）。
- [ADR 0012](../adr/0012-remove-leaf-select.md): select は固定ブロックに 0/5 で負けて削除。1.13.x の select モデルは 1.14.0 で読めない。
- [ADR 0013](../adr/0013-remove-space-encoders.md): 空間 encoder 6 種は波 4 の選別で候補ゼロのため削除。1.13.x でそれらを使ったモデルは 1.14.0 で読めない（pickle は `ModuleNotFoundError`）。
- [MatGBM design](../matgbm/design.md): nested OOF、outer 停止禁止、pseudo/reference 分離。
- [MatGBM experiment protocol](../matgbm/experiment-protocol.md): 置換、seed、route replay と 4 条件 factorial の対照。
