# MatGBM review profile

この profile は MatGBM 変更の review-loop 用受入境界である。判断源: ADR 0011、`order-M12-leaf-experts-select.md`、`answer-C4int-sol.md`。

## 危険クラス

- **bitwise 退行**: `leaf_only_features=None` と expert 未指定の既存経路は予測 bitwise 同一を保つ。
- **routing 列添字**: leaf-only を raw routing、sampled features、importance から除きつつ、ordered numerical block の末尾に指定順で置く。
- **native 隔離**: bundle、leaf model、data 層に `torch`、`lightgbm`、`cupy` を import しない。histogram/splitter/Rust/CUDA の契約を変更しない。
- **直列化の版**: leaf-only/expert 使用時だけ format v9、既存組合せは旧版、reader は 1〜9 を扱う。
- **K 腕 LOO 過選択**: constant を候補に含め、strict margin と置換 false block を検査する。未較正の winner's curse を性能主張で隠さない。
- **OOF リーク**: 学習行の教師あり表現は `fit_transform_train`、eval/predict は凍結 `transform`。outer-valid label は producer、選択、停止、重み学習に渡さない。

## 機械検査可能な受入境界

- leaf-only 指定列が split feature、`get_raw_features()`、importance、列サンプリング母集団に入らない。
- leaf-only 未指定かつ expert 未指定で固定 fixture の予測が `np.array_equal`。
- single expert の `select` と `adaptive` が allclose、行置換 false block の選択率が真 block を上回らない。
- `fit_transform_train != transform` の dummy encoder で train は前者、eval/predict は後者を受ける。
- `leaf_experts` と `encoder` の同時指定、カテゴリ・NaN・inf の leaf-only を拒否する。
- scalar 以外の `select` は `NotImplementedError`。save/load、同 seed、NumPy/Rust routing を検査する。
- provenance artifact は禁止 label ID との祖先交差、行順、重複 ID、schema、親 hash 欠落を失敗とする。

## sibling 契約

`adaptive` は constant 対ひとつの線形表現の LOO gate であり、single-block `select` の比較基準である。`embedded_linear` は expert block を連結した固定 ridge の比較 arm であり、select の係数ゼロ化や block 選択を行わない。両者とも raw routing と凍結 encoder lifecycle を保つ。

## 既知の失敗カタログ

- [ADR 0011](../adr/0011-leaf-only-features-and-expert-selection.md): K 腕 winner's curse、scalar-only select、native selector 未実装。
- [MatGBM design](../matgbm/design.md): nested OOF、outer 停止禁止、pseudo/reference 分離。
- [MatGBM experiment protocol](../matgbm/experiment-protocol.md): 置換、seed、route replay と 4 条件 factorial の対照。
