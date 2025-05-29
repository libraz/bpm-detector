# BPM & Key Detector

[![codecov](https://codecov.io/gh/libraz/bpm-detector/branch/main/graph/badge.svg)](https://codecov.io/gh/libraz/bpm-detector)
[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

音楽ファイルのBPM（テンポ）・キー（調）検出から包括的な楽曲解析まで対応した音楽制作支援ツールです。

## 機能

### 基本解析
- **BPM検出**: 高精度なテンポ検出アルゴリズム
  - 自動的な高速/低速レイヤー選択
  - ハーモニッククラスタリング
  - 信頼度スコア付き
- **キー検出**: 音楽理論に基づいたキー検出
  - Krumhansl-Schmucklerキープロファイル使用
  - メジャー・マイナー両方に対応
  - クロマ特徴量ベースの解析

### 包括的楽曲解析（新機能！）
- **コード進行解析**: 自動和音検出とハーモニー解析
  - コード進行の識別（C-Am-F-G）
  - 機能和声分析（I-vi-IV-V）
  - 転調検出
  - コード複雑度スコア
- **楽曲構造解析**: 自動セクション検出と楽曲形式分析
  - セクション境界（イントロ、Aメロ、サビ、ブリッジ）
  - 楽曲形式の識別（ABABCB）
  - 反復パターン検出
  - 構造複雑度分析
- **リズム・グルーヴ解析**: 詳細なリズムパターン分析
  - 拍子検出（4/4、3/4、6/8等）
  - グルーヴタイプ分類（ストレート、スウィング、シャッフル）
  - シンコペーション度測定
  - リズム複雑度スコア
- **音色・楽器編成**: 音響テクスチャと楽器分析
  - 楽器分類（ピアノ、ギター、ドラム等）
  - 音色特徴（明度、温かさ、粗さ）
  - エフェクト使用度検出（リバーブ、ディストーション、コーラス）
  - 音響密度分析
- **メロディー・ハーモニー解析**: 音楽的内容分析
  - メロディー音域と輪郭分析
  - ハーモニー複雑度測定
  - 協和・不協和度評価
  - 音程分布分析
- **ダイナミクス・エネルギー**: 音響ダイナミクスとエネルギープロファイル
  - ダイナミックレンジ分析
  - エネルギープロファイル生成
  - クライマックスポイント検出
  - ラウドネス分析
- **音楽制作参考資料**: 自動参考資料生成
  - 制作ノートと推奨事項
  - 類似楽曲特徴
  - 楽曲発注用参考タグ
  - 類似度マッチング用特徴ベクトル生成

## クイックリンク

- 📦 [PyPIパッケージ](https://pypi.org/project/bpm-detector/) (近日公開予定)
- 🐳 [Dockerイメージ](https://github.com/libraz/bpm-detector/pkgs/container/bpm-detector)
- 📊 [テストカバレッジ](https://codecov.io/gh/libraz/bpm-detector)
- 🔧 [CI/CDステータス](https://github.com/libraz/bpm-detector/actions)
- 📖 [ドキュメント](https://github.com/libraz/bpm-detector)
- 🐛 [Issues](https://github.com/libraz/bpm-detector/issues)
- 💡 [機能リクエスト](https://github.com/libraz/bpm-detector/issues/new?template=feature_request.md)

## インストール

### 方法1: PyPIからインストール（近日公開予定）

```bash
pip install bpm-detector
```

### 方法2: ソースからインストール（現在）

```bash
# リポジトリをクローン
git clone git@github.com:libraz/bpm-detector.git
cd bpm-detector

# 仮想環境を作成（オプションですが推奨）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 開発モードでインストール
pip install -e .
```

### 方法3: ryeを使用（ryeがインストールされている場合）

```bash
# リポジトリをクローン
git clone git@github.com:libraz/bpm-detector.git
cd bpm-detector

# ryeで依存関係をインストール
rye sync
```

## 使用方法

### コマンドラインインターフェース

インストール後、`bpm-detector`コマンドを使用できます：

```bash
# 基本的な使用方法（BPMのみ）
bpm-detector your_audio_file.wav

# キー検出も含める場合
bmp-detector --detect-key your_audio_file.wav

# 複数ファイルの処理
bpm-detector --detect-key *.wav *.mp3

# プログレスバーを表示
bpm-detector --progress --detect-key your_audio_file.wav
```

### Python API

#### 基本解析（高速）
```python
from bpm_detector import AudioAnalyzer

# アナライザーを初期化
analyzer = AudioAnalyzer()

# 基本解析（BPM + キーのみ）- 高速！
results = analyzer.analyze_file('song.wav', detect_key=True, comprehensive=False)

print(f"BPM: {results['basic_info']['bpm']:.1f}")
print(f"キー: {results['basic_info']['key']}")
print(f"長さ: {results['basic_info']['duration']:.1f}秒")
```

#### 包括的解析（詳細）
```python
# 包括的解析 - 全機能！
results = analyzer.analyze_file('song.wav', comprehensive=True)

# 基本情報
basic = results['basic_info']
print(f"BPM: {basic['bpm']:.1f}, キー: {basic['key']}")

# コード進行
chords = results['chord_progression']
print(f"メインコード進行: {' → '.join(chords['main_progression'])}")
print(f"コード複雑度: {chords['chord_complexity']:.1%}")

# 楽曲構造
structure = results['structure']
print(f"楽曲形式: {structure['form']}")
print(f"セクション数: {structure['section_count']}")

# リズム解析
rhythm = results['rhythm']
print(f"拍子: {rhythm['time_signature']}")
print(f"グルーヴ: {rhythm['groove_type']}")

# 制作参考資料を生成
reference_sheet = analyzer.generate_reference_sheet(results)
print(reference_sheet)
```

#### パフォーマンス比較
```python
import time

# 高速解析（0.1-0.7秒）
start = time.time()
basic_results = analyzer.analyze_file('song.wav', comprehensive=False)
print(f"基本解析: {time.time() - start:.2f}秒")

# 包括的解析（2.5-15秒、音声長による）
start = time.time()
full_results = analyzer.analyze_file('song.wav', comprehensive=True)
print(f"包括的解析: {time.time() - start:.2f}秒")
```

### Docker使用方法

Dockerを使用して検出器を実行することもできます：

```bash
# 最新イメージを取得
docker pull ghcr.io/libraz/bpm-detector:latest

# 音声ファイルで実行（音声ディレクトリをマウント）
docker run --rm -v /path/to/your/audio:/workspace ghcr.io/libraz/bpm-detector:latest --detect-key audio.wav

# インタラクティブモード
docker run --rm -it -v /path/to/your/audio:/workspace ghcr.io/libraz/bpm-detector:latest
```

### 開発モード

インストールせずにソースから実行する場合：

```bash
# Pythonモジュールとして実行
python -m bpm_detector.cli your_audio_file.wav

# ryeを使用
rye run python -m bpm_detector.cli your_audio_file.wav

# Dockerイメージをローカルでビルド
docker build -t bpm-detector .
docker run --rm -v $(pwd):/workspace bpm-detector --help
```

## オプション

### コマンドラインオプション
- `--detect-key`: キー検出を有効にする
- `--comprehensive`: 包括的楽曲解析を有効にする（新機能！）
- `--progress`: プログレスバーを表示
- `--sr SR`: サンプリングレート（デフォルト: 22050）
- `--hop HOP`: ホップ長（デフォルト: 128）
- `--min_bpm MIN_BPM`: 最小BPM（デフォルト: 40.0）
- `--max_bpm MAX_BPM`: 最大BPM（デフォルト: 300.0）
- `--start_bpm START_BPM`: 開始BPM（デフォルト: 150.0）

### Python APIオプション
```python
analyzer.analyze_file(
    path='song.wav',
    detect_key=True,           # キー検出を有効
    comprehensive=True,        # 全高度機能を有効
    min_bpm=40.0,             # 最小BPM範囲
    max_bpm=300.0,            # 最大BPM範囲
    start_bpm=150.0,          # 開始BPM推定値
    progress_callback=None     # プログレスコールバック関数
)
```

## 出力例

### 基本解析出力
```
example.wav
  > BPM Candidates Top10
  * 120.00 BPM : 45
    240.00 BPM : 23
     60.00 BPM : 18
    180.00 BPM : 12
    ...
  > Estimated BPM : 120.00 BPM  (conf 78.3%)
  > Estimated Key : C Major  (conf 85.2%)
```

### 包括的解析出力
```
example.wav
  > BPM: 120.0, キー: C Major, 長さ: 180.0秒
  > コード進行: C → Am → F → G (I-vi-IV-V)
  > 構造: イントロ-Aメロ-サビ-Aメロ-サビ-ブリッジ-サビ (ABABCB)
  > リズム: 4/4拍子, ストレートグルーヴ, 中程度のシンコペーション
  > 楽器: ピアノ主体, ギター, ドラム
  > エネルギー: 中レベル, 2分30秒でクライマックス
```

### 参考資料例
```markdown
# 楽曲制作参考資料

## 基本情報
- **テンポ**: 120.0 BPM
- **キー**: C Major
- **拍子**: 4/4
- **長さ**: 180秒

## ハーモニー・コード進行
- **メインコード進行**: C - Am - F - G
- **コード複雑度**: 65.0%
- **ハーモニックリズム**: 2.0 変化/秒

## 制作ノート
- アレンジ密度: 中程度
- プロダクションスタイル: ロックポップ
- ミックス特徴: 明るいミックス, パンチの効いたドラム

## 参考タグ
アップビート, メジャーキー, ピアノ主体, ギター主体, 中エネルギー
```

注意: 実際のCLI出力にはカラーが含まれます：
- ファイル名は明るいシアンで表示
- セクションヘッダー（"> BPM Candidates"）は黄色
- 選択されたBPM候補（*）は緑色
- 最終推定値は明るい緑色（BPM）とマゼンタ（キー）

## パフォーマンス・技術詳細

### パフォーマンスベンチマーク
| 音声長 | 基本解析 | 包括的解析 | 速度差 |
|--------|----------|------------|--------|
| 5秒    | 0.7秒    | 2.5秒      | 3.4倍  |
| 10秒   | 0.1秒    | 4.9秒      | 47倍   |
| 20秒   | 0.2秒    | 9.9秒      | 43倍   |
| 30秒   | 0.3秒    | 15.0秒     | 45倍   |

**推奨**: リアルタイム用途では`comprehensive=False`、詳細分析では`comprehensive=True`を使用。

### BPM検出アルゴリズム
- librosaのテンポ検出機能を使用
- ハーモニッククラスタリングによる候補の統合
- 高速レイヤー（×1.5, ×2）の自動選択

### キー検出アルゴリズム
- クロマ特徴量（Chroma features）の抽出
- Krumhansl-Schmucklerキープロファイルとの相関計算
- 24キー（12メジャー + 12マイナー）からの最適選択

### 高度解析アルゴリズム
- **コード検出**: クロマ特徴量とテンプレートマッチング
- **構造解析**: 自己類似度行列と境界検出
- **リズム解析**: オンセット検出とパターン認識
- **音色解析**: MFCCとスペクトラル特徴抽出
- **メロディー解析**: librosa.pyinによる基本周波数追跡
- **ダイナミクス解析**: RMSエネルギーとスペクトラルエネルギープロファイル

### 包括的解析がオプションである理由
包括的解析は以下の理由でオプション機能として実装されています：

1. **パフォーマンス**: 高度解析により大幅な計算負荷増加（3-45倍遅い）
2. **用途**: DJミキシング、テンポマッチング、基本解析のみが必要な場合が多い
3. **処理時間**: バッチ処理時、詳細機能が不要な場合は高速な基本解析を選択可能
4. **柔軟性**: 要件に応じて速度と機能の完全性のバランスを調整可能

## プロジェクト統計

- **テストカバレッジ**: 100%（54/54テスト成功）
- **対応Python**: 3.12以上
- **Dockerイメージサイズ**: 約1.6GB
- **ビルド時間**: 約4分
- **対応フォーマット**: WAV, MP3, FLAC, M4A, OGG
- **解析機能**: 7つの包括的モジュール + 類似度エンジン

## 依存関係

### コア依存関係
- librosa >= 0.11.0
- soundfile >= 0.13.1
- numpy >= 2.2.6
- tqdm >= 4.67.1
- audioread >= 3.0.1
- colorama >= 0.4.6

### 高度解析依存関係
- scikit-learn >= 1.3.0
- scipy >= 1.11.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- pandas >= 2.0.0

## コントリビューション

コントリビューションを歓迎します！詳細は[コントリビューションガイドライン](CONTRIBUTING.md)をご覧ください。

1. リポジトリをフォーク
2. 機能ブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを開く

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は[LICENSE](LICENSE)をご覧ください。