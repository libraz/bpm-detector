# BPM & Key Detector

[![codecov](https://codecov.io/gh/libraz/bpm-detector/branch/main/graph/badge.svg)](https://codecov.io/gh/libraz/bpm-detector)
[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker Pulls](https://img.shields.io/docker/pulls/ghcr.io/libraz/bpm-detector)](https://github.com/libraz/bpm-detector/pkgs/container/bpm-detector)

音楽ファイルのBPM（テンポ）とキー（調）を自動検出するPythonツールです。

## 機能

- **BPM検出**: 高精度なテンポ検出アルゴリズム
  - 自動的な高速/低速レイヤー選択
  - ハーモニッククラスタリング
  - 信頼度スコア付き
- **キー検出**: 音楽理論に基づいたキー検出
  - Krumhansl-Schmucklerキープロファイル使用
  - メジャー・マイナー両方に対応
  - クロマ特徴量ベースの解析

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

プログラムから検出器を使用することもできます：

```python
from bpm_detector import AudioAnalyzer

# アナライザーを初期化
analyzer = AudioAnalyzer()

# ファイルを解析
results = analyzer.analyze_file('your_audio_file.wav', detect_key=True)

print(f"BPM: {results['bpm']:.2f}")
print(f"キー: {results['key']}")
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

- `--detect-key`: キー検出を有効にする
- `--progress`: プログレスバーを表示
- `--sr SR`: サンプリングレート（デフォルト: 22050）
- `--hop HOP`: ホップ長（デフォルト: 128）
- `--min_bpm MIN_BPM`: 最小BPM（デフォルト: 40.0）
- `--max_bpm MAX_BPM`: 最大BPM（デフォルト: 300.0）
- `--start_bpm START_BPM`: 開始BPM（デフォルト: 150.0）

## 出力例

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

注意: 実際の出力にはカラーが含まれます：
- ファイル名は明るいシアンで表示
- セクションヘッダー（"> BPM Candidates"）は黄色
- 選択されたBPM候補（*）は緑色
- 最終推定値は明るい緑色（BPM）とマゼンタ（キー）

## 技術詳細

### BPM検出アルゴリズム
- librosaのテンポ検出機能を使用
- ハーモニッククラスタリングによる候補の統合
- 高速レイヤー（×1.5, ×2）の自動選択

### キー検出アルゴリズム
- クロマ特徴量（Chroma features）の抽出
- Krumhansl-Schmucklerキープロファイルとの相関計算
- 24キー（12メジャー + 12マイナー）からの最適選択

### キー検出がオプションである理由
キー検出は以下の理由でオプション機能（`--detect-key`）として実装されています：

1. **パフォーマンス**: クロマ特徴量の抽出と相関計算により計算負荷が増加します
2. **用途**: DJミキシング、テンポマッチング、リズム解析など、BPM検出のみが必要な場合が多くあります
3. **処理時間**: 大量のファイルをバッチ処理する際、キー情報が不要な場合は高速なBPMのみの解析を選択できます
4. **柔軟性**: ユーザーの具体的な要件に応じて、速度と機能の完全性のバランスを取ることができます

## プロジェクト統計

- **テストカバレッジ**: 90%以上（171行中154行をカバー）
- **対応Python**: 3.12
- **Dockerイメージサイズ**: 約1.6GB
- **ビルド時間**: 約4分
- **対応フォーマット**: WAV, MP3, FLAC, M4A, OGG

## 依存関係

- librosa >= 0.11.0
- soundfile >= 0.13.1
- numpy >= 2.2.6
- tqdm >= 4.67.1
- audioread >= 3.0.1
- colorama >= 0.4.6

## コントリビューション

コントリビューションを歓迎します！詳細は[コントリビューションガイドライン](CONTRIBUTING.md)をご覧ください。

1. リポジトリをフォーク
2. 機能ブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを開く

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は[LICENSE](LICENSE)をご覧ください。