# BPM & Key Detector

[![CI](https://img.shields.io/github/actions/workflow/status/libraz/bpm-detector/ci.yml?branch=main&label=CI)](https://github.com/libraz/bpm-detector/actions)
[![codecov](https://codecov.io/gh/libraz/bpm-detector/branch/main/graph/badge.svg)](https://codecov.io/gh/libraz/bpm-detector)
[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/libraz/bpm-detector/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)

音楽ファイルのBPM（テンポ）・キー（調）検出から包括的な楽曲解析まで対応した音楽制作支援ツールです。

## 関連プロジェクト: libsonare

同じ作者が **[libsonare](https://github.com/libraz/libsonare)** という依存ゼロのオーディオ DSP ツールキット（C++ / Python / WASM、Apache-2.0）も開発しています。本プロジェクトのアルゴリズムを C++ に移植し、放送品質のマスタリング・ミキシング・編集 DSP まで含めた拡張版で、ヘッドレス DAW として開発中です。

🎵 **[音楽解析デモを試す](https://libsonare.libraz.net/music-analysis.html)** — BPM・キー・コード・構造解析をブラウザ上で完結して実行できます（インストール不要）。([デモ一覧](https://libsonare.libraz.net/demos.html))

本プロジェクトは、軽量で MIT ライセンスの Python 専用版として BPM・キー検出を安定版として引き続き利用できます。新規開発は libsonare に注力しています。

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
  - 低信頼度時は `key` が `None` になり、暫定候補（例: `Unknown(E♭?)`）は `key_detection_details` に残る

### 包括的楽曲解析
- **コード進行解析**: 自動和音検出とハーモニー解析
  - コード進行の識別（C-Am-F-G）
  - 機能和声分析（I-vi-IV-V）
  - 転調検出
  - コード複雑度スコア
  - 全12ルートのメジャー・マイナー・7th・sus4コードを検出
- **楽曲構造解析**: 自動セクション検出と楽曲形式分析
  - セクション境界（イントロ、Aメロ、サビ、ブリッジ）
  - 楽曲形式の識別（ABABCB）
  - 反復パターン検出
  - 構造複雑度分析
  - 落ちサビ・ラスサビ検出
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

### 高度な構造解析
- **セクション分類**: インテリジェントな楽曲セクション検出
  - イントロ、Aメロ、サビ、ブリッジ、アウトロの自動識別
  - 音響特徴を使用したコンテキスト認識分類
  - ボーカル存在検出と話し言葉識別
  - ダイナミックセクションのエネルギー構築検出
- **境界検出**: 精密な構造境界識別
  - セクション境界の自己類似度行列解析
  - 音楽的精度のためのビート整列境界スナップ
  - 反復パターン検出と解析
  - ノベルティベース境界検出アルゴリズム
- **セクション処理**: 高度な後処理と精緻化
  - 持続時間と特徴に基づくスマートセクション統合
  - 楽器サブタイプ分類のためのスペクトラル解析
  - 文字記法による楽曲形式解析（ABABCB）
  - 構造複雑度スコアリング

### 並列処理・パフォーマンス
- **スマート並列処理**: CPUベースの自動最適化
  - システム能力の自動検出（CPUコア、メモリ、負荷）
  - システムパフォーマンスに基づく適応的ワーカー数
  - 動的負荷監視と調整
  - シーケンシャル処理への優雅なフォールバック
- **プログレス追跡**: リアルタイムプログレス監視
  - 並列タスクの階層プログレス表示
  - 各解析モジュールの詳細プログレス
  - 時間推定とパフォーマンスメトリクス
  - タスクステータス付きインタラクティブプログレスバー
- **パフォーマンス最適化**: 包括解析を複数コアに分散
  - 実測値と再現手順は[パフォーマンスベンチマーク](#パフォーマンスベンチマーク)を参照
  - インテリジェントメモリ管理
  - ワークロードに基づくプロセス対スレッドプール選択

### 選択的解析
- **柔軟な解析オプション**: 必要な情報だけを解析
  - `--rhythm`: リズムと拍子のみを解析
  - `--chords`: コード進行のみを解析
  - `--structure`: 楽曲構造のみを解析
  - `--timbre`: 音色と楽器のみを解析
  - `--melody`: メロディとハーモニーのみを解析
  - `--dynamics`: ダイナミクスのみを解析
  - オプションを組み合わせてカスタム解析パイプラインを構築
- **パフォーマンスの利点**: 不要な計算をスキップして高速化
  - ターゲット解析で処理時間を大幅に削減
  - メモリ使用量の低減
  - 特定の機能のみが必要なバッチ処理に最適

## クイックリンク

- 📦 [PyPIパッケージ](https://pypi.org/project/bpm-detector/) (近日公開予定)
- 🐳 [Dockerイメージ](https://github.com/libraz/bpm-detector/pkgs/container/bpm-detector)
- 📊 [テストカバレッジ](https://codecov.io/gh/libraz/bpm-detector)
- 🔧 [CI/CDステータス](https://github.com/libraz/bpm-detector/actions)
- 📖 [ドキュメント](https://github.com/libraz/bpm-detector)
- 🐛 [Issues](https://github.com/libraz/bpm-detector/issues)
- 💡 [機能リクエスト](https://github.com/libraz/bpm-detector/issues/new)

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
bpm-detector --detect-key your_audio_file.wav

# 複数ファイルの処理
bpm-detector --detect-key *.wav *.mp3

# 進捗表示を抑制（-q）
bpm-detector --quiet --detect-key your_audio_file.wav

# 選択的解析 - 必要な情報だけを高速に解析
bpm-detector --rhythm your_audio_file.wav  # BPM + 拍子のみ
bpm-detector --detect-key --rhythm your_audio_file.wav  # BPM + キー + リズム
bpm-detector --melody --timbre your_audio_file.wav  # BPM + メロディ + 楽器
bpm-detector --rhythm --chords --structure your_audio_file.wav  # 複数の解析

# 利用可能な選択的解析オプション：
#   --rhythm    : リズムと拍子を解析
#   --chords    : コード進行を解析
#   --structure : 楽曲構造を解析
#   --timbre    : 音色と楽器を解析
#   --melody    : メロディとハーモニーを解析
#   --dynamics  : ダイナミクスを解析

# 包括的解析（全解析。キー検出には別途 --detect-key が必要）
bpm-detector --comprehensive your_audio_file.wav

# 並列処理
bpm-detector --comprehensive your_audio_file.wav  # 自動並列処理がデフォルトで有効
bpm-detector --comprehensive --max-workers 4 your_audio_file.wav  # 手動ワーカー数設定
bpm-detector --comprehensive --no-parallel your_audio_file.wav  # 並列処理を無効化
bpm-detector --show-system-info  # システム能力と並列設定を表示
bpm-detector --comprehensive --detailed-progress your_audio_file.wav  # 詳細プログレス追跡
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

#### 選択的解析（高速＆効率的）
```python
from bpm_detector import AudioAnalyzer

analyzer = AudioAnalyzer()

# リズムと拍子のみを解析（最速）
results = analyzer.analyze_file(
    'song.wav',
    detect_key=False,
    comprehensive=False,
    analyze_rhythm=True
)
print(f"BPM: {results['basic_info']['bpm']:.1f}")
print(f"拍子: {results['rhythm']['time_signature']}")
print(f"グルーヴ: {results['rhythm']['groove_type']}")

# 複数の選択的解析
results = analyzer.analyze_file(
    'song.wav',
    detect_key=True,
    comprehensive=False,
    analyze_rhythm=True,
    analyze_melody=True,
    analyze_timbre=True
)

# 選択した解析結果にアクセス
print(f"キー: {results['basic_info']['key']}")
print(f"拍子: {results['rhythm']['time_signature']}")
print(f"楽器: {results['timbre']['dominant_instruments']}")
print(f"ボーカルレンジ: {results['melody_harmony']['melodic_range']}")

# 利用可能な選択的解析パラメータ：
#   analyze_rhythm=True    : リズムと拍子
#   analyze_chords=True    : コード進行
#   analyze_structure=True : 楽曲構造
#   analyze_timbre=True    : 音色と楽器
#   analyze_melody=True    : メロディとハーモニー
#   analyze_dynamics=True  : ダイナミクス
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

#### スマート並列処理
```python
from bpm_detector import SmartParallelAudioAnalyzer

# 自動最適化付きスマート並列アナライザー
analyzer = SmartParallelAudioAnalyzer(auto_parallel=True)

# プログレス追跡付き単一ファイル
def progress_callback(progress, message):
    print(f"プログレス: {progress:.1f}% - {message}")

results = analyzer.analyze_file(
    'song.wav',
    comprehensive=True,
    progress_callback=progress_callback
)

# 並列処理による複数ファイル（{パス: 結果} の辞書を返す）
files = ['song1.wav', 'song2.wav', 'song3.wav']
batch_results = analyzer.analyze_files(files, comprehensive=True)

# 手動設定
analyzer = SmartParallelAudioAnalyzer(
    auto_parallel=True,
    max_workers=4  # 自動ワーカー数をオーバーライド
)

# システム設定確認
from bpm_detector import AutoParallelConfig
config = AutoParallelConfig.get_optimal_config()
print(f"並列処理有効: {config.enable_parallel}")
print(f"最大ワーカー数: {config.max_workers}")
print(f"戦略: {config.strategy.value}")
```

#### パフォーマンス比較
```python
import time
from bpm_detector import AudioAnalyzer, SmartParallelAudioAnalyzer

# 従来のアナライザー
traditional = AudioAnalyzer()
parallel = SmartParallelAudioAnalyzer(auto_parallel=True)

# 基本解析
start = time.time()
basic_results = traditional.analyze_file('song.wav', comprehensive=False)
print(f"基本解析: {time.time() - start:.2f}秒")

# シーケンシャル包括的解析
start = time.time()
sequential_results = traditional.analyze_file('song.wav', comprehensive=True)
sequential_time = time.time() - start
print(f"シーケンシャル包括的解析: {sequential_time:.2f}秒")

# 並列包括的解析
start = time.time()
parallel_results = parallel.analyze_file('song.wav', comprehensive=True)
parallel_time = time.time() - start
print(f"並列包括的解析: {parallel_time:.2f}秒")
print(f"速度向上: {sequential_time/parallel_time:.2f}倍")
```

実測値は[パフォーマンスベンチマーク](#パフォーマンスベンチマーク)を参照してください。

### Docker使用方法

Dockerを使用して検出器を実行することもできます：

```bash
# 最新イメージを取得
docker pull ghcr.io/libraz/bpm-detector:latest

# 音声ファイルで実行（音声ディレクトリをマウント）
docker run --rm -v /path/to/your/audio:/workspace ghcr.io/libraz/bpm-detector:latest --detect-key audio.wav

# CLI のヘルプを表示（デフォルトのコマンド）
docker run --rm ghcr.io/libraz/bpm-detector:latest
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

#### 基本オプション
- `--detect-key`: キー検出を有効にする
- `--detect-modulation`: 時間経過に伴う転調を検出して表示（`--detect-key` が必要）
- `--comprehensive`: 包括的楽曲解析を有効にする
- `--quiet`, `-q`: 進捗表示を抑制
- `--sr SR`: サンプリングレート（デフォルト: 22050）
- `--hop HOP`: ホップ長（デフォルト: 128）
- `--min_bpm MIN_BPM`: 最小BPM（デフォルト: 40.0）
- `--max_bpm MAX_BPM`: 最大BPM（デフォルト: 300.0）
- `--start_bpm START_BPM`: 開始BPM（デフォルト: 150.0）

#### 選択的解析オプション
- `--rhythm`, `--chords`, `--structure`, `--timbre`, `--melody`, `--dynamics`: 指定した解析だけを実行（組み合わせ可。`--comprehensive` はすべてを有効にする）

#### 並列処理オプション
- `--auto-parallel`: 自動並列最適化を有効にする（デフォルト: 有効）
- `--no-parallel`: 並列処理を無効にする
- `--max-workers N`: 自動ワーカー数をオーバーライド
- `--detailed-progress`: 各解析タスクの詳細プログレスを表示
- `--show-system-info`: システム情報と並列設定を表示

### Python APIオプション
```python
analyzer.analyze_file(
    path='song.wav',
    detect_key=True,           # キー検出を有効（デフォルト: True）
    detect_modulation=False,   # 転調を results['modulation_analysis'] に格納（detect_key が必要）
    comprehensive=True,        # 全高度機能を有効（デフォルト: True）
    min_bpm=40.0,              # 最小BPM範囲
    max_bpm=300.0,             # 最大BPM範囲
    start_bpm=150.0,           # 開始BPM推定値
    progress_callback=None,    # プログレスコールバック関数
    analyze_rhythm=False,      # 選択的解析（comprehensive=False のときに使用）
    analyze_chords=False,
    analyze_structure=False,
    analyze_timbre=False,
    analyze_melody=False,
    analyze_dynamics=False,
)
```

API では `detect_key` と `comprehensive` のデフォルトが `True` です（CLI ではどちらも指定時のみ有効）。BPM とキーだけを素早く求める場合は `comprehensive=False` を渡してください。

## 出力例

### 基本解析出力
```
song.mp3
  > Estimated BPM : 132.50 BPM  (conf 98.4%)
  > Estimated Key : F# Major  (conf 62.4%)
```

### 包括的解析出力
```
song.mp3
  > Duration: 262.1s, BPM: 132.5, Key: F# Major
  > Chord Progression: F# → C# → F# → D#m
  > Structure: IARBARBARBCAC (13 sections)
  > Section Details (13 sections):
    1. Intro (00:00, 3bars, mid E, mid C)
    2. Verse(A-melo) (00:05, 15bars, mid E, mid C)
    3. Pre_Chorus(B-melo) (00:32, 15bars, mid E, mid C)
    4. Chorus(Sabi) (00:59, 9bars, mid E, mid C)
    ...
  > Rhythm: 4/4 time, straight groove
  > Instruments: vocals, piano, guitar
  > Timbre: Brightness 0.6, Warmth 1.0
  > Melody: 64.9% coverage, 4.4 octave range
  > Full Range: D#2 - G#6 (Tenor)
  > Vocal Range: C#3 - G#5 (Tenor)
  > Harmony: 77.3% consonance, 68.4% complexity
  > Dynamics: 80.0dB range, 44.5% variation

  > Estimated BPM : 132.50 BPM  (conf 98.4%)
  > Estimated Key : F# Major  (conf 62.4%)
```

### 参考資料例（抜粋）
出力は英語です。

```markdown
# Music Production Reference Sheet

## Basic Information
- **Tempo**: 120.0 BPM
- **Key**: C Major
- **Time Signature**: 4/4
- **Duration**: 180 seconds

## Song Structure
- **Section Count**: 7
- **Structural Complexity**: 0.6
- **Repetition Ratio**: 57.1%

## Harmony & Chord Progression
- **Main Chord Progression**: C - Am - F - G
- **Chord Complexity**: 65.0%
- **Harmonic Rhythm**: 2.0 changes/sec

...

## Production Notes
- Arrangement Density: medium
- Production Style: rock_pop
- Mix Characteristics: bright_mix, punchy_drums

## Reference Tags
upbeat, major-key, piano-driven, guitar-driven, mid-energy
```

注意: 実際のCLI出力にはカラーが含まれます：
- ファイル名は明るいシアンで表示
- 各解析の要約行はカテゴリごとに色分け（青、マゼンタ、シアン、黄色）
- 最終推定値は明るい緑色（BPM）とマゼンタ（キー）

## パフォーマンス・技術詳細

### パフォーマンスベンチマーク

#### シーケンシャル対並列処理

計測環境は macOS 26.6（arm64、論理18コア）、Python 3.12.9。ハーネスが生成する
合成コード進行のクリップを対象に、ウォームアップ後に1回ずつ計測しているため、
librosa の初回 JIT コンパイル時間は含みません。

| 音声長 | 基本解析 | シーケンシャル包括的解析 | 並列包括的解析 | 並列速度向上 |
|--------|----------|-------------------------|----------------|--------------|
| 5秒    | 0.3秒    | 3.3秒                   | 2.6秒          | 1.3倍        |
| 10秒   | 0.7秒    | 6.8秒                   | 5.1秒          | 1.3倍        |
| 20秒   | 1.3秒    | 13.3秒                  | 10.5秒         | 1.3倍        |
| 30秒   | 1.9秒    | 20.0秒                  | 15.6秒         | 1.3倍        |

解析時間はクリップ長に比例します。並列化の効きはコア数ではなく分割できない
モジュールで頭打ちになるため、数コアを超えて増やしてもほとんど縮みません。
手元で再現するには `python examples/benchmark_table.py` を実行してください。

#### 自動並列化の設定
`AutoParallelConfig` は論理コア数と現在の負荷から戦略を選びます：

| 論理コア数 | 戦略 | ワーカー数 | プール |
|------------|------|------------|--------|
| 8以上      | 積極的並列 | コア数 − 2 | プロセス |
| 4-7        | バランス並列（高負荷時は2ワーカーの保守的並列） | 最大8 | スレッド |
| 2-3        | 保守的並列（高負荷時・メモリ不足時はシーケンシャル） | 最大3 | スレッド |
| 1          | シーケンシャルのみ | 1 | — |

実際に選ばれる設定は `bpm-detector --show-system-info` で確認できます。

**推奨事項**:
- リアルタイムアプリケーションには`comprehensive=False`を使用
- バッチ処理と詳細分析には`SmartParallelAudioAnalyzer`を使用
- システム固有のチューニングは自動並列最適化に任せる

### BPM検出アルゴリズム
- librosaのテンポ検出機能を使用
- ハーモニッククラスタリングによる候補の統合
- 高速レイヤー（×1.5, ×2）の自動選択

### キー検出アルゴリズム
- クロマ特徴量（Chroma features）の抽出
- Krumhansl-Schmucklerキープロファイルとの相関計算
- 24キー（12メジャー + 12マイナー）からの最適選択

### 高度解析アルゴリズム
- **コード検出**: クロマ特徴量とハーモニッククラスタリングによるテンプレートマッチング
- **構造解析**: ノベルティベース境界検出による自己類似度行列
- **セクション分類**: エネルギー、スペクトラル、ボーカル特徴を使用したコンテキスト認識分類
- **境界検出**: 反復解析付きビート整列境界スナップ
- **セクション処理**: スマート統合とスペクトラルベース精緻化
- **リズム解析**: 拍子とグルーヴ分類によるオンセット検出
- **音色解析**: MFCC、スペクトラルコントラスト、楽器分類
- **メロディー解析**: ピッチ安定性解析による基本周波数追跡
- **ハーモニー解析**: 協和・不協和度評価とハーモニー複雑度
- **ダイナミクス解析**: クライマックス検出付きRMSエネルギープロファイル
- **類似度エンジン**: 多次元特徴ベクトル生成と比較
- **並列処理**: 動的負荷分散によるCPUベース適応最適化

### 包括的解析がオプションである理由
包括的解析は以下の理由でオプション機能として実装されています：

1. **パフォーマンス**: 高度解析により計算負荷が増加（上表では基本解析のおよそ10倍）
2. **用途**: DJミキシング、テンポマッチング、基本解析のみが必要な場合が多い
3. **処理時間**: バッチ処理時、詳細機能が不要な場合は高速な基本解析を選択可能
4. **柔軟性**: 要件に応じて速度と機能の完全性のバランスを調整可能

## プロジェクト統計

- **テスト**: 全解析モジュールにわたる400以上のテスト
- **対応Python**: 3.12以上（CI は 3.12 と 3.14 で実行）
- **対応フォーマット**: WAV, MP3, FLAC, OGG は libsndfile で読み込み。M4A などその他の形式には FFmpeg が必要（Docker イメージには同梱）

## 依存関係

Python 3.12 以上が必要です。実行時の依存とその最小バージョンは [`pyproject.toml`](pyproject.toml) に、検証済みの組み合わせは `requirements.lock` に記載しています。

## コントリビューション

コントリビューションを歓迎します！詳細は[コントリビューションガイドライン](CONTRIBUTING.md)をご覧ください。

1. リポジトリをフォーク
2. 機能ブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを開く

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は[LICENSE](LICENSE)をご覧ください。
