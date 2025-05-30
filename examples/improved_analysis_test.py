#!/usr/bin/env python3
"""
改善された分析機能のテスト
レビューで指摘された問題点の修正を確認
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from bpm_detector.music_analyzer import AudioAnalyzer
import time

def test_improved_analysis():
    """改善された分析機能をテスト"""
    
    # 分析対象ファイル
    audio_file = "examples/icecream.mp3"
    
    if not os.path.exists(audio_file):
        print(f"❌ テストファイルが見つかりません: {audio_file}")
        return
    
    print("🎧 改善された分析機能のテスト")
    print("=" * 60)
    print(f"📁 ファイル: {audio_file}")
    print()
    
    # 分析器を初期化
    analyzer = AudioAnalyzer()
    
    # 分析実行
    print("🔍 分析開始...")
    start_time = time.time()
    
    try:
        results = analyzer.analyze_file(audio_file)
        analysis_time = time.time() - start_time
        
        print(f"✅ 分析完了 ({analysis_time:.2f}秒)")
        print()
        
        # Key Detection の改善確認
        print("🎵 Key Detection の改善結果:")
        print("-" * 40)
        key_info = results.get('melody_harmony', {}).get('key_detection', {})
        print(f"Key: {key_info.get('key', 'None')}")
        print(f"Mode: {key_info.get('mode', 'Unknown')}")
        print(f"Confidence: {key_info.get('confidence', 0.0):.3f}")
        print(f"Key Strength: {key_info.get('key_strength', 0.0):.3f}")
        print()
        
        # 構造解析の改善確認
        print("🏗️ 構造解析の改善結果:")
        print("-" * 40)
        structure = results.get('structure', {})
        sections = structure.get('sections', [])
        
        print(f"総セクション数: {len(sections)}")
        print(f"Form: {structure.get('form', 'Unknown')}")
        print()
        
        # セクションタイプの分布を確認
        section_types = {}
        for section in sections:
            section_type = section.get('type', 'unknown')
            section_types[section_type] = section_types.get(section_type, 0) + 1
        
        print("セクションタイプ分布:")
        for section_type, count in sorted(section_types.items()):
            print(f"  {section_type}: {count}個")
        
        # Outro の数を特に確認
        outro_count = section_types.get('outro', 0)
        print(f"\n🎯 Outro セクション数: {outro_count}")
        if outro_count <= 2:
            print("✅ Outro過多問題が改善されました！")
        else:
            print("⚠️ まだOutroが多すぎます")
        
        print()
        
        # 詳細なセクション情報
        print("📋 詳細セクション情報:")
        print("-" * 40)
        for i, section in enumerate(sections):
            print(f"{i+1:2d}. {section.get('type', 'unknown'):12s} "
                  f"{section.get('start_time', 0):6.1f}s - {section.get('end_time', 0):6.1f}s "
                  f"(Energy: {section.get('energy_level', 0):.2f}, "
                  f"Complexity: {section.get('complexity', 0):.2f})")
        
        print()
        
        # Harmony の改善確認
        print("🎼 Harmony 分析の改善結果:")
        print("-" * 40)
        consonance = results.get('melody_harmony', {}).get('consonance', {})
        print(f"Consonance Level: {consonance.get('consonance_level', 0):.1%}")
        print(f"Dissonance Level: {consonance.get('dissonance_level', 0):.1%}")
        
        harmony_complexity = results.get('melody_harmony', {}).get('harmony_complexity', {})
        print(f"Harmonic Complexity: {harmony_complexity.get('harmonic_complexity', 0):.1%}")
        
        print()
        print("🎯 改善点の検証:")
        print("-" * 40)
        
        # Key Detection の改善確認
        if key_info.get('key') != 'None':
            print("✅ Key Detection: 調が検出されました")
        else:
            print("❌ Key Detection: まだ調が検出されていません")
        
        # Outro過多問題の改善確認
        if outro_count <= 2:
            print("✅ Section Classification: Outro過多問題が解決")
        else:
            print("❌ Section Classification: Outro過多問題が残存")
        
        # Consonance の妥当性確認
        consonance_level = consonance.get('consonance_level', 0)
        if 0.7 <= consonance_level <= 0.95:
            print("✅ Consonance Analysis: ポップソングとして妥当な値")
        else:
            print(f"⚠️ Consonance Analysis: 値が範囲外 ({consonance_level:.1%})")
        
    except Exception as e:
        print(f"❌ 分析エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_improved_analysis()