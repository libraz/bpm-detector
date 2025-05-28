"""Command-line interface for BPM and Key detector."""

import argparse
import os
import math
from typing import Optional

import soundfile as sf
from tqdm import tqdm
from colorama import init, Fore, Style

from .detector import AudioAnalyzer, SR_DEFAULT, HOP_DEFAULT, BIN_WIDTH

# Initialize colorama for cross-platform colored output
init(autoreset=True)


def progress_bar(total_frames: int, sr: int) -> tqdm:
    """Create a progress bar for audio processing."""
    total_seconds = total_frames / sr
    bar = tqdm(total=100, bar_format='{l_bar}{bar}| {n_fmt}%')
    # simplistic four-phase: 0-25 load, 25-50 BPM, 50-75 key, 75-100 finalize
    bar.update(0)
    return bar


def print_results(results: dict, detect_key: bool = False) -> None:
    """Print analysis results with colored output."""
    filename = os.path.basename(results['filename'])
    bpm = results['bpm']
    bpm_conf = results['bpm_confidence']
    candidates = results['bpm_candidates']
    
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{filename}{Style.RESET_ALL}")
    print(f"  {Fore.YELLOW}> BPM Candidates Top10{Style.RESET_ALL}")
    
    for b, h in candidates:
        if math.isclose(b, bpm, abs_tol=BIN_WIDTH/2):
            print(f"  {Fore.GREEN}* {b:6.2f} BPM : {h}{Style.RESET_ALL}")
        else:
            print(f"    {b:6.2f} BPM : {h}")
    
    print(f"  {Fore.GREEN}{Style.BRIGHT}> Estimated BPM : {bpm:.2f} BPM  (conf {bpm_conf:.1f}%){Style.RESET_ALL}")
    
    if detect_key and 'key' in results:
        key = results['key']
        key_conf = results['key_confidence']
        print(f"  {Fore.MAGENTA}{Style.BRIGHT}> Estimated Key : {key}  (conf {key_conf:.1f}%){Style.RESET_ALL}")
    
    print()


def analyze_file(path: str, analyzer: AudioAnalyzer, args: argparse.Namespace) -> None:
    """Analyze a single audio file."""
    info = sf.info(path)
    bar = progress_bar(info.frames, args.sr) if args.progress else None
    
    def progress_callback(increment: int) -> None:
        if bar:
            bar.update(increment)
    
    try:
        results = analyzer.analyze_file(
            path=path,
            detect_key=args.detect_key,
            min_bpm=args.min_bpm,
            max_bpm=args.max_bpm,
            start_bpm=args.start_bpm,
            progress_callback=progress_callback
        )
        
        if bar:
            bar.close()
        
        print_results(results, args.detect_key)
        
    except Exception as e:
        if bar:
            bar.close()
        print(f"{Fore.RED}Error processing {path}: {e}{Style.RESET_ALL}")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description='Smart BPM and Key detector')
    parser.add_argument('files', nargs='+', help='Audio file paths')
    parser.add_argument('--sr', type=int, default=SR_DEFAULT, help='Sample rate')
    parser.add_argument('--hop', type=int, default=HOP_DEFAULT, help='Hop length')
    parser.add_argument('--min_bpm', type=float, default=40.0, help='Minimum BPM')
    parser.add_argument('--max_bpm', type=float, default=300.0, help='Maximum BPM')
    parser.add_argument('--start_bpm', type=float, default=150.0, help='Starting BPM')
    parser.add_argument('--progress', action='store_true', help='Show progress bar')
    parser.add_argument('--detect-key', action='store_true', help='Enable key detection')
    
    args = parser.parse_args()
    
    # Auto-enable progress for multiple files
    if len(args.files) > 1 and not args.progress:
        args.progress = True
    
    # Initialize analyzer
    analyzer = AudioAnalyzer(sr=args.sr, hop_length=args.hop)
    
    # Process files
    for filepath in args.files:
        if not os.path.exists(filepath):
            print(f"{Fore.RED}File not found: {filepath}{Style.RESET_ALL}")
            continue
        
        analyze_file(filepath, analyzer, args)


if __name__ == '__main__':
    main()