"""Command-line interface for BPM and Key detector."""

import argparse
import math
import os
import warnings
from typing import Optional

import soundfile as sf
from colorama import Fore, Style, init
from tqdm import tqdm

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

from .music_analyzer import BIN_WIDTH, HOP_DEFAULT, SR_DEFAULT, AudioAnalyzer

# Initialize colorama for cross-platform colored output
init(autoreset=True)


def progress_bar(total_frames: int, sr: int) -> tqdm:
    """Create a progress bar for audio processing."""
    total_seconds = total_frames / sr
    bar = tqdm(total=100, bar_format="{l_bar}{bar}| {n_fmt}%")
    # simplistic four-phase: 0-25 load, 25-50 BPM, 50-75 key, 75-100 finalize
    bar.update(0)
    return bar


def print_results(results: dict, detect_key: bool = False, comprehensive: bool = False) -> None:
    """Print analysis results with colored output."""
    # Handle new result format with basic_info
    if "basic_info" in results:
        basic_info = results["basic_info"]
        filename = os.path.basename(basic_info["filename"])
        bpm = basic_info["bpm"]
        bpm_conf = basic_info["bpm_confidence"]
        candidates = basic_info["bpm_candidates"]
    else:
        # Fallback to old format
        filename = os.path.basename(results["filename"])
        bpm = results["bpm"]
        bpm_conf = results["bpm_confidence"]
        candidates = results["bpm_candidates"]

    print(f"\n{Fore.CYAN}{Style.BRIGHT}{filename}{Style.RESET_ALL}")
    
    if comprehensive and "basic_info" in results:
        # Show comprehensive summary first
        duration = basic_info.get("duration", 0)
        print(f"  {Fore.YELLOW}> Duration: {duration:.1f}s, BPM: {bpm:.1f}, Key: {basic_info.get('key', 'Unknown')}{Style.RESET_ALL}")
        
        # Show additional analysis if available
        if "chord_progression" in results:
            chords = results["chord_progression"]
            main_prog = chords.get("main_progression", [])
            if main_prog:
                print(f"  {Fore.BLUE}> Chord Progression: {' → '.join(main_prog[:4])}{Style.RESET_ALL}")
        
        if "structure" in results:
            structure = results["structure"]
            form = structure.get("form", "Unknown")
            sections = structure.get("section_count", 0)
            print(f"  {Fore.BLUE}> Structure: {form} ({sections} sections){Style.RESET_ALL}")
        
        if "rhythm" in results:
            rhythm = results["rhythm"]
            time_sig = rhythm.get("time_signature", "4/4")
            groove = rhythm.get("groove_type", "straight")
            print(f"  {Fore.BLUE}> Rhythm: {time_sig} time, {groove} groove{Style.RESET_ALL}")
        
        print()  # Extra line for comprehensive results
    
    print(f"  {Fore.YELLOW}> BPM Candidates Top10{Style.RESET_ALL}")

    for b, h in candidates:
        if math.isclose(b, bpm, abs_tol=BIN_WIDTH / 2):
            print(f"  {Fore.GREEN}* {b:6.2f} BPM : {h}{Style.RESET_ALL}")
        else:
            print(f"    {b:6.2f} BPM : {h}")

    print(
        f"  {Fore.GREEN}{Style.BRIGHT}> Estimated BPM : {bpm:.2f} BPM  (conf {bpm_conf:.1f}%){Style.RESET_ALL}"
    )

    if detect_key:
        # Handle new result format
        if "basic_info" in results:
            key = basic_info.get("key")
            key_conf = basic_info.get("key_confidence", 0.0)
        else:
            key = results.get("key")
            key_conf = results.get("key_confidence", 0.0)
            
        if key:
            print(
                f"  {Fore.MAGENTA}{Style.BRIGHT}> Estimated Key : {key}  (conf {key_conf:.1f}%){Style.RESET_ALL}"
            )

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
            comprehensive=args.comprehensive,
            min_bpm=args.min_bpm,
            max_bpm=args.max_bpm,
            start_bpm=args.start_bpm,
            progress_callback=progress_callback,
        )

        if bar:
            bar.close()

        print_results(results, args.detect_key, args.comprehensive)

    except Exception as e:
        if bar:
            bar.close()
        print(f"{Fore.RED}Error processing {path}: {e}{Style.RESET_ALL}")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Smart BPM and Key detector")
    parser.add_argument("files", nargs="+", help="Audio file paths")
    parser.add_argument("--sr", type=int, default=SR_DEFAULT, help="Sample rate")
    parser.add_argument("--hop", type=int, default=HOP_DEFAULT, help="Hop length")
    parser.add_argument("--min_bpm", type=float, default=40.0, help="Minimum BPM")
    parser.add_argument("--max_bpm", type=float, default=300.0, help="Maximum BPM")
    parser.add_argument("--start_bpm", type=float, default=150.0, help="Starting BPM")
    parser.add_argument("--progress", action="store_true", help="Show progress bar")
    parser.add_argument(
        "--detect-key", action="store_true", help="Enable key detection"
    )
    parser.add_argument(
        "--comprehensive", action="store_true", help="Enable comprehensive music analysis"
    )

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


if __name__ == "__main__":
    main()
