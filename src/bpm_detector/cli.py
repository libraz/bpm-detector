"""Command-line interface for BPM and Key detector."""

import argparse
import math
import os
import signal
import sys
import warnings
import psutil
from typing import Optional
from multiprocessing import cpu_count

import soundfile as sf
from colorama import Fore, Style, init
from tqdm import tqdm

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

from .music_analyzer import BIN_WIDTH, HOP_DEFAULT, SR_DEFAULT, AudioAnalyzer
from .parallel_analyzer import SmartParallelAudioAnalyzer
from .auto_parallel import AutoParallelConfig
from .progress_manager import create_progress_display

# Initialize colorama for cross-platform colored output
init(autoreset=True)

# Global variables for signal handling
current_analyzer = None
interrupted = False


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global interrupted, current_analyzer
    interrupted = True
    
    print(f"\n{Fore.YELLOW}⚠️  Interrupted by user. Cleaning up...{Style.RESET_ALL}")
    
    # Stop system monitor if analyzer exists
    if current_analyzer:
        try:
            if hasattr(current_analyzer, 'cleanup'):
                current_analyzer.cleanup()
            elif hasattr(current_analyzer, 'system_monitor'):
                current_analyzer.system_monitor.stop_monitoring()
        except Exception:
            pass
    
    print(f"{Fore.GREEN}✅ Cleanup completed. Exiting gracefully.{Style.RESET_ALL}")
    sys.exit(0)


# Register signal handler
signal.signal(signal.SIGINT, signal_handler)


def progress_bar(total_frames: int, sr: int) -> tqdm:
    """Create a progress bar for audio processing."""
    total_seconds = total_frames / sr
    bar = tqdm(total=100, bar_format="{l_bar}{bar}| {n_fmt}%")
    # simplistic four-phase: 0-25 load, 25-50 BPM, 50-75 key, 75-100 finalize
    bar.update(0)
    return bar


def show_system_info():
    """Display system information and parallel configuration."""
    config = AutoParallelConfig.get_optimal_config()
    
    print(f"{Fore.CYAN}🖥️  System Information{Style.RESET_ALL}")
    print(f"   CPU Cores: {cpu_count()} logical, {psutil.cpu_count(logical=False)} physical")
    print(f"   Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB total, {psutil.virtual_memory().available / (1024**3):.1f} GB available")
    print(f"   Current Load: CPU {psutil.cpu_percent()}%, Memory {psutil.virtual_memory().percent}%")
    print()
    print(f"{Fore.GREEN}🚀 Parallel Configuration{Style.RESET_ALL}")
    print(f"   Parallel Enabled: {config.enable_parallel}")
    print(f"   Max Workers: {config.max_workers}")
    print(f"   Process Pool: {config.use_process_pool}")
    print(f"   Memory Limit: {config.memory_limit_mb} MB")
    print(f"   Strategy: {config.strategy.value}")
    print(f"   Reason: {config.reason}")


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
            section_count = structure.get("section_count", 0)
            sections_list = structure.get("sections", [])
            
            print(f"  {Fore.BLUE}> Structure: {form} ({section_count} sections){Style.RESET_ALL}")
            
            # Show section details if available
            if sections_list:
                print(f"  {Fore.BLUE}> Section Details ({len(sections_list)} sections):{Style.RESET_ALL}")
                
                # Show all sections with enhanced formatting including ASCII labels
                for i, section in enumerate(sections_list):
                    section_type = section.get('type', 'unknown')
                    ascii_label = section.get('ascii_label', section_type)
                    start_time = section.get('start_time', 0)
                    duration = section.get('duration', 0)
                    chord_prog = section.get('chord_progression', 'Unknown')
                    
                    # Format time as mm:ss
                    start_mm = int(start_time // 60)
                    start_ss = int(start_time % 60)
                    start_time_str = f"{start_mm:02d}:{start_ss:02d}"
                    
                    # Calculate bars (1 bar ≈ 1.842s @130.5 BPM)
                    bars = round(duration / (4 * 60.0 / bpm))
                    
                    # Get features and symbolize them
                    energy_level = section.get('energy_level', 0.5)
                    complexity = section.get('complexity', 0.5)
                    
                    # Symbolize energy: low/mid/high
                    if energy_level < 0.33:
                        energy_symbol = "low E"
                    elif energy_level < 0.67:
                        energy_symbol = "mid E"
                    else:
                        energy_symbol = "high E"
                    
                    # Symbolize complexity: low/mid/high
                    if complexity < 0.33:
                        complexity_symbol = "low C"
                    elif complexity < 0.67:
                        complexity_symbol = "mid C"
                    else:
                        complexity_symbol = "high C"
                    
                    features = f"{energy_symbol}, {complexity_symbol}"
                    
                    # Display section with ASCII label
                    section_display = f"{section_type.title()}({ascii_label})" if ascii_label != section_type else section_type.title()
                    
                    if chord_prog != 'Unknown':
                        print(f"    {i+1}. {section_display} ({start_time_str}, {bars}bars, {features}): {chord_prog}")
                    else:
                        print(f"    {i+1}. {section_display} ({start_time_str}, {bars}bars, {features})")
        
        if "rhythm" in results:
            rhythm = results["rhythm"]
            time_sig = rhythm.get("time_signature", "4/4")
            groove = rhythm.get("groove_type", "straight")
            print(f"  {Fore.BLUE}> Rhythm: {time_sig} time, {groove} groove{Style.RESET_ALL}")
        
        # Show additional detailed analysis
        if "timbre" in results:
            timbre = results["timbre"]
            instruments = timbre.get("dominant_instruments", [])
            if instruments:
                inst_names = [inst.get("instrument", "unknown") for inst in instruments[:3]]
                print(f"  {Fore.MAGENTA}> Instruments: {', '.join(inst_names)}{Style.RESET_ALL}")
            
            # Show timbral characteristics
            brightness = timbre.get("brightness", 0)
            warmth = timbre.get("warmth", 0)
            print(f"  {Fore.MAGENTA}> Timbre: Brightness {brightness:.1f}, Warmth {warmth:.1f}{Style.RESET_ALL}")
        
        if "melody_harmony" in results:
            melody = results["melody_harmony"]
            if melody.get("melody_present", False):
                coverage = melody.get("melody_coverage", 0)
                range_info = melody.get("melodic_range", {})
                range_oct = range_info.get("range_octaves", 0)
                
                # Show full melodic range (including instruments)
                full_lowest = range_info.get("lowest_note_name", "Unknown")
                full_highest = range_info.get("highest_note_name", "Unknown")
                full_category = range_info.get("vocal_range_category", "Unknown")
                
                # Show vocal-only range
                vocal_lowest = range_info.get("vocal_lowest_note_name", "Unknown")
                vocal_highest = range_info.get("vocal_highest_note_name", "Unknown")
                vocal_category = range_info.get("vocal_range_category", "Unknown")
                
                print(f"  {Fore.CYAN}> Melody: {coverage:.1%} coverage, {range_oct:.1f} octave range{Style.RESET_ALL}")
                print(f"  {Fore.CYAN}> Full Range: {full_lowest} - {full_highest} ({full_category}){Style.RESET_ALL}")
                
                if vocal_lowest != "No Vocal Detected":
                    print(f"  {Fore.GREEN}> Vocal Range: {vocal_lowest} - {vocal_highest} ({vocal_category}){Style.RESET_ALL}")
                else:
                    print(f"  {Fore.YELLOW}> Vocal Range: No clear vocal melody detected{Style.RESET_ALL}")
            
            consonance = melody.get("consonance", {}).get("consonance_level", 0)
            complexity = melody.get("harmony_complexity", {}).get("harmonic_complexity", 0)
            print(f"  {Fore.CYAN}> Harmony: {consonance:.1%} consonance, {complexity:.1%} complexity{Style.RESET_ALL}")
        
        if "dynamics" in results:
            dynamics = results["dynamics"]
            # Get dynamic range from the nested structure
            dynamic_range = dynamics.get("dynamic_range", {})
            range_db = dynamic_range.get("dynamic_range_db", 0)
            
            # Calculate variation from energy variance
            energy_variance = dynamics.get("energy_variance", 0)
            # Convert variance to percentage (rough approximation)
            variation = min(1.0, energy_variance * 100) if energy_variance > 0 else 0
            
            print(f"  {Fore.YELLOW}> Dynamics: {range_db:.1f}dB range, {variation:.1%} variation{Style.RESET_ALL}")
        
        print()  # Extra line for comprehensive results
    
    # Only show the final BPM result, not the detailed candidates
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


def print_multiple_results(results: dict, detect_key: bool = False, comprehensive: bool = False) -> None:
    """Print results for multiple files."""
    print(f"\n{Fore.CYAN}{Style.BRIGHT}Analysis Results Summary{Style.RESET_ALL}")
    print(f"Processed {len(results)} files\n")
    
    for filepath, file_results in results.items():
        if isinstance(file_results, dict) and "error" in file_results:
            print(f"{Fore.RED}❌ {os.path.basename(filepath)}: {file_results['error']}{Style.RESET_ALL}")
        else:
            print_results(file_results, detect_key, comprehensive)


def analyze_file_with_progress(path: str, analyzer, args: argparse.Namespace) -> None:
    """Analyze a single audio file with progress display."""
    
    # Create progress callback for smart analyzer
    if hasattr(analyzer, '_parallel_config') and analyzer._parallel_config and analyzer._parallel_config.enable_parallel:
        # Use detailed progress display for parallel analyzer
        progress_display = None
        if args.progress:
            progress_display = create_progress_display(detailed=args.detailed_progress)
        
        def smart_progress_callback(progress: float, message: str = ""):
            global interrupted
            if interrupted:
                raise KeyboardInterrupt("Analysis interrupted by user")
            
            # Fallback progress display if no detailed display
            if not progress_display and args.progress:
                clean_message = message.replace('\n', ' ').replace('\r', ' ')
                print(f"\r{' ' * 80}\r{Fore.BLUE}Progress: {progress:.1f}% - {clean_message}{Style.RESET_ALL}", end="", flush=True)
        
        try:
            results = analyzer.analyze_file(
                path=path,
                detect_key=args.detect_key,
                comprehensive=args.comprehensive,
                min_bpm=args.min_bpm,
                max_bpm=args.max_bpm,
                start_bpm=args.start_bpm,
                progress_callback=smart_progress_callback if args.progress and not progress_display else None,
                progress_display=progress_display,
                detailed_progress=args.detailed_progress
            )
            
            if progress_display:
                progress_display.close()
            elif args.progress:
                print(f"\r{' ' * 80}\r", end="")  # Clear progress line
            
            print_results(results, args.detect_key, args.comprehensive)
            
        except Exception as e:
            if progress_display:
                progress_display.close()
            elif args.progress:
                print(f"\r{' ' * 80}\r", end="")  # Clear progress line
            print(f"{Fore.RED}Error processing {path}: {e}{Style.RESET_ALL}")
    
    else:
        # Use traditional progress for regular analyzer
        info = sf.info(path)
        bar = progress_bar(info.frames, args.sr) if args.progress else None

        def progress_callback(increment: int) -> None:
            """Update progress bar to reflect absolute progress percentage."""
            if bar:
                # ``increment`` represents the current percentage completed,
                # so set ``bar.n`` directly and refresh to synchronize the
                # visual bar with the provided value.
                bar.n = increment
                bar.refresh()

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
    parser = argparse.ArgumentParser(description="Smart BPM and Key detector with parallel processing")
    parser.add_argument("files", nargs="*", help="Audio file paths")
    parser.add_argument("--sr", type=int, default=SR_DEFAULT, help="Sample rate")
    parser.add_argument("--hop", type=int, default=HOP_DEFAULT, help="Hop length")
    parser.add_argument("--min_bpm", type=float, default=40.0, help="Minimum BPM")
    parser.add_argument("--max_bpm", type=float, default=300.0, help="Maximum BPM")
    parser.add_argument("--start_bpm", type=float, default=150.0, help="Starting BPM")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress display")
    parser.add_argument(
        "--detect-key", action="store_true", help="Enable key detection"
    )
    parser.add_argument(
        "--comprehensive", action="store_true", help="Enable comprehensive music analysis"
    )
    
    # Parallel processing options
    parallel_group = parser.add_argument_group('Parallel Processing')
    parallel_group.add_argument(
        "--auto-parallel", action="store_true", default=True,
        help="Enable automatic parallel optimization (default: enabled)"
    )
    parallel_group.add_argument(
        "--no-parallel", action="store_true",
        help="Disable parallel processing"
    )
    parallel_group.add_argument(
        "--max-workers", type=int, default=None,
        help="Override automatic worker count"
    )
    parallel_group.add_argument(
        "--detailed-progress", action="store_true",
        help="Show detailed progress for each analysis task"
    )
    parallel_group.add_argument(
        "--show-system-info", action="store_true",
        help="Show system information and parallel configuration"
    )

    args = parser.parse_args()
    
    # Show system info and exit if requested
    if args.show_system_info:
        show_system_info()
        return
    
    # Check if files are provided
    if not args.files:
        parser.print_help()
        return

    # Progress is enabled by default, disabled with --quiet
    args.progress = not args.quiet

    # Initialize analyzer
    global current_analyzer
    if args.no_parallel:
        analyzer = AudioAnalyzer(sr=args.sr, hop_length=args.hop)
        print(f"{Fore.YELLOW}⚡ Parallel processing disabled by user{Style.RESET_ALL}")
    else:
        analyzer = SmartParallelAudioAnalyzer(
            auto_parallel=args.auto_parallel,
            max_workers=args.max_workers,
            sr=args.sr,
            hop_length=args.hop
        )
    
    current_analyzer = analyzer

    # Process files
    if len(args.files) == 1:
        # Single file processing
        filepath = args.files[0]
        if not os.path.exists(filepath):
            print(f"{Fore.RED}File not found: {filepath}{Style.RESET_ALL}")
            return
        
        analyze_file_with_progress(filepath, analyzer, args)
    
    else:
        # Multiple file processing
        valid_files = [f for f in args.files if os.path.exists(f)]
        invalid_files = [f for f in args.files if not os.path.exists(f)]
        
        if invalid_files:
            print(f"{Fore.RED}Files not found: {', '.join(invalid_files)}{Style.RESET_ALL}")
        
        if not valid_files:
            return
        
        # Use smart analyzer for multiple files if available
        if hasattr(analyzer, 'analyze_file') and hasattr(analyzer, '_parallel_config'):
            try:
                def multi_progress_callback(progress: float, message: str = ""):
                    global interrupted
                    if interrupted:
                        raise KeyboardInterrupt("Analysis interrupted by user")
                    
                    if args.progress:
                        clean_message = message.replace('\n', ' ').replace('\r', ' ')
                        print(f"\r{' ' * 80}\r{Fore.BLUE}Overall Progress: {progress:.1f}% - {clean_message}{Style.RESET_ALL}", end="", flush=True)
                
                results = analyzer.analyze_file(
                    path=valid_files,
                    comprehensive=args.comprehensive,
                    detect_key=args.detect_key,
                    min_bpm=args.min_bpm,
                    max_bpm=args.max_bpm,
                    start_bpm=args.start_bpm,
                    progress_callback=multi_progress_callback if args.progress else None,
                    detailed_progress=args.detailed_progress
                )
                
                if args.progress:
                    print(f"\r{' ' * 80}\r", end="")  # Clear progress line
                
                print_multiple_results(results, args.detect_key, args.comprehensive)
                
            except Exception as e:
                if args.progress:
                    print(f"\r{' ' * 80}\r", end="")  # Clear progress line
                print(f"{Fore.RED}Error in batch processing: {e}{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}Falling back to sequential processing...{Style.RESET_ALL}")
                
                # Fallback to sequential processing
                for filepath in valid_files:
                    analyze_file_with_progress(filepath, analyzer, args)
        else:
            # Sequential processing for regular analyzer
            for filepath in valid_files:
                analyze_file_with_progress(filepath, analyzer, args)


if __name__ == "__main__":
    main()
