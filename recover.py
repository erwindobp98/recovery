#!/usr/bin/env python3
"""
OPTIMIZED Streaming multiprocessing BIP39 recovery (up to 4 missing words).
- Dynamic performance tuning berdasarkan hardware
- Thermal management dan memory optimization
- Auto-benchmarking untuk optimal configuration
- Enhanced progress monitoring dan error handling
WARNING: Prints private keys. Use only for wallets you OWN.
"""

import os
import sys
import time
import json
import math
import platform
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

# Third-party imports with error handling
try:
    import psutil
    from mnemonic import Mnemonic
    from bip_utils import Bip39SeedGenerator, Bip44, Bip44Coins, Bip44Changes
    from eth_account import Account
    from colorama import init as colorama_init, Fore, Style, Back
except ImportError as e:
    print(f"[ERROR] Missing required package: {e}")
    print("Install with: pip install psutil mnemonic bip_utils eth_account colorama")
    sys.exit(1)

# ============== CONFIGURATION CLASS ==============
@dataclass
class Config:
    # Basic settings
    partial_phrase: str = "also naive cream notice lounge mask olympic grocery slush arrive spoil ?"
    target_address: str = "0xf7c1e38dbc725a69d18922a1ce0c3c0ae62c154d"
    
    # Performance settings (will be auto-optimized)
    process_workers: int = 0
    batch_size: int = 0  
    max_in_flight: int = 0
    
    # Monitoring
    report_interval: float = 5.0
    checkpoint_file: str = "recover_checkpoint.json"
    
    # Derivation path
    account_index: int = 0
    change: Bip44Changes = Bip44Changes.CHAIN_EXT
    address_index: int = 0
    
    # Advanced options
    enable_thermal_monitoring: bool = True
    enable_auto_benchmark: bool = True
    cache_addresses: bool = False
    prefetch_batches: int = 1

# ============== HARDWARE DETECTION & OPTIMIZATION ==============
class HardwareOptimizer:
    @staticmethod
    def detect_cpu_info() -> Dict[str, Any]:
        """Detect CPU characteristics for optimization"""
        try:
            cpu_count_logical = os.cpu_count() or 4
            cpu_count_physical = psutil.cpu_count(logical=False) or cpu_count_logical // 2
            
            # Detect CPU vendor
            processor_info = platform.processor().lower()
            is_intel = 'intel' in processor_info
            is_amd = 'amd' in processor_info
            
            return {
                'logical_cores': cpu_count_logical,
                'physical_cores': cpu_count_physical,
                'is_intel': is_intel,
                'is_amd': is_amd,
                'vendor': processor_info
            }
        except Exception as e:
            print(f"[WARN] CPU detection failed: {e}")
            return {
                'logical_cores': 4,
                'physical_cores': 2,
                'is_intel': True,
                'is_amd': False,
                'vendor': 'unknown'
            }
    
    @staticmethod
    def detect_memory_info() -> Dict[str, Any]:
        """Detect memory for optimization"""
        try:
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            total_gb = memory.total / (1024**3)
            
            return {
                'total_gb': total_gb,
                'available_gb': available_gb,
                'usage_percent': memory.percent
            }
        except Exception as e:
            print(f"[WARN] Memory detection failed: {e}")
            return {'total_gb': 8.0, 'available_gb': 4.0, 'usage_percent': 50.0}
    
    @staticmethod
    def get_optimal_workers(cpu_info: Dict[str, Any]) -> int:
        """Calculate optimal worker count based on CPU"""
        logical = cpu_info['logical_cores']
        physical = cpu_info['physical_cores']
        
        # Intel with hyperthreading: prefer physical cores for crypto
        if cpu_info['is_intel'] and logical > physical:
            return max(1, physical - 1)
        
        # AMD Ryzen: can utilize logical cores better
        elif cpu_info['is_amd']:
            return max(1, min(logical - 1, int(physical * 1.5)))
        
        # Default: conservative approach
        else:
            return max(1, logical - 2)
    
    @staticmethod
    def get_optimal_batch_size(memory_info: Dict[str, Any]) -> int:
        """Calculate optimal batch size based on available memory"""
        available_gb = memory_info['available_gb']
        
        if available_gb >= 32:
            return 32768  # High-end systems
        elif available_gb >= 16:
            return 16384  # Gaming/workstation
        elif available_gb >= 8:
            return 8192   # Mid-range
        elif available_gb >= 4:
            return 4096   # Entry level
        else:
            return 2048   # Low memory
    
    @staticmethod
    def check_thermal_sensors() -> bool:
        """Check if thermal monitoring is available"""
        try:
            temps = psutil.sensors_temperatures()
            return len(temps) > 0
        except:
            return False

# ============== THERMAL MANAGEMENT ==============
class ThermalManager:
    def __init__(self, config: Config):
        self.config = config
        self.thermal_available = HardwareOptimizer.check_thermal_sensors()
        self.base_workers = config.process_workers
        
    def get_current_temp(self) -> Optional[float]:
        """Get current CPU temperature"""
        if not self.thermal_available:
            return None
            
        try:
            temps = psutil.sensors_temperatures()
            if 'coretemp' in temps:
                return max(temp.current for temp in temps['coretemp'] if temp.current)
            elif 'cpu_thermal' in temps:  # Raspberry Pi
                return temps['cpu_thermal'][0].current
        except:
            pass
        return None
    
    def adjust_workers_for_thermal(self) -> int:
        """Adjust worker count based on temperature"""
        temp = self.get_current_temp()
        if temp is None:
            return self.base_workers
            
        # Thermal throttling thresholds
        if temp > 85:  # Critical
            return max(1, self.base_workers // 4)
        elif temp > 80:  # High
            return max(2, self.base_workers // 2)
        elif temp > 75:  # Moderate
            return max(2, int(self.base_workers * 0.75))
        else:
            return self.base_workers

# ============== AUTO-BENCHMARKING ==============
class AutoBenchmark:
    @staticmethod
    def run_mini_benchmark(workers: int, batch_size: int, duration: float = 30.0) -> float:
        """Run a mini benchmark to test configuration performance"""
        print(f"[BENCHMARK] Testing {workers} workers, batch {batch_size}...")
        
        # Use a smaller phrase for benchmarking (1 missing word)
        test_phrase = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon ?"
        test_words = test_phrase.split()
        test_missing = [i for i, w in enumerate(test_words) if w == "?"]
        
        mnemo = Mnemonic("english")
        wordlist = mnemo.wordlist
        
        start_time = time.time()
        processed = 0
        
        with ProcessPoolExecutor(max_workers=workers) as executor:
            # Submit a few test batches
            futures = []
            for i in range(min(3, workers * 2)):  # Limited test batches
                fut = executor.submit(
                    benchmark_worker, i * batch_size, batch_size,
                    test_words, test_missing, [wordlist], "dummy_address"
                )
                futures.append(fut)
            
            # Process results until timeout
            while futures and (time.time() - start_time) < duration:
                done, futures = wait(futures, timeout=1.0, return_when=FIRST_COMPLETED)
                for fut in done:
                    try:
                        _, _, count = fut.result()
                        processed += count
                    except:
                        pass
            
            # Cancel remaining futures
            for fut in futures:
                fut.cancel()
        
        elapsed = time.time() - start_time
        rate = processed / elapsed if elapsed > 0 else 0
        print(f"[BENCHMARK] Rate: {rate:.1f} combinations/sec")
        return rate
    
    @staticmethod
    def find_optimal_config(cpu_info: Dict, memory_info: Dict) -> Dict[str, int]:
        """Auto-benchmark to find optimal configuration"""
        print(f"{Fore.CYAN}[AUTO-BENCHMARK] Finding optimal configuration...{Style.RESET_ALL}")
        
        # Test configurations
        base_workers = HardwareOptimizer.get_optimal_workers(cpu_info)
        base_batch = HardwareOptimizer.get_optimal_batch_size(memory_info)
        
        configs_to_test = [
            {"workers": base_workers, "batch_size": base_batch},
            {"workers": base_workers - 1, "batch_size": base_batch * 2},
            {"workers": base_workers + 1, "batch_size": base_batch // 2},
            {"workers": max(2, base_workers // 2), "batch_size": base_batch * 2},
        ]
        
        best_config = configs_to_test[0]
        best_rate = 0.0
        
        for config in configs_to_test:
            if config["workers"] <= 0:
                continue
                
            try:
                rate = AutoBenchmark.run_mini_benchmark(
                    config["workers"], config["batch_size"], duration=15.0
                )
                if rate > best_rate:
                    best_rate = rate
                    best_config = config
            except Exception as e:
                print(f"[WARN] Benchmark failed for {config}: {e}")
                continue
        
        print(f"{Fore.GREEN}[AUTO-BENCHMARK] Optimal: {best_config['workers']} workers, "
              f"batch {best_config['batch_size']} ({best_rate:.1f}/sec){Style.RESET_ALL}")
        
        return best_config

# ============== WORKER FUNCTIONS ==============
def benchmark_worker(start_index: int, count: int, words_list: List[str], 
                    missing_idx: List[int], clues: List[List[str]], 
                    target_addr: str) -> Tuple[bool, Optional[Dict], int]:
    """Simplified worker for benchmarking"""
    from mnemonic import Mnemonic as _Mnemonic
    local_mnemo = _Mnemonic("english")
    
    checked = 0
    radices = [len(l) for l in clues]
    
    # Pre-compute radices products
    radices_prod = []
    for i in range(len(radices)):
        prod = 1
        for r in radices[i+1:]:
            prod *= r
        radices_prod.append(prod)
    
    for offset in range(min(count, 1000)):  # Limit for benchmark
        idx = start_index + offset
        if idx >= math.prod(radices):
            break
            
        # Generate combination using mixed-radix
        digits = []
        rem = idx
        for r, factor in zip(radices, radices_prod):
            digit = (rem // factor) % r if factor != 0 else rem % r
            digits.append(digit)
        
        # Build phrase
        phrase_words = words_list[:]
        for pos, digit in zip(missing_idx, digits):
            phrase_words[pos] = clues[missing_idx.index(pos)][digit]
        
        phrase = " ".join(phrase_words)
        checked += 1
        
        # Quick checksum validation only
        if not local_mnemo.check(phrase):
            continue
    
    return False, None, checked

def optimized_process_batch(start_index: int, count: int, words_list: List[str],
                           missing_idx: List[int], clues: List[List[str]],
                           target_addr: str, config_dict: Dict) -> Tuple[bool, Optional[Dict], int]:
    """Optimized worker process with better error handling"""
    try:
        from mnemonic import Mnemonic as _Mnemonic
        from bip_utils import Bip39SeedGenerator as _Gen, Bip44 as _B44, Bip44Coins as _Coins, Bip44Changes as _Changes
        from eth_account import Account as _Account
        
        local_mnemo = _Mnemonic("english")
        checked = 0
        
        # Extract config
        account_idx = config_dict.get('account_index', 0)
        change = config_dict.get('change', 0)  # Convert to int for pickling
        addr_idx = config_dict.get('address_index', 0)
        
        # Pre-compute radices products
        radices = [len(l) for l in clues]
        radices_prod = []
        for i in range(len(radices)):
            prod = 1
            for r in radices[i+1:]:
                prod *= r
            radices_prod.append(prod)
        
        total_combinations = math.prod(radices)
        
        for offset in range(count):
            idx = start_index + offset
            if idx >= total_combinations:
                break
                
            # Generate combination using mixed-radix arithmetic
            digits = []
            rem = idx
            for r, factor in zip(radices, radices_prod):
                digit = (rem // factor) % r if factor != 0 else rem % r
                digits.append(digit)
            
            # Build phrase
            phrase_words = words_list[:]
            for pos, digit in zip(missing_idx, digits):
                phrase_words[pos] = clues[missing_idx.index(pos)][digit]
            
            phrase = " ".join(phrase_words)
            checked += 1
            
            # Fast BIP39 checksum validation first
            if not local_mnemo.check(phrase):
                continue
            
            try:
                # Generate seed and derive address
                seed_bytes = _Gen(phrase).Generate()
                bip44_acc = _B44.FromSeed(seed_bytes, _Coins.ETHEREUM)
                derived = (bip44_acc.Purpose().Coin().Account(account_idx)
                          .Change(_Changes.CHAIN_EXT if change == 0 else _Changes.CHAIN_INT)
                          .AddressIndex(addr_idx))
                
                priv_hex = derived.PrivateKey().Raw().ToHex()
                addr = _Account.from_key(bytes.fromhex(priv_hex)).address.lower()
                
                if addr == target_addr.lower():
                    return True, {
                        "phrase": phrase,
                        "private_key": priv_hex,
                        "address": addr,
                        "index": idx
                    }, checked
                    
            except Exception as derive_error:
                # Skip this combination if derivation fails
                continue
        
        return False, None, checked
        
    except Exception as e:
        # Return error info for debugging
        return False, {"error": str(e)}, checked

# ============== PROGRESS DISPLAY ==============
class ProgressDisplay:
    def __init__(self):
        self.start_time = time.time()
        self.last_temp_check = 0
        
    def format_time(self, seconds: float) -> str:
        if seconds == float('inf') or seconds > 365 * 24 * 3600:
            return "∞"
        
        days, remainder = divmod(int(seconds), 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if days:
            return f"{days}d{hours:02d}h"
        elif hours:
            return f"{hours}h{minutes:02d}m"
        elif minutes:
            return f"{minutes}m{seconds:02d}s"
        else:
            return f"{seconds}s"
    
    def print_progress(self, checked: int, total: int, rate: float, 
                      current_workers: int, current_temp: Optional[float] = None):
        elapsed = time.time() - self.start_time
        pct = (checked / total * 100) if total > 0 else 0
        
        # Progress bar
        bar_width = 40
        filled = int((pct / 100.0) * bar_width)
        bar = f"[{'█' * filled}{'░' * (bar_width - filled)}]"
        
        # Color coding
        if pct < 25:
            color = Fore.RED
        elif pct < 50:
            color = Fore.YELLOW
        elif pct < 75:
            color = Fore.BLUE
        else:
            color = Fore.GREEN
        
        # ETA calculation
        remaining = max(0, total - checked)
        eta = remaining / rate if rate > 0 else float('inf')
        
        # Status line
        temp_str = f" 🌡️{current_temp:.1f}°C" if current_temp else ""
        status = (f"\r{color}{bar} {pct:6.2f}%{Style.RESET_ALL} "
                 f"│ {checked:,}/{total:,} │ {rate:.1f}/s │ "
                 f"⏱️{self.format_time(elapsed)} │ ETA {self.format_time(eta)} │ "
                 f"👥{current_workers}{temp_str}")
        
        # Ensure we clear the line
        print(status.ljust(120), end='', flush=True)

# ============== CHECKPOINT MANAGEMENT ==============
class CheckpointManager:
    def __init__(self, filepath: str):
        self.filepath = filepath
    
    def load(self) -> Dict[str, Any]:
        if not os.path.exists(self.filepath):
            return {"offset": 0, "checked": 0, "found": False}
        
        try:
            with open(self.filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARN] Checkpoint corrupted: {e}. Starting fresh.")
            return {"offset": 0, "checked": 0, "found": False}
    
    def save(self, state: Dict[str, Any]):
        try:
            tmp_file = self.filepath + ".tmp"
            with open(tmp_file, 'w') as f:
                json.dump(state, f, indent=2)
            os.replace(tmp_file, self.filepath)
        except Exception as e:
            print(f"[WARN] Failed to save checkpoint: {e}")
    
    def reset(self):
        """Reset checkpoint to start fresh"""
        try:
            if os.path.exists(self.filepath):
                os.remove(self.filepath)
            print(f"[INFO] Checkpoint reset successfully")
        except Exception as e:
            print(f"[WARN] Failed to reset checkpoint: {e}")
    
    def exists(self) -> bool:
        """Check if checkpoint file exists"""
        return os.path.exists(self.filepath)

# ============== MAIN RECOVERY ENGINE ==============
class RecoveryEngine:
    def __init__(self, config: Config):
        self.config = config
        self.mnemo = Mnemonic("english")
        self.checkpoint_mgr = CheckpointManager(config.checkpoint_file)
        self.progress = ProgressDisplay()
        self.thermal_mgr = None
        
        # Initialize colorama
        colorama_init(autoreset=True)
        
    def initialize(self):
        """Initialize and optimize configuration"""
        print(f"{Fore.CYAN}=== OPTIMIZED BIP39 RECOVERY TOOL ==={Style.RESET_ALL}")
        
        # Hardware detection
        cpu_info = HardwareOptimizer.detect_cpu_info()
        memory_info = HardwareOptimizer.detect_memory_info()
        
        print(f"[INFO] CPU: {cpu_info['logical_cores']} logical, {cpu_info['physical_cores']} physical cores")
        print(f"[INFO] RAM: {memory_info['available_gb']:.1f}GB available / {memory_info['total_gb']:.1f}GB total")
        
        # Auto-benchmark if enabled
        if self.config.enable_auto_benchmark:
            optimal = AutoBenchmark.find_optimal_config(cpu_info, memory_info)
            self.config.process_workers = optimal['workers']
            self.config.batch_size = optimal['batch_size']
        else:
            # Use hardware-optimized defaults
            self.config.process_workers = HardwareOptimizer.get_optimal_workers(cpu_info)
            self.config.batch_size = HardwareOptimizer.get_optimal_batch_size(memory_info)
        
        # Set pipeline depth
        self.config.max_in_flight = self.config.process_workers + 2
        
        # Initialize thermal management
        if self.config.enable_thermal_monitoring:
            self.thermal_mgr = ThermalManager(self.config)
        
        print(f"[CONFIG] Workers: {self.config.process_workers}, Batch: {self.config.batch_size}, Pipeline: {self.config.max_in_flight}")
    
    def reset_checkpoint(self):
        """Reset checkpoint untuk pencarian baru"""
        self.checkpoint_mgr.reset()
        
    def check_existing_checkpoint(self) -> bool:
        """Check dan tanya user apakah mau resume atau start fresh"""
        if self.checkpoint_mgr.exists():
            checkpoint = self.checkpoint_mgr.load()
            checked = checkpoint.get("checked", 0)
            if checked > 0:
                print(f"\n{Fore.YELLOW}[CHECKPOINT] Ditemukan progress sebelumnya: {checked:,} combinations checked{Style.RESET_ALL}")
                while True:
                    choice = input("Resume from checkpoint? (y/n/r=reset): ").strip().lower()
                    if choice in ['y', 'yes', '']:
                        return True  # Resume
                    elif choice in ['n', 'no']:
                        return False  # Start fresh (don't reset, just ignore)
                    elif choice in ['r', 'reset']:
                        self.reset_checkpoint()
                        return False  # Start fresh after reset
                    else:
                        print("Please enter 'y' to resume, 'n' to start fresh, or 'r' to reset")
        return False
        
    def parse_phrase(self) -> Tuple[List[str], List[int], int]:
        """Parse partial phrase and setup search space"""
        words = self.config.partial_phrase.split()
        if len(words) != 12:
            raise ValueError("Partial phrase harus 12 kata (gunakan '?' untuk kata hilang)")
        
        missing_indexes = [i for i, w in enumerate(words) if w == "?"]
        num_missing = len(missing_indexes)
        
        if num_missing == 0:
            raise ValueError("Tidak ada '?' dalam partial phrase")
        if num_missing > 4:
            raise ValueError("Script ini dioptimalkan untuk maksimal 4 kata hilang")
        
        total_combinations = len(self.mnemo.wordlist) ** num_missing
        
        print(f"[INFO] Missing positions: {missing_indexes}")
        print(f"[INFO] Total combinations: {total_combinations:,}")
        
        return words, missing_indexes, total_combinations
    
    def run_recovery(self):
        """Main recovery process"""
        try:
            words, missing_indexes, total_combinations = self.parse_phrase()
            clues_lists = [self.mnemo.wordlist] * len(missing_indexes)
            
            # Load checkpoint dengan user confirmation
            should_resume = self.check_existing_checkpoint()
            if should_resume:
                checkpoint = self.checkpoint_mgr.load()
                offset = checkpoint.get("offset", 0)
                checked_total = checkpoint.get("checked", 0)
                print(f"[INFO] Resuming from offset {offset:,}, checked: {checked_total:,}")
            else:
                offset = 0
                checked_total = 0
                print(f"[INFO] Starting fresh search")
            
            # Prepare config for workers (make it pickleable)
            worker_config = {
                'account_index': self.config.account_index,
                'change': 0 if self.config.change == Bip44Changes.CHAIN_EXT else 1,
                'address_index': self.config.address_index
            }
            
            found_result = None
            last_report = 0
            
            with ProcessPoolExecutor(max_workers=self.config.process_workers) as executor:
                futures_map = {}
                next_start = offset
                
                # Submit initial jobs
                while len(futures_map) < self.config.max_in_flight and next_start < total_combinations:
                    fut = executor.submit(
                        optimized_process_batch, next_start, self.config.batch_size,
                        words, missing_indexes, clues_lists, 
                        self.config.target_address.lower(), worker_config
                    )
                    futures_map[fut] = next_start
                    next_start += self.config.batch_size
                
                # Main processing loop
                while futures_map:
                    done, _ = wait(futures_map.keys(), return_when=FIRST_COMPLETED)
                    
                    for fut in list(done):
                        start_idx = futures_map.pop(fut)
                        
                        try:
                            found, data, count = fut.result()
                        except Exception as e:
                            print(f"\n{Fore.RED}[ERROR]{Style.RESET_ALL} Worker failed: {e}")
                            found, data, count = False, None, 0
                        
                        checked_total += count
                        
                        # Update checkpoint
                        checkpoint_state = {
                            "offset": max(next_start, start_idx + self.config.batch_size),
                            "checked": checked_total,
                            "timestamp": time.time()
                        }
                        self.checkpoint_mgr.save(checkpoint_state)
                        
                        # Progress reporting
                        now = time.time()
                        if now - last_report >= self.config.report_interval:
                            elapsed = now - self.progress.start_time
                            rate = checked_total / elapsed if elapsed > 0 else 0
                            
                            # Check thermal status
                            current_workers = self.config.process_workers
                            current_temp = None
                            if self.thermal_mgr:
                                current_temp = self.thermal_mgr.get_current_temp()
                                current_workers = self.thermal_mgr.adjust_workers_for_thermal()
                            
                            self.progress.print_progress(
                                checked_total, total_combinations, rate, 
                                current_workers, current_temp
                            )
                            last_report = now
                        
                        # Check if found
                        if found:
                            found_result = data
                            
                            # Reset checkpoint setelah found (siap untuk pencarian baru)
                            self.reset_checkpoint()
                            print(f"\n{Fore.GREEN}[SUCCESS] Checkpoint telah direset untuk pencarian selanjutnya{Style.RESET_ALL}")
                            
                            # Cancel remaining futures
                            for pending in list(futures_map.keys()):
                                pending.cancel()
                            break
                        
                        # Submit next job
                        if next_start < total_combinations:
                            fut = executor.submit(
                                optimized_process_batch, next_start, self.config.batch_size,
                                words, missing_indexes, clues_lists,
                                self.config.target_address.lower(), worker_config
                            )
                            futures_map[fut] = next_start
                            next_start += self.config.batch_size
                    
                    if found_result:
                        break
            
            # Final results
            print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
            elapsed_total = time.time() - self.progress.start_time
            final_rate = checked_total / elapsed_total if elapsed_total > 0 else 0
            
            print(f"[COMPLETE] Processed {checked_total:,} combinations in {self.progress.format_time(elapsed_total)}")
            print(f"[PERFORMANCE] Average rate: {final_rate:.1f} combinations/sec")
            
            if found_result:
                print(f"\n{Back.GREEN}{Fore.BLACK} SUCCESS: WALLET RECOVERED! {Style.RESET_ALL}")
                print(f"Seed phrase : {found_result['phrase']}")
                print(f"Address     : {found_result['address']}")
                print(f"Private key : {found_result['private_key']}")
                print(f"\n{Fore.YELLOW}⚠️  IMMEDIATELY verify this on a secure/offline machine!{Style.RESET_ALL}")
            else:
                print(f"{Fore.YELLOW}[RESULT] No matching wallet found in search space{Style.RESET_ALL}")
                
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}[INTERRUPTED] Recovery stopped by user{Style.RESET_ALL}")
            print("[INFO] Progress saved. Run again to resume.")
        except Exception as e:
            print(f"\n{Fore.RED}[FATAL ERROR] {e}{Style.RESET_ALL}")
            raise

# ============== MAIN ENTRY POINT ==============
def main():
    """Main entry point with command line options"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimized BIP39 Recovery Tool")
    parser.add_argument("--reset", action="store_true", help="Reset checkpoint and start fresh")
    parser.add_argument("--phrase", type=str, help="Partial phrase with ? for missing words")
    parser.add_argument("--address", type=str, help="Target address to find")
    parser.add_argument("--no-benchmark", action="store_true", help="Skip auto-benchmarking")
    parser.add_argument("--no-thermal", action="store_true", help="Disable thermal monitoring")
    
    args = parser.parse_args()
    
    # Configuration - EDIT THESE VALUES or use command line args
    config = Config(
        partial_phrase=args.phrase or "also naive cream notice lounge mask olympic grocery slush arrive spoil ?",
        target_address=args.address or "0xf7c1e38dbc725a69d18922a1ce0c3c0ae62c154d",
        
        # Performance options
        enable_auto_benchmark=not args.no_benchmark,
        enable_thermal_monitoring=not args.no_thermal,
        report_interval=3.0,
        
        # Advanced (biasanya tidak perlu diubah)
        checkpoint_file="recovery_checkpoint.json",
        account_index=0,
        address_index=0
    )
    
    # Create recovery engine
    engine = RecoveryEngine(config)
    
    # Handle reset command
    if args.reset:
        engine.reset_checkpoint()
        print(f"{Fore.GREEN}[SUCCESS] Checkpoint has been reset{Style.RESET_ALL}")
        return
    
    # Initialize and run
    engine.initialize()
    engine.run_recovery()

if __name__ == "__main__":
    main()
