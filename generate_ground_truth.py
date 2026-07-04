#!/usr/bin/env python3
"""
Generate ground truth output files for all samples.

Transpiles and runs every .pine sample from both sample/ and sample_pds/,
then copies the output files to ground_truth/{suite}/{sample_name}/.

Usage:
    python generate_ground_truth.py                    # Run all
    python generate_ground_truth.py --suite sample     # Only sample/
    python generate_ground_truth.py --suite sample_pds # Only sample_pds/
    python generate_ground_truth.py -j 12              # 12 parallel workers
    python generate_ground_truth.py --data workdir/data/VN30F1M_15m.ohlcv
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
WORKDIR = BASE_DIR / 'workdir'
SCRIPTS_DIR = WORKDIR / 'scripts'
OUTPUT_DIR = WORKDIR / 'output'
GROUND_TRUTH_DIR = BASE_DIR / 'ground_truth'
PYNE_BIN = BASE_DIR / '.venv' / 'bin' / 'pyne'

SUITES = {
    'sample': BASE_DIR / 'sample' / 'pinescript',
    'sample_pds': BASE_DIR / 'sample_pds',
}

# Skip files containing these keywords (require external data/API)
SKIP_KEYWORDS = ['import', 'request', 'ticker', 'symbols', 'tz', 'matrix', 'chart.point', 'polyline.new']

sys.path.insert(0, str(BASE_DIR))
from pine2pyne import transpile
from pine2pyne.errors import Pine2PyneError


def should_skip_file(pine_file):
    """Check if file contains skip keywords."""
    try:
        content = pine_file.read_text(encoding='utf-8').lower()
        for keyword in SKIP_KEYWORDS:
            if keyword in content:
                return True, keyword
    except Exception:
        pass
    return False, None


def find_pyne():
    """Find the pyne binary."""
    if PYNE_BIN.exists():
        return str(PYNE_BIN)
    found = shutil.which('pyne')
    if found:
        return found
    print("ERROR: pyne not found.", file=sys.stderr)
    sys.exit(1)


def process_sample(pine_file, suite_name, pyne_bin, data_file, timeout=15):
    """Transpile, run, and collect output for one sample.

    Returns dict with status and list of output files collected.
    """
    name = pine_file.stem
    result = {
        'name': name,
        'suite': suite_name,
        'filename': pine_file.name,
        'status': 'unknown',
        'error': '',
        'output_files': [],
    }

    # Step 1: Transpile
    try:
        source = pine_file.read_text(encoding='utf-8')
        output = transpile(source)
        if not output:
            result['status'] = 'skip_empty'
            return result
        output_file = SCRIPTS_DIR / f'{name}.py'
        output_file.write_text(output, encoding='utf-8')
    except (Pine2PyneError, Exception) as exc:
        result['status'] = 'transpile_fail'
        result['error'] = str(exc)[:300]
        return result

    # Step 2: Run with pyne
    output_file = SCRIPTS_DIR / f'{name}.py'
    try:
        proc = subprocess.run(
            [pyne_bin, '-w', str(WORKDIR), 'run',
             str(output_file), str(data_file)],
            capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode != 0:
            result['status'] = 'runtime_fail'
            result['error'] = proc.stderr[-300:] if proc.stderr else ''
            return result
    except subprocess.TimeoutExpired:
        result['status'] = 'timeout'
        return result

    # Step 3: Collect output files
    # pyne run outputs: {name}.csv, {name}_strat.csv, {name}_trade.csv, {name}_plot_meta.json
    collected = []
    for suffix in ['.csv', '_strat.csv', '_trade.csv', '_plot_meta.json']:
        src = OUTPUT_DIR / f'{name}{suffix}'
        if src.exists():
            dest_dir = GROUND_TRUTH_DIR / suite_name / name
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / src.name
            shutil.copy2(src, dest)
            collected.append(src.name)

    result['status'] = 'ok'
    result['output_files'] = collected
    return result


def run_suite(suite_name, suite_dir, pyne_bin, data_file, timeout, jobs):
    """Run all samples in a suite."""
    pine_files = sorted(suite_dir.glob('*.pine'))
    if not pine_files:
        print(f"  No .pine files found in {suite_dir}")
        return []

    # Filter skippable files
    testable = []
    skipped = 0
    for pine_file in pine_files:
        skip, reason = should_skip_file(pine_file)
        if skip:
            skipped += 1
        else:
            testable.append(pine_file)

    total = len(testable)
    print(f"  {suite_name}: {total} testable, {skipped} skipped (of {len(pine_files)} total)")

    results = []
    if jobs == 1:
        for index, pine_file in enumerate(testable, 1):
            result = process_sample(pine_file, suite_name, pyne_bin, data_file, timeout)
            status_char = 'OK' if result['status'] == 'ok' else result['status'].upper()
            print(f"  [{index:4d}/{total}] {pine_file.name:55s} {status_char}")
            results.append(result)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            future_to_file = {
                executor.submit(process_sample, pf, suite_name, pyne_bin, data_file, timeout): pf
                for pf in testable
            }
            done_count = 0
            for future in as_completed(future_to_file):
                done_count += 1
                result = future.result()
                status_char = 'OK' if result['status'] == 'ok' else result['status'].upper()
                print(f"  [{done_count:4d}/{total}] {result['filename']:55s} {status_char}")
                results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(description='Generate ground truth outputs for all samples')
    parser.add_argument('--suite', choices=['sample', 'sample_pds', 'both'], default='both',
                        help='Which suite(s) to run (default: both)')
    parser.add_argument('--data', type=Path, default=WORKDIR / 'data' / 'VN30F1M_15m.ohlcv',
                        help='OHLCV data file for pyne run')
    parser.add_argument('--timeout', type=int, default=15,
                        help='Timeout per file in seconds (default: 15)')
    parser.add_argument('-j', '--jobs', type=int, default=12,
                        help='Parallel workers (default: 12, 1=sequential)')
    args = parser.parse_args()

    if not args.data.exists():
        print(f"ERROR: Data file not found: {args.data}", file=sys.stderr)
        return 1

    pyne_bin = find_pyne()

    # Clean and create ground truth directory
    if GROUND_TRUTH_DIR.exists():
        print(f"Removing existing ground_truth/ directory...")
        shutil.rmtree(GROUND_TRUTH_DIR)
    GROUND_TRUTH_DIR.mkdir(parents=True)

    # Ensure scripts dir exists
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    suites_to_run = ['sample', 'sample_pds'] if args.suite == 'both' else [args.suite]

    print("=" * 80)
    print("GROUND TRUTH GENERATION")
    print("=" * 80)
    print(f"  Data:     {args.data}")
    print(f"  Output:   {GROUND_TRUTH_DIR}")
    print(f"  Jobs:     {args.jobs}")
    print(f"  Timeout:  {args.timeout}s")
    print(f"  Started:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    all_results = []
    for suite_name in suites_to_run:
        suite_dir = SUITES[suite_name]
        print(f"\n--- Suite: {suite_name} ---")
        results = run_suite(suite_name, suite_dir, pyne_bin, str(args.data), args.timeout, args.jobs)
        all_results.extend(results)

    # Summary
    ok_count = sum(1 for r in all_results if r['status'] == 'ok')
    transpile_fail = sum(1 for r in all_results if r['status'] == 'transpile_fail')
    runtime_fail = sum(1 for r in all_results if r['status'] == 'runtime_fail')
    timeout_count = sum(1 for r in all_results if r['status'] == 'timeout')
    total_outputs = sum(len(r['output_files']) for r in all_results)

    print("\n" + "=" * 80)
    print("GROUND TRUTH SUMMARY")
    print("=" * 80)
    print(f"  Total tested:       {len(all_results)}")
    print(f"  Passed (OK):        {ok_count}")
    print(f"  Transpile fail:     {transpile_fail}")
    print(f"  Runtime fail:       {runtime_fail}")
    print(f"  Timeout:            {timeout_count}")
    print(f"  Output files saved: {total_outputs}")
    print(f"  Ground truth dir:   {GROUND_TRUTH_DIR}")
    print("=" * 80)

    # Save manifest
    manifest = {
        'generated_at': datetime.now().isoformat(),
        'data_file': str(args.data),
        'total_tested': len(all_results),
        'passed': ok_count,
        'transpile_fail': transpile_fail,
        'runtime_fail': runtime_fail,
        'timeout': timeout_count,
        'total_output_files': total_outputs,
        'samples': all_results,
    }
    manifest_path = GROUND_TRUTH_DIR / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  Manifest:           {manifest_path}")

    # List runtime failures for reference
    if runtime_fail > 0:
        print(f"\n--- Runtime failures ({runtime_fail}) ---")
        for r in sorted(all_results, key=lambda x: x['filename']):
            if r['status'] == 'runtime_fail':
                error_short = r['error'].split('\n')[-1][:80] if r['error'] else ''
                print(f"  {r['filename']:55s} {error_short}")

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
