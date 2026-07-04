#!/usr/bin/env python3
"""
Test Pine Script samples from sample_pds/: transpile to Python, then run with pyne.

Usage:
    python test_all_sample_pds.py                    # Run all samples (24 cores)
    python test_all_sample_pds.py "ex_001*"          # Filter by pattern
    python test_all_sample_pds.py --timeout 30       # Custom timeout
    python test_all_sample_pds.py --verbose           # Show stderr on failure
    python test_all_sample_pds.py --retry-failed     # Only run files that failed last time
    python test_all_sample_pds.py -j 1               # Run sequentially
    python test_all_sample_pds.py -j 0               # Use all available CPU cores
"""

import argparse
import os
import shutil
import subprocess
import sys
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import re

BASE_DIR = Path(__file__).parent.resolve()
SAMPLE_DIR = BASE_DIR / 'sample_pds'
WORKDIR = BASE_DIR / 'workdir'
SCRIPTS_DIR = WORKDIR / 'scripts'
DEFAULT_DATA = WORKDIR / 'data' / 'demo.ohlcv'
PYNE_BIN = BASE_DIR / '.venv' / 'bin' / 'pyne'
RESULTS_JSON = BASE_DIR / 'test_results_pds.json'
RESULTS_TXT = BASE_DIR / 'test_results_pds.txt'
RESULTS_BAK = BASE_DIR / 'test_results_pds.json.bak'

sys.path.insert(0, str(BASE_DIR))

from pine2pyne import transpile
from pine2pyne.errors import Pine2PyneError

# Skip files containing these keywords (require external data/API)
SKIP_KEYWORDS = ['import', 'request', 'ticker', 'symbols', 'tz', 'matrix', 'chart.point', 'polyline.new']

ERROR_PATTERNS = {
    'missing_import': [
        r"NameError: name '(\w+)' is not defined",
        r"name '(\w+)' is not defined",
    ],
    'pynecore_api': [
        r"has no attribute '(\w+)'",
        r"not implemented",
        r"No matching implementation found",
        r"AttributeError.*'(\w+)'",
    ],
    'wrong_args': [
        r"takes \d+ (positional )?argument",
        r"missing \d+ required",
        r"unexpected keyword argument",
    ],
    'type_error': [
        r"not supported between",
        r"TypeError.*'(\w+)'",
    ],
    'transpile_error': [
        r"ParserError",
        r"LexerError",
        r"TransformerError",
    ],
}


def classify_error(error_text, verbose=False):
    """Classify error based on patterns."""
    for error_type, patterns in ERROR_PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern, error_text)
            if match:
                detail = match.group(1) if match.groups() else ""
                return error_type, detail

    if verbose:
        print(f"\n{'='*80}\nUNKNOWN ERROR DETECTED:\n{'-'*80}\n{error_text}\n{'='*80}\n")

    return 'unknown', error_text[:200]


def should_skip_file(pine_file):
    """Check if file contains skip keywords (request, ticker, symbols)."""
    try:
        content = pine_file.read_text(encoding='utf-8').lower()
        for keyword in SKIP_KEYWORDS:
            if keyword in content:
                return True, keyword
    except Exception:
        pass
    return False, None


def load_failed_files():
    """Load list of failed files from last test run."""
    if not RESULTS_JSON.exists():
        return None

    try:
        data = json.loads(RESULTS_JSON.read_text())
        if isinstance(data, list) and len(data) > 0:
            last_run = data[-1]
        elif isinstance(data, dict):
            last_run = data
        else:
            return None

        if 'failed_files' not in last_run:
            return 'missing_key'
        return set(last_run.get('failed_files', []))

    except (json.JSONDecodeError, OSError, KeyError):
        pass

    return None


def find_pyne():
    """Find the pyne binary."""
    if PYNE_BIN.exists():
        return str(PYNE_BIN)
    found = shutil.which('pyne')
    if found:
        return found
    print("ERROR: pyne not found. Activate the venv or install pynecore.", file=sys.stderr)
    sys.exit(1)


def test_file(pine_file, pyne_bin, data_file, timeout=15, verbose=False):
    """Test a single Pine file: transpile then run with pyne."""
    name = pine_file.stem
    result = {
        'name': name,
        'filename': pine_file.name,
        'transpile': False,
        'runtime': False,
        'error_type': None,
        'error_detail': '',
        'error_stderr': '',
        'transpile_time': 0,
        'runtime_time': 0,
    }

    # Step 1: Transpile .pine -> .py
    start = datetime.now()
    try:
        source = pine_file.read_text(encoding='utf-8')
        output = transpile(source)
        if not output:
            result['transpile_time'] = (datetime.now() - start).total_seconds()
            result['transpile'] = True
            result['runtime'] = True
            return result
        output_file = SCRIPTS_DIR / f'{name}.py'
        output_file.write_text(output, encoding='utf-8')
        result['transpile_time'] = (datetime.now() - start).total_seconds()
        result['transpile'] = True
    except Pine2PyneError as e:
        result['transpile_time'] = (datetime.now() - start).total_seconds()
        error_type, detail = classify_error(str(e), verbose=verbose)
        result['error_type'] = error_type or 'transpile_error'
        result['error_detail'] = detail or str(e)[:200]
        return result
    except Exception as e:
        result['transpile_time'] = (datetime.now() - start).total_seconds()
        result['error_type'] = 'exception'
        result['error_detail'] = str(e)[:200]
        return result

    # Step 2: Run with pyne
    output_file = SCRIPTS_DIR / f'{name}.py'
    start = datetime.now()
    try:
        proc = subprocess.run(
            [pyne_bin, '-w', str(WORKDIR), 'run',
             str(output_file), str(data_file)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        result['runtime_time'] = (datetime.now() - start).total_seconds()

        if proc.returncode == 0:
            result['runtime'] = True
        else:
            error_type, detail = classify_error(proc.stderr, verbose=verbose)
            result['error_type'] = error_type
            result['error_detail'] = detail
            result['error_stderr'] = proc.stderr[-500:] if proc.stderr else ''

    except subprocess.TimeoutExpired:
        result['error_type'] = 'timeout'
        result['runtime_time'] = float(timeout)

    return result


def print_summary(results, total, skipped=0, skip_reasons=None):
    """Print final summary statistics."""
    transpile_ok = sum(1 for r in results if r['transpile'])
    transpile_fail = total - transpile_ok
    runtime_ok = sum(1 for r in results if r['runtime'])
    runtime_fail = sum(1 for r in results if r['transpile'] and not r['runtime'])

    error_counts = defaultdict(int)
    missing_imports = defaultdict(int)
    missing_apis = defaultdict(int)

    for r in results:
        if r['error_type']:
            error_counts[r['error_type']] += 1
            if r['error_type'] == 'missing_import':
                missing_imports[r['error_detail']] += 1
            elif r['error_type'] == 'pynecore_api':
                missing_apis[r['error_detail']] += 1

    print("\n" + "=" * 80)
    print("FINAL STATISTICS (sample_pds)")
    print("=" * 80)

    print(f"\nFiles:")
    print(f"  Total files:             {total + skipped}")
    if skipped > 0:
        skip_keywords = ', '.join(f'{kw}({cnt})' for kw, cnt in sorted(skip_reasons.items(), key=lambda x: -x[1])) if skip_reasons else 'unknown'
        print(f"  Skipped:                {skipped:4d} ({skip_keywords})")
    print(f"  Tested:                  {total:4d}")

    print(f"\nTranspilation:")
    print(f"  Successfully transpiled: {transpile_ok:4d} ({transpile_ok/total*100:5.1f}%)")
    print(f"  Failed to transpile:     {transpile_fail:4d} ({transpile_fail/total*100:5.1f}%)")

    print(f"\nRuntime (pyne run):")
    print(f"  Successfully ran:        {runtime_ok:4d} ({runtime_ok/total*100:5.1f}%)")
    print(f"  Failed to run:           {runtime_fail:4d} ({runtime_fail/total*100:5.1f}%)")
    print(f"  Not tested (no .py):     {transpile_fail:4d} ({transpile_fail/total*100:5.1f}%)")

    if error_counts:
        print(f"\nError Classification:")
        for error_type, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            print(f"  {error_type:20s}: {count:4d} ({count/total*100:4.1f}%)")

    if missing_imports:
        print(f"\nMissing Imports (Top 10):")
        for name, count in sorted(missing_imports.items(), key=lambda x: -x[1])[:10]:
            print(f"  {name:30s}: {count:4d}")

    if missing_apis:
        print(f"\nPyneCore API Gaps (Top 10):")
        for name, count in sorted(missing_apis.items(), key=lambda x: -x[1])[:10]:
            print(f"  {name:30s}: {count:4d}")

    return error_counts, missing_imports, missing_apis


def save_reports(results, total, error_counts, missing_imports, missing_apis, skipped=0):
    """Save JSON and text reports."""
    transpile_ok = sum(1 for r in results if r['transpile'])
    transpile_fail = total - transpile_ok
    runtime_ok = sum(1 for r in results if r['runtime'])
    runtime_fail = sum(1 for r in results if r['transpile'] and not r['runtime'])

    failed_files = [r['filename'] for r in results if not r['runtime']]

    run_entry = {
        'timestamp': datetime.now().isoformat(),
        'statistics': {
            'total': total,
            'skipped': skipped,
            'transpile_success': transpile_ok,
            'transpile_failed': transpile_fail,
            'runtime_success': runtime_ok,
            'runtime_failed': runtime_fail,
        },
        'error_counts': dict(error_counts),
        'missing_imports': dict(missing_imports),
        'missing_apis': dict(missing_apis),
        'failed_files': failed_files,
        'results': results,
    }

    # Backup existing results before overwriting
    if RESULTS_JSON.exists():
        try:
            shutil.copy2(RESULTS_JSON, RESULTS_BAK)
        except OSError:
            pass

    with RESULTS_JSON.open('w') as f:
        json.dump(run_entry, f, indent=2)

    # Text summary - keep only last 5 runs
    new_entry = []
    new_entry.append(f"\n{'=' * 70}\n")
    new_entry.append(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    new_entry.append(f"Total: {total}  Transpile OK: {transpile_ok}  Runtime OK: {runtime_ok}\n\n")
    for r in results:
        status = "PASS" if r['runtime'] else ("TRANS" if r['transpile'] else "FAIL")
        err = r['error_type'] or ''
        new_entry.append(f"{status:5s} {r['name']:55s} {err}\n")

    existing_runs = []
    if RESULTS_TXT.exists():
        try:
            content = RESULTS_TXT.read_text()
            runs = content.split('\n' + '=' * 70 + '\n')
            existing_runs = ['\n' + '=' * 70 + '\n' + run for run in runs if run.strip()][-4:]
        except OSError:
            pass

    with RESULTS_TXT.open('w') as f:
        f.writelines(existing_runs)
        f.writelines(new_entry)

    print(f"\nReports saved:")
    print(f"  {RESULTS_JSON}")
    print(f"  {RESULTS_BAK} (previous run backup)")
    print(f"  {RESULTS_TXT} (appended)")


def _print_result(idx, total, result, verbose=False):
    """Print a single test result line."""
    name = result['filename']
    if result['runtime']:
        print(f"[{idx:3d}/{total}] {name:55s} OK  transpile:{result['transpile_time']:.2f}s  run:{result['runtime_time']:.2f}s")
    elif result['transpile']:
        print(f"[{idx:3d}/{total}] {name:55s} RUNTIME FAIL  [{result['error_type']}] {result['error_detail'][:40]}")
        if verbose and result['error_stderr']:
            for line in result['error_stderr'].strip().splitlines()[-5:]:
                print(f"        {line}")
    else:
        print(f"[{idx:3d}/{total}] {name:55s} TRANSPILE FAIL  [{result['error_type']}] {result['error_detail'][:40]}")


def _run_sequential(testable_files, pyne_bin, data_file, timeout, verbose):
    """Run tests sequentially."""
    results = []
    total = len(testable_files)

    for idx, pine_file in enumerate(testable_files, 1):
        print(f"[{idx:3d}/{total}] {pine_file.name:55s} ", end='', flush=True)

        result = test_file(pine_file, pyne_bin, data_file, timeout=timeout, verbose=verbose)
        results.append(result)

        if result['runtime']:
            print(f"OK  transpile:{result['transpile_time']:.2f}s  run:{result['runtime_time']:.2f}s")
        elif result['transpile']:
            print(f"RUNTIME FAIL  [{result['error_type']}] {result['error_detail'][:40]}")
            if verbose and result['error_stderr']:
                for line in result['error_stderr'].strip().splitlines()[-5:]:
                    print(f"        {line}")
        else:
            print(f"TRANSPILE FAIL  [{result['error_type']}] {result['error_detail'][:40]}")

        if idx % 50 == 0:
            ok_so_far = sum(1 for r in results if r['runtime'])
            print(f"\n--- {idx}/{total} done, {ok_so_far} passing ---\n")

    return results


def _run_parallel(testable_files, pyne_bin, data_file, timeout, verbose, jobs):
    """Run tests in parallel using ProcessPoolExecutor."""
    total = len(testable_files)
    results = []
    completed = 0

    print(f"Running {total} tests across {jobs} workers...\n")

    with ProcessPoolExecutor(max_workers=jobs) as executor:
        future_to_file = {
            executor.submit(test_file, pine_file, pyne_bin, data_file, timeout, verbose): pine_file
            for pine_file in testable_files
        }

        for future in as_completed(future_to_file):
            completed += 1
            result = future.result()
            results.append(result)
            _print_result(completed, total, result, verbose)

            if completed % 50 == 0:
                ok_so_far = sum(1 for r in results if r['runtime'])
                print(f"\n--- {completed}/{total} done, {ok_so_far} passing ---\n")

    # Sort results by filename for consistent reporting
    results.sort(key=lambda r: r['filename'])
    return results


def main():
    parser = argparse.ArgumentParser(description='Test sample_pds Pine Script samples: transpile + pyne run')
    parser.add_argument('pattern', nargs='?', default='*.pine',
                        help='Glob pattern to filter .pine files (default: *.pine)')
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA,
                        help='OHLCV data file for pyne run')
    parser.add_argument('--timeout', type=int, default=15,
                        help='Timeout per file in seconds (default: 15)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show stderr output on failures')
    parser.add_argument('--retry-failed', '--failed-only', action='store_true',
                        help='Only run files that failed in the last test run')
    parser.add_argument('-j', '--jobs', type=int, default=24,
                        help='Number of parallel workers (default: 24, 0 = all cores, 1 = sequential)')
    args = parser.parse_args()

    # Resolve jobs=0 to actual CPU count
    if args.jobs == 0:
        args.jobs = os.cpu_count() or 1
    elif args.jobs < 0:
        print("ERROR: --jobs must be >= 0", file=sys.stderr)
        return 1

    pattern = args.pattern
    if not pattern.endswith('.pine'):
        pattern += '.pine' if '*' in pattern or '?' in pattern else '*.pine'

    pyne_bin = find_pyne()

    if not SAMPLE_DIR.is_dir():
        print(f"ERROR: Sample directory not found: {SAMPLE_DIR}", file=sys.stderr)
        return 1
    if not args.data.is_file():
        print(f"ERROR: Data file not found: {args.data}", file=sys.stderr)
        return 1
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    pine_files = sorted(SAMPLE_DIR.glob(pattern))
    if not pine_files:
        print(f"No .pine files matching '{pattern}' in {SAMPLE_DIR}")
        return 1

    if args.retry_failed:
        failed_files = load_failed_files()
        if failed_files is None:
            print("ERROR: No previous test results found. Run tests first without --retry-failed.", file=sys.stderr)
            return 1
        if failed_files == 'missing_key':
            print("ERROR: Previous test results are in old format (missing 'failed_files' key).", file=sys.stderr)
            print("Please run tests once without --retry-failed to update the format.", file=sys.stderr)
            return 1
        if not failed_files:
            print("No failed files in last run. All tests passed!")
            return 0

        original_count = len(pine_files)
        pine_files = [f for f in pine_files if f.name in failed_files]

        if not pine_files:
            print(f"No failed files match pattern '{pattern}'. Original failures: {len(failed_files)}")
            return 1

        print(f"Retry mode: {len(pine_files)} failed files (out of {original_count} matching pattern)")

    total = len(pine_files)

    print("=" * 80)
    print("PINE SCRIPT SAMPLE_PDS TEST")
    if args.retry_failed:
        print(" >>> RETRY FAILED MODE <<<")
    print("=" * 80)
    print(f"  Samples:  {SAMPLE_DIR}")
    print(f"  Output:   {SCRIPTS_DIR}")
    print(f"  Data:     {args.data}")
    print(f"  Pyne:     {pyne_bin}")
    print(f"  Files:    {total}  (pattern: {pattern})")
    if args.retry_failed:
        print(f"  Mode:     Retry failed files only")
    print(f"  Timeout:  {args.timeout}s")
    print(f"  Jobs:     {args.jobs} {'(sequential)' if args.jobs == 1 else '(parallel)'}")
    print(f"  Started:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()

    run_start = datetime.now()

    # Pre-filter skipped files
    skipped = 0
    skip_reasons = defaultdict(int)
    testable_files = []

    for pine_file in pine_files:
        skip, keyword = should_skip_file(pine_file)
        if skip:
            skipped += 1
            skip_reasons[keyword] += 1
            print(f"  {pine_file.name:55s} SKIP (contains '{keyword}')")
        else:
            testable_files.append(pine_file)

    if skipped:
        print()

    if args.jobs == 1:
        results = _run_sequential(testable_files, pyne_bin, args.data, args.timeout, args.verbose)
    else:
        results = _run_parallel(testable_files, pyne_bin, args.data, args.timeout, args.verbose, args.jobs)

    total_tested = len(results)
    error_counts, missing_imports, missing_apis = print_summary(results, total_tested, skipped, skip_reasons)
    save_reports(results, total_tested, error_counts, missing_imports, missing_apis, skipped)

    elapsed = (datetime.now() - run_start).total_seconds()
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ({elapsed:.1f}s elapsed)")

    runtime_ok = sum(1 for r in results if r['runtime'])
    return 0 if runtime_ok == total_tested else 1


if __name__ == '__main__':
    sys.exit(main())
