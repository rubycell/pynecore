#!/usr/bin/env python3
"""
Compare current sample outputs against ground truth.

Re-transpiles and runs all samples, then diffs output files against
ground_truth/{suite}/{sample_name}/ to detect regressions.

Usage:
    python compare_ground_truth.py                     # Run all
    python compare_ground_truth.py --suite sample_pds  # Only PDS suite
    python compare_ground_truth.py -j 12               # 12 parallel workers
"""

import argparse
import filecmp
import json
import re
import shutil
import subprocess
import sys
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


# Patterns for non-deterministic content that should be normalized before comparison
NORMALIZE_PATTERNS = [
    # Python object memory addresses: 0x7f1234abcdef
    (re.compile(r'0x[0-9a-f]{8,16}'), '0xNORMALIZED'),
]


def files_match(file_a: Path, file_b: Path) -> bool:
    """Compare two files, normalizing non-deterministic content.

    First tries fast byte-level comparison. Falls back to content-aware
    comparison that strips Python memory addresses and similar noise.
    """
    if filecmp.cmp(file_a, file_b, shallow=False):
        return True

    # Fast path failed — try normalized comparison
    try:
        content_a = file_a.read_text(encoding='utf-8')
        content_b = file_b.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        return False

    for pattern, replacement in NORMALIZE_PATTERNS:
        content_a = pattern.sub(replacement, content_a)
        content_b = pattern.sub(replacement, content_b)

    return content_a == content_b


def find_pyne():
    """Find the pyne binary."""
    if PYNE_BIN.exists():
        return str(PYNE_BIN)
    found = shutil.which('pyne')
    if found:
        return found
    print("ERROR: pyne not found.", file=sys.stderr)
    sys.exit(1)


def compare_sample(pine_file, suite_name, pyne_bin, data_file, timeout=20):
    """Transpile, run, and compare output against ground truth for one sample."""
    name = pine_file.stem
    ground_truth_sample_dir = GROUND_TRUTH_DIR / suite_name / name
    result = {
        'name': name,
        'suite': suite_name,
        'filename': pine_file.name,
        'status': 'unknown',
        'prev_status': 'unknown',
        'error': '',
        'diff_files': [],
        'missing_files': [],
        'new_files': [],
    }

    # Determine previous status from ground truth
    had_ground_truth = ground_truth_sample_dir.exists() and any(ground_truth_sample_dir.iterdir())
    result['prev_status'] = 'ok' if had_ground_truth else 'fail'

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

    # Step 3: Compare output files against ground truth
    if not had_ground_truth:
        # Previously failed, now passing — this is a NEW fix
        result['status'] = 'new_pass'
        new_files = []
        for suffix in ['.csv', '_strat.csv', '_trade.csv', '_plot_meta.json']:
            src = OUTPUT_DIR / f'{name}{suffix}'
            if src.exists():
                new_files.append(src.name)
        result['new_files'] = new_files
        return result

    # Compare each expected output file
    diff_files = []
    missing_files = []
    for gt_file in sorted(ground_truth_sample_dir.iterdir()):
        current_file = OUTPUT_DIR / gt_file.name
        if not current_file.exists():
            missing_files.append(gt_file.name)
        elif not files_match(gt_file, current_file):
            diff_files.append(gt_file.name)

    result['diff_files'] = diff_files
    result['missing_files'] = missing_files
    result['status'] = 'ok' if not diff_files and not missing_files else 'regression'
    return result


def run_suite(suite_name, suite_dir, pyne_bin, data_file, timeout, jobs):
    """Run and compare all samples in a suite."""
    pine_files = sorted(suite_dir.glob('*.pine'))
    if not pine_files:
        print(f"  No .pine files found in {suite_dir}")
        return []

    testable = []
    skipped = 0
    for pine_file in pine_files:
        skip, _ = should_skip_file(pine_file)
        if skip:
            skipped += 1
        else:
            testable.append(pine_file)

    total = len(testable)
    print(f"  {suite_name}: {total} testable, {skipped} skipped")

    results = []
    if jobs == 1:
        for index, pine_file in enumerate(testable, 1):
            result = compare_sample(pine_file, suite_name, pyne_bin, data_file, timeout)
            tag = format_status(result)
            print(f"  [{index:4d}/{total}] {pine_file.name:55s} {tag}")
            results.append(result)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            future_to_file = {
                executor.submit(compare_sample, pf, suite_name, pyne_bin, data_file, timeout): pf
                for pf in testable
            }
            done_count = 0
            for future in as_completed(future_to_file):
                done_count += 1
                result = future.result()
                tag = format_status(result)
                print(f"  [{done_count:4d}/{total}] {result['filename']:55s} {tag}")
                results.append(result)

    return results


def format_status(result):
    """Format status with change indicator."""
    status = result['status']
    prev = result['prev_status']

    if status == 'ok':
        return 'OK'
    elif status == 'new_pass':
        return 'NEW PASS (was failing)'
    elif status == 'regression':
        files = ', '.join(result['diff_files'][:3])
        return f'REGRESSION: {files}'
    elif status == 'runtime_fail' and prev == 'fail':
        return 'FAIL (same as before)'
    elif status == 'runtime_fail' and prev == 'ok':
        return 'NEW FAIL (was passing!)'
    elif status == 'transpile_fail':
        return f'TRANSPILE FAIL'
    elif status == 'timeout':
        return 'TIMEOUT'
    else:
        return status.upper()


def main():
    parser = argparse.ArgumentParser(description='Compare sample outputs against ground truth')
    parser.add_argument('--suite', choices=['sample', 'sample_pds', 'both'], default='both')
    parser.add_argument('--data', type=Path, default=WORKDIR / 'data' / 'VN30F1M_15m.ohlcv')
    parser.add_argument('--timeout', type=int, default=20)
    parser.add_argument('-j', '--jobs', type=int, default=12)
    args = parser.parse_args()

    if not args.data.exists():
        print(f"ERROR: Data file not found: {args.data}", file=sys.stderr)
        return 1

    if not GROUND_TRUTH_DIR.exists():
        print("ERROR: ground_truth/ directory not found. Run generate_ground_truth.py first.", file=sys.stderr)
        return 1

    manifest_path = GROUND_TRUTH_DIR / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        print(f"Ground truth generated at: {manifest.get('generated_at', 'unknown')}")
        print(f"Ground truth had: {manifest.get('passed', '?')} passing, "
              f"{manifest.get('runtime_fail', '?')} runtime fails")

    pyne_bin = find_pyne()
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    suites_to_run = ['sample', 'sample_pds'] if args.suite == 'both' else [args.suite]

    print("=" * 80)
    print("GROUND TRUTH COMPARISON")
    print("=" * 80)
    print(f"  Data:     {args.data}")
    print(f"  Jobs:     {args.jobs}")
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
    new_pass = sum(1 for r in all_results if r['status'] == 'new_pass')
    regressions = [r for r in all_results if r['status'] == 'regression']
    new_fails = [r for r in all_results if r['status'] == 'runtime_fail' and r['prev_status'] == 'ok']
    same_fails = sum(1 for r in all_results if r['status'] == 'runtime_fail' and r['prev_status'] == 'fail')
    transpile_fails = sum(1 for r in all_results if r['status'] == 'transpile_fail')
    timeouts = sum(1 for r in all_results if r['status'] == 'timeout')

    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    print(f"  Total tested:       {len(all_results)}")
    print(f"  OK (identical):     {ok_count}")
    print(f"  NEW PASS (fixed):   {new_pass}")
    print(f"  REGRESSION:         {len(regressions)}")
    print(f"  NEW FAIL:           {len(new_fails)}")
    print(f"  Same fail as before:{same_fails}")
    print(f"  Transpile fail:     {transpile_fails}")
    print(f"  Timeout:            {timeouts}")
    print("=" * 80)

    if regressions:
        print(f"\n!!! REGRESSIONS DETECTED ({len(regressions)}) !!!")
        for r in sorted(regressions, key=lambda x: x['filename']):
            print(f"  {r['filename']}")
            for diff_file in r['diff_files']:
                print(f"    CHANGED: {diff_file}")
            for missing_file in r['missing_files']:
                print(f"    MISSING: {missing_file}")

    if new_fails:
        print(f"\n!!! NEW FAILURES ({len(new_fails)}) !!!")
        for r in sorted(new_fails, key=lambda x: x['filename']):
            error_short = r['error'].split('\n')[-1][:80] if r['error'] else ''
            print(f"  {r['filename']:55s} {error_short}")

    if new_pass:
        print(f"\n+++ NEWLY PASSING ({new_pass}) +++")
        for r in sorted(all_results, key=lambda x: x['filename']):
            if r['status'] == 'new_pass':
                print(f"  {r['filename']}")

    # Exit code: 0 if no regressions or new failures
    if regressions or new_fails:
        print("\nFAILED: Regressions or new failures detected.")
        return 1

    print("\nPASSED: No regressions. All previously passing samples produce identical output.")
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
