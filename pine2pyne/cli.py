"""
Command-line interface for pine2pyne transpiler.
"""
import argparse
import sys
from pathlib import Path
from typing import List, Optional
from . import transpile, __version__
from .errors import Pine2PyneError


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog='pine2pyne',
        description='Transpile Pine Script v6 to PyneCore Python',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Output to stdout
  python -m pine2pyne strategy.pine

  # Output to file
  python -m pine2pyne strategy.pine -o strategy.py

  # Validate syntax only
  python -m pine2pyne strategy.pine --validate

  # Batch convert multiple files
  python -m pine2pyne strategy/*.pine -o output/
        '''
    )

    parser.add_argument(
        'input',
        nargs='+',
        type=str,
        help='Input Pine Script file(s) (.pine)'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output file or directory. If directory, outputs will have .py extension'
    )

    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate syntax only (parse without generating code)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s {__version__}'
    )

    return parser.parse_args(args)


def transpile_file(input_path: Path, output_path: Optional[Path], validate_only: bool, verbose: bool) -> bool:
    """
    Transpile a single Pine Script file.

    Returns:
        True if successful, False otherwise
    """
    try:
        # Read input file
        if verbose:
            print(f'Reading {input_path}...')

        source = input_path.read_text(encoding='utf-8')

        if validate_only:
            # Only parse to validate syntax
            from .lexer import Lexer
            from .parser import Parser

            if verbose:
                print('Tokenizing...')
            lexer = Lexer(source)
            tokens = lexer.tokenize()

            if verbose:
                print('Parsing...')
            parser = Parser(tokens)
            ast = parser.parse()

            print(f'✓ {input_path} - Valid syntax')
            return True

        # Full transpilation
        if verbose:
            print('Transpiling...')

        output_code = transpile(source)

        # Skip writing if output is empty (comment-only file)
        if not output_code:
            if verbose:
                print(f'Skipping {input_path} (comment-only file)')
            return True

        # Write output
        if output_path:
            if verbose:
                print(f'Writing to {output_path}...')
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(output_code, encoding='utf-8')
            print(f'✓ {input_path} → {output_path}')
        else:
            # Output to stdout
            print(output_code)

        return True

    except Pine2PyneError as e:
        print(f'✗ {input_path}: {e}', file=sys.stderr)
        return False
    except FileNotFoundError:
        print(f'✗ {input_path}: File not found', file=sys.stderr)
        return False
    except Exception as e:
        print(f'✗ {input_path}: Unexpected error: {e}', file=sys.stderr)
        if verbose:
            import traceback
            traceback.print_exc()
        return False


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for CLI."""
    args = parse_args(argv)

    # Resolve input files
    input_paths: List[Path] = []
    for input_pattern in args.input:
        path = Path(input_pattern)
        if path.is_file():
            input_paths.append(path)
        elif '*' in input_pattern or '?' in input_pattern:
            # Glob pattern
            from glob import glob
            for match in glob(input_pattern):
                input_paths.append(Path(match))
        else:
            print(f'Error: {input_pattern} is not a file or valid pattern', file=sys.stderr)
            return 1

    if not input_paths:
        print('Error: No input files found', file=sys.stderr)
        return 1

    # Determine output strategy
    output_is_dir = False
    output_base = None

    if args.output:
        output_path = Path(args.output)
        if output_path.is_dir() or (not output_path.exists() and len(input_paths) > 1):
            output_is_dir = True
            output_base = output_path
        elif len(input_paths) > 1:
            print('Error: Multiple input files require output to be a directory', file=sys.stderr)
            return 1
        else:
            output_base = output_path

    # Process files
    success_count = 0
    fail_count = 0

    for input_path in input_paths:
        # Determine output path
        if args.output is None:
            output_path = None  # stdout
        elif output_is_dir:
            output_name = input_path.stem + '.py'
            output_path = output_base / output_name
        else:
            output_path = output_base

        # Transpile
        if transpile_file(input_path, output_path, args.validate, args.verbose):
            success_count += 1
        else:
            fail_count += 1

    # Print summary for multiple files
    if len(input_paths) > 1:
        print(f'\nSummary: {success_count} succeeded, {fail_count} failed')

    return 0 if fail_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
