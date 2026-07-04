"""
pine2pyne: Pine Script v6 to PyneCore Python transpiler.

A tool to convert TradingView Pine Script indicators and strategies
to PyneCore Python code for local backtesting and analysis.
"""

__version__ = '0.1.0'
__author__ = 'Pine2Pyne Transpiler'

from .lexer import Lexer, tokenize
from .parser import Parser, parse
from .transformer import Transformer, PyneTransformedScript
from .codegen import CodeGenerator, generate_code
from .errors import (
    Pine2PyneError,
    LexerError,
    ParserError,
    TransformerError,
    CodeGenError,
    UnsupportedFeatureError,
)


def _is_comment_only(source: str) -> bool:
    """Check if source contains only comments and whitespace."""
    for line in source.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith('//'):
            return False
    return True


def transpile(source: str) -> str:
    """
    Transpile Pine Script source code to PyneCore Python.

    Args:
        source: Pine Script v6 source code

    Returns:
        PyneCore-compatible Python source code (empty string if source is comment-only)

    Raises:
        Pine2PyneError: If any stage of transpilation fails
    """
    # Skip empty or comment-only files
    if not source or _is_comment_only(source):
        return ""

    # Tokenize
    lexer = Lexer(source)
    tokens = lexer.tokenize()

    # Parse
    parser = Parser(tokens)
    ast = parser.parse()

    # Transform
    transformer = Transformer()
    transformed = transformer.transform(ast)

    # Generate code
    codegen = CodeGenerator()
    output = codegen.generate(transformed)

    return output


__all__ = [
    'transpile',
    'Lexer',
    'Parser',
    'Transformer',
    'CodeGenerator',
    'tokenize',
    'parse',
    'generate_code',
    'Pine2PyneError',
    'LexerError',
    'ParserError',
    'TransformerError',
    'CodeGenError',
    'UnsupportedFeatureError',
    'PyneTransformedScript',
]
