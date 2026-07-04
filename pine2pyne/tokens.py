"""
Token type definitions for Pine Script v6 lexer.
"""
from enum import Enum, auto
from dataclasses import dataclass
from typing import Any, Optional


class TokenType(Enum):
    """All token types supported by the Pine Script v6 lexer."""

    # Literals
    INT_LITERAL = auto()
    FLOAT_LITERAL = auto()
    STRING_LITERAL = auto()
    BOOL_LITERAL = auto()
    COLOR_LITERAL = auto()
    NA_LITERAL = auto()

    # Identifiers and types
    IDENTIFIER = auto()
    TYPE_IDENTIFIER = auto()

    # Keywords
    VAR = auto()
    VARIP = auto()
    IF = auto()
    ELSE = auto()
    FOR = auto()
    TO = auto()
    BY = auto()
    IN = auto()
    WHILE = auto()
    SWITCH = auto()
    BREAK = auto()
    CONTINUE = auto()
    IMPORT = auto()
    EXPORT = auto()
    METHOD = auto()
    TYPE = auto()
    ENUM = auto()
    NOT = auto()
    AND = auto()
    OR = auto()
    FUNCTION = auto()
    TRUE = auto()
    FALSE = auto()

    # Operators
    ASSIGN = auto()          # =
    REASSIGN = auto()        # :=
    PLUS_ASSIGN = auto()     # +=
    MINUS_ASSIGN = auto()    # -=
    MULT_ASSIGN = auto()     # *=
    DIV_ASSIGN = auto()      # /=
    MOD_ASSIGN = auto()      # %=
    PLUS = auto()            # +
    MINUS = auto()           # -
    MULT = auto()            # *
    DIV = auto()             # /
    MOD = auto()             # %
    GT = auto()              # >
    LT = auto()              # <
    GTE = auto()             # >=
    LTE = auto()             # <=
    EQ = auto()              # ==
    NEQ = auto()             # !=
    ARROW = auto()           # =>
    TERNARY = auto()         # ?
    COLON = auto()           # :
    DOT = auto()             # .

    # Delimiters
    LPAREN = auto()          # (
    RPAREN = auto()          # )
    LBRACKET = auto()        # [
    RBRACKET = auto()        # ]
    COMMA = auto()           # ,
    NEWLINE = auto()         # \n
    INDENT = auto()          # Increase indentation
    DEDENT = auto()          # Decrease indentation
    EOF = auto()             # End of file

    # Special
    COMMENT = auto()         # // comment
    VERSION_ANNOTATION = auto()     # //@version=6
    COMPILER_ANNOTATION = auto()    # //@description, //@function, etc.


# Keywords mapping
KEYWORDS = {
    'var': TokenType.VAR,
    'varip': TokenType.VARIP,
    'if': TokenType.IF,
    'else': TokenType.ELSE,
    'for': TokenType.FOR,
    'to': TokenType.TO,
    'by': TokenType.BY,
    'in': TokenType.IN,
    'while': TokenType.WHILE,
    'switch': TokenType.SWITCH,
    'break': TokenType.BREAK,
    'continue': TokenType.CONTINUE,
    'import': TokenType.IMPORT,
    'export': TokenType.EXPORT,
    'method': TokenType.METHOD,
    'type': TokenType.TYPE,
    'enum': TokenType.ENUM,
    'not': TokenType.NOT,
    'and': TokenType.AND,
    'or': TokenType.OR,
    'function': TokenType.FUNCTION,
    'true': TokenType.TRUE,
    'false': TokenType.FALSE,
    'na': TokenType.NA_LITERAL,
}

# Type identifiers (Pine Script built-in types)
TYPE_KEYWORDS = {
    'int', 'float', 'bool', 'string', 'color',
    'line', 'label', 'box', 'table', 'linefill', 'polyline',
    'array', 'matrix', 'map',
    'series', 'simple', 'input', 'const',
}


@dataclass
class Token:
    """Represents a single token from the Pine Script source."""
    type: TokenType
    value: Any
    line: int
    column: int
    lexeme: Optional[str] = None
    is_double_quoted: bool = False  # Track if string literal was double-quoted in source

    def __repr__(self) -> str:
        if self.lexeme:
            return f"Token({self.type.name}, {self.value!r}, {self.line}:{self.column}, {self.lexeme!r})"
        return f"Token({self.type.name}, {self.value!r}, {self.line}:{self.column})"

    def __str__(self) -> str:
        return self.__repr__()
