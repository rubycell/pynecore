"""
Lexer (tokenizer) for Pine Script v6.

Handles indentation-significant syntax, multi-line continuations,
and all Pine Script v6 token types.
"""
import re
from typing import List, Optional
from .tokens import Token, TokenType, KEYWORDS, TYPE_KEYWORDS


class LexerError(Exception):
    """Raised when lexer encounters an invalid token."""
    def __init__(self, message: str, line: int, column: int):
        super().__init__(f"Lexer error at {line}:{column}: {message}")
        self.line = line
        self.column = column


class Lexer:
    """Tokenizes Pine Script v6 source code."""

    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens: List[Token] = []
        self.indent_stack: List[int] = [0]  # Track indentation levels
        self.paren_depth = 0  # Track depth inside (), [], {} for indentation-free zones

    def current_char(self) -> Optional[str]:
        """Get current character without advancing."""
        if self.pos >= len(self.source):
            return None
        return self.source[self.pos]

    def peek_char(self, offset: int = 1) -> Optional[str]:
        """Look ahead at character at pos + offset."""
        pos = self.pos + offset
        if pos >= len(self.source):
            return None
        return self.source[pos]

    def advance(self) -> Optional[str]:
        """Consume and return current character, updating position."""
        if self.pos >= len(self.source):
            return None
        char = self.source[self.pos]
        self.pos += 1
        if char == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        return char

    def skip_whitespace(self, skip_newlines: bool = False) -> None:
        """Skip spaces and tabs (and optionally newlines)."""
        while self.current_char() in (' ', '\t') or (skip_newlines and self.current_char() == '\n'):
            self.advance()

    def read_line_comment(self) -> Token:
        """Read // style comment until end of line."""
        start_line = self.line
        start_col = self.column

        # Check for compiler annotations
        if self.peek_char() == '@':
            return self.read_annotation()

        # Regular comment
        comment = ''
        while self.current_char() and self.current_char() != '\n':
            comment += self.advance()

        return Token(TokenType.COMMENT, comment.strip(), start_line, start_col, comment)

    def read_annotation(self) -> Token:
        """Read //@annotation style compiler directive."""
        start_line = self.line
        start_col = self.column

        # Skip //
        self.advance()
        self.advance()
        # Skip @
        self.advance()

        # Read annotation name
        annotation = ''
        while self.current_char() and self.current_char() not in (' ', '\t', '\n', '='):
            annotation += self.advance()

        # Check for version annotation
        if annotation == 'version':
            self.skip_whitespace()
            if self.current_char() == '=':
                self.advance()
                self.skip_whitespace()
                version = ''
                while self.current_char() and self.current_char() != '\n':
                    version += self.advance()
                return Token(TokenType.VERSION_ANNOTATION, version.strip(), start_line, start_col)

        # Read rest of annotation
        value = ''
        while self.current_char() and self.current_char() != '\n':
            value += self.advance()

        return Token(TokenType.COMPILER_ANNOTATION, f"{annotation}{value}".strip(), start_line, start_col)

    def read_string(self, quote: str) -> Token:
        """Read string literal (single or double quoted)."""
        start_line = self.line
        start_col = self.column

        # Skip opening quote
        self.advance()

        string = ''
        while self.current_char() and self.current_char() != quote:
            char = self.current_char()
            if char == '\\':
                # Handle escape sequences — unescape to actual characters
                self.advance()
                next_char = self.current_char()
                if next_char == 'n':
                    string += '\n'
                    self.advance()
                elif next_char == 't':
                    string += '\t'
                    self.advance()
                elif next_char == 'r':
                    string += '\r'
                    self.advance()
                elif next_char in ('"', "'", '\\'):
                    string += next_char
                    self.advance()
                else:
                    string += char
            elif char == '\n':
                # Pine Script allows multiline strings with regular quotes
                string += self.advance()
            else:
                string += self.advance()

        if not self.current_char():
            raise LexerError(f"Unterminated string literal", start_line, start_col)

        # Skip closing quote
        self.advance()

        # Mark if double quoted (for preserving quote style in output)
        is_double_quoted = (quote == '"')
        return Token(TokenType.STRING_LITERAL, string, start_line, start_col,
                    f"{quote}{string}{quote}", is_double_quoted=is_double_quoted)

    def read_number(self) -> Token:
        """Read integer or float literal."""
        start_line = self.line
        start_col = self.column

        number = ''
        has_dot = False
        has_e = False

        while self.current_char():
            char = self.current_char()

            if char.isdigit():
                number += self.advance()
            elif char == '.' and not has_dot and not has_e:
                next_ch = self.peek_char()
                if next_ch and next_ch.isdigit():
                    # 100.5 — normal decimal
                    has_dot = True
                    number += self.advance()
                elif next_ch is None or not (next_ch.isalpha() or next_ch == '_' or next_ch == '.'):
                    # 100. — trailing dot float (not member access like 100.toString)
                    has_dot = True
                    number += self.advance()
                else:
                    break
            elif char in ('e', 'E') and not has_e:
                has_e = True
                number += self.advance()
                # Handle optional +/- after e
                if self.current_char() in ('+', '-'):
                    number += self.advance()
            else:
                break

        if has_dot or has_e:
            return Token(TokenType.FLOAT_LITERAL, float(number), start_line, start_col, number)
        else:
            return Token(TokenType.INT_LITERAL, int(number), start_line, start_col, number)

    def read_color_literal(self) -> Token:
        """Read #RRGGBB or #RRGGBBAA color literal."""
        start_line = self.line
        start_col = self.column

        color = self.advance()  # #

        # Read hex digits
        while self.current_char() and self.current_char() in '0123456789abcdefABCDEF':
            color += self.advance()

        # Validate length (6 or 8 hex digits after #)
        if len(color) not in (7, 9):
            raise LexerError(f"Invalid color literal: {color}", start_line, start_col)

        return Token(TokenType.COLOR_LITERAL, color, start_line, start_col, color)

    def read_identifier(self) -> Token:
        """Read identifier or keyword."""
        start_line = self.line
        start_col = self.column

        identifier = ''
        while self.current_char() and (self.current_char().isalnum() or self.current_char() in ('_', '.')):
            identifier += self.advance()

        # Check if it's a keyword
        if identifier in KEYWORDS:
            token_type = KEYWORDS[identifier]
            return Token(token_type, identifier, start_line, start_col, identifier)

        # Check if it's a type keyword
        if identifier in TYPE_KEYWORDS:
            return Token(TokenType.TYPE_IDENTIFIER, identifier, start_line, start_col, identifier)

        return Token(TokenType.IDENTIFIER, identifier, start_line, start_col, identifier)

    def _is_continuation_line(self) -> bool:
        """Check if this line is a continuation of the previous expression.

        Pine Script treats indented lines as continuations when:
        1. The line starts with a binary operator keyword (or/and)
        2. The previous line ended with an operator that needs a right operand (?, :, and, or, +, -, etc.)
        """
        # Check if line starts with a continuation keyword
        if self.current_char() and self.current_char().isalpha():
            word = ''
            temp_pos = self.pos
            while temp_pos < len(self.source) and self.source[temp_pos].isalpha():
                word += self.source[temp_pos]
                temp_pos += 1
            if word in ('or', 'and'):
                return True

        # Check if previous line ended with an operator that needs a right operand
        # These operators always indicate the expression continues on the next line
        CONTINUATION_END_TOKENS = {
            TokenType.TERNARY,   # ?
            TokenType.COLON,     # : (ternary false branch)
            TokenType.AND,       # and
            TokenType.OR,        # or
            TokenType.PLUS,      # +
            TokenType.MINUS,     # -
            TokenType.MULT,      # *
            TokenType.DIV,       # /
            TokenType.MOD,       # %
            TokenType.COMMA,     # ,
        }
        if self.tokens:
            last_token = self.tokens[-1]
            if last_token.type in CONTINUATION_END_TOKENS:
                return True

        return False

    def handle_indentation(self, indent_level: int) -> List[Token]:
        """
        Generate INDENT/DEDENT tokens based on indentation change.

        Uses lenient matching: if exact level not in stack, finds closest level.
        This handles real-world Pine Script files with varied indentation styles.
        """
        tokens = []
        current_level = self.indent_stack[-1]

        if indent_level > current_level:
            # Increase indentation
            self.indent_stack.append(indent_level)
            tokens.append(Token(TokenType.INDENT, None, self.line, self.column))
        elif indent_level < current_level:
            # Decrease indentation (may generate multiple DEDENT tokens)
            while self.indent_stack and self.indent_stack[-1] > indent_level:
                self.indent_stack.pop()
                tokens.append(Token(TokenType.DEDENT, None, self.line, self.column))

            # Lenient matching: if exact level not found, find closest match
            if not self.indent_stack:
                # Should never happen, but reset to base
                self.indent_stack.append(0)
            elif self.indent_stack[-1] != indent_level:
                # Find closest level at or below target
                # This handles cases where continuation lines use different indentation
                closest_level = max(level for level in self.indent_stack if level <= indent_level)

                # If we found a closer match, pop until we reach it
                while self.indent_stack and self.indent_stack[-1] > closest_level:
                    self.indent_stack.pop()
                    tokens.append(Token(TokenType.DEDENT, None, self.line, self.column))

                # If we need to add the new level
                if closest_level < indent_level:
                    self.indent_stack.append(indent_level)
                    tokens.append(Token(TokenType.INDENT, None, self.line, self.column))

        return tokens

    def tokenize(self) -> List[Token]:
        """Tokenize the entire source code."""
        self.tokens = []
        at_line_start = True

        while self.pos < len(self.source):
            char = self.current_char()

            # Handle newlines and indentation
            if char == '\n':
                # Skip empty lines and lines with only comments
                saved_pos = self.pos
                saved_line = self.line
                saved_col = self.column

                self.advance()  # Skip \n

                # Count indentation on next line
                indent_level = 0
                while self.current_char() in (' ', '\t'):
                    if self.current_char() == ' ':
                        indent_level += 1
                    else:  # tab
                        indent_level += 4
                    self.advance()

                # Check if line is empty or comment-only
                if self.current_char() in ('\n', None) or (self.current_char() == '/' and self.peek_char() == '/'):
                    # Empty line or comment line - skip indentation tracking
                    if self.current_char() is None:
                        break
                    continue

                # Check for line continuation: increased indent + starts with binary operator
                # In Pine Script, indented lines starting with or/and continue the previous expression
                if (self.paren_depth == 0
                     and indent_level > self.indent_stack[-1]
                     and self._is_continuation_line()):
                    # Suppress NEWLINE and INDENT — treat as seamless continuation
                    at_line_start = False
                    continue

                # Inside parentheses/brackets/braces: suppress NEWLINE and indentation
                # Multi-line expressions inside delimiters are seamless
                if self.paren_depth > 0:
                    at_line_start = False
                    continue

                # Emit NEWLINE token for previous line
                self.tokens.append(Token(TokenType.NEWLINE, '\n', saved_line, saved_col))

                # Handle indentation changes
                if self.paren_depth == 0:
                    indent_tokens = self.handle_indentation(indent_level)
                    self.tokens.extend(indent_tokens)

                at_line_start = False
                continue

            # Skip inline whitespace
            if char in (' ', '\t'):
                self.skip_whitespace()
                continue

            # Comments and annotations
            if char == '/' and self.peek_char() == '/':
                token = self.read_line_comment()
                # Don't add comments to token stream (or add them with a flag)
                # For now, skip comments except annotations
                if token.type in (TokenType.VERSION_ANNOTATION, TokenType.COMPILER_ANNOTATION):
                    self.tokens.append(token)
                continue

            # String literals
            if char in ('"', "'"):
                self.tokens.append(self.read_string(char))
                continue

            # Color literals
            if char == '#':
                self.tokens.append(self.read_color_literal())
                continue

            # Numbers
            if char.isdigit():
                self.tokens.append(self.read_number())
                continue

            # Identifiers and keywords
            if char.isalpha() or char == '_':
                self.tokens.append(self.read_identifier())
                continue

            # Operators and delimiters
            start_line = self.line
            start_col = self.column

            # Two-character operators
            if char == ':' and self.peek_char() == '=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.REASSIGN, ':=', start_line, start_col, ':='))
                continue

            # Compound assignment operators
            if char == '+' and self.peek_char() == '=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.PLUS_ASSIGN, '+=', start_line, start_col, '+='))
                continue

            if char == '-' and self.peek_char() == '=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.MINUS_ASSIGN, '-=', start_line, start_col, '-='))
                continue

            if char == '*' and self.peek_char() == '=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.MULT_ASSIGN, '*=', start_line, start_col, '*='))
                continue

            if char == '/' and self.peek_char() == '=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.DIV_ASSIGN, '/=', start_line, start_col, '/='))
                continue

            if char == '%' and self.peek_char() == '=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.MOD_ASSIGN, '%=', start_line, start_col, '%='))
                continue

            if char == '=' and self.peek_char() == '>':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.ARROW, '=>', start_line, start_col, '=>'))
                continue

            if char == '=' and self.peek_char() == '=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.EQ, '==', start_line, start_col, '=='))
                continue

            if char == '!' and self.peek_char() == '=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.NEQ, '!=', start_line, start_col, '!='))
                continue

            if char == '>' and self.peek_char() == '=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.GTE, '>=', start_line, start_col, '>='))
                continue

            if char == '<' and self.peek_char() == '=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.LTE, '<=', start_line, start_col, '<='))
                continue

            # Single-character operators and delimiters
            single_char_tokens = {
                '=': TokenType.ASSIGN,
                '+': TokenType.PLUS,
                '-': TokenType.MINUS,
                '*': TokenType.MULT,
                '/': TokenType.DIV,
                '%': TokenType.MOD,
                '>': TokenType.GT,
                '<': TokenType.LT,
                '?': TokenType.TERNARY,
                ':': TokenType.COLON,
                '.': TokenType.DOT,
                '(': TokenType.LPAREN,
                ')': TokenType.RPAREN,
                '[': TokenType.LBRACKET,
                ']': TokenType.RBRACKET,
                ',': TokenType.COMMA,
            }

            if char in single_char_tokens:
                token_type = single_char_tokens[char]

                # Track delimiter depth for indentation-free zones
                if char in ('(', '[', '{'):
                    self.paren_depth += 1
                elif char in (')', ']', '}'):
                    self.paren_depth = max(0, self.paren_depth - 1)

                self.advance()
                self.tokens.append(Token(token_type, char, start_line, start_col, char))
                continue

            # Unknown character
            raise LexerError(f"Unexpected character: {char!r}", self.line, self.column)

        # Emit final DEDENT tokens to return to base indentation
        while len(self.indent_stack) > 1:
            self.indent_stack.pop()
            self.tokens.append(Token(TokenType.DEDENT, None, self.line, self.column))

        # Add EOF token
        self.tokens.append(Token(TokenType.EOF, None, self.line, self.column))

        return self.tokens


def tokenize(source: str) -> List[Token]:
    """Convenience function to tokenize Pine Script source."""
    lexer = Lexer(source)
    return lexer.tokenize()
