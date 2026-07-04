"""
Custom error types for the Pine Script transpiler with source location tracking.
"""


class Pine2PyneError(Exception):
    """Base exception for all transpiler errors."""
    def __init__(self, message: str, line: int = 0, column: int = 0, source_file: str = None):
        self.message = message
        self.line = line
        self.column = column
        self.source_file = source_file
        super().__init__(self.format_message())

    def format_message(self) -> str:
        """Format error message with location information."""
        location = f"{self.line}:{self.column}"
        if self.source_file:
            location = f"{self.source_file}:{location}"
        return f"Error at {location}: {self.message}"


class LexerError(Pine2PyneError):
    """Error during lexical analysis (tokenization)."""
    pass


class ParserError(Pine2PyneError):
    """Error during parsing (syntax error)."""
    pass


class TransformerError(Pine2PyneError):
    """Error during AST transformation."""
    pass


class CodeGenError(Pine2PyneError):
    """Error during code generation."""
    pass


class UnsupportedFeatureError(Pine2PyneError):
    """Feature not yet supported by the transpiler."""
    def __init__(self, feature: str, line: int = 0, column: int = 0):
        message = f"Unsupported feature: {feature}"
        super().__init__(message, line, column)
        self.feature = feature
