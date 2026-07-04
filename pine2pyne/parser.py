"""
Recursive descent parser for Pine Script v6.

Converts token stream from lexer into an Abstract Syntax Tree (AST).
"""
from typing import List, Optional, Union
from .tokens import Token, TokenType
from .ast_nodes import *
from .errors import ParserError


class Parser:
    """Recursive descent parser for Pine Script v6."""

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def current_token(self) -> Token:
        """Get current token without advancing."""
        if self.pos >= len(self.tokens):
            return self.tokens[-1]  # Return EOF
        return self.tokens[self.pos]

    def peek_token(self, offset: int = 1) -> Token:
        """Look ahead at token at pos + offset."""
        pos = self.pos + offset
        if pos >= len(self.tokens):
            return self.tokens[-1]  # Return EOF
        return self.tokens[pos]

    def advance(self) -> Token:
        """Consume and return current token."""
        token = self.current_token()
        if token.type != TokenType.EOF:
            self.pos += 1
        return token

    def expect(self, token_type: TokenType) -> Token:
        """Consume token of expected type or raise error."""
        token = self.current_token()
        if token.type != token_type:
            raise ParserError(
                f"Expected {token_type.name}, got {token.type.name}",
                token.line,
                token.column
            )
        return self.advance()

    def match(self, *token_types: TokenType) -> bool:
        """Check if current token matches any of the given types."""
        return self.current_token().type in token_types

    def skip_newlines(self) -> None:
        """Skip any NEWLINE tokens."""
        while self.match(TokenType.NEWLINE):
            self.advance()

    def _looks_like_generic_call(self) -> bool:
        """
        Check if the current position looks like a generic function call.
        Pattern: <type1, type2, ...>(args)
        This prevents `close < ema` from being parsed as generics.
        """
        saved_pos = self.pos
        try:
            # We're currently at <, skip it
            self.pos += 1

            # Skip through the generic parameters
            depth = 1
            while depth > 0 and self.pos < len(self.tokens):
                token = self.tokens[self.pos]
                if token.type == TokenType.LT:
                    depth += 1
                elif token.type == TokenType.GT:
                    depth -= 1
                    if depth == 0:
                        # Found closing >, check next token
                        self.pos += 1
                        if self.pos < len(self.tokens):
                            next_token = self.tokens[self.pos]
                            # Generic calls MUST be followed by ( for function call
                            self.pos = saved_pos
                            return next_token.type == TokenType.LPAREN
                        self.pos = saved_pos
                        return False
                elif token.type not in (TokenType.TYPE_IDENTIFIER, TokenType.IDENTIFIER,
                                        TokenType.COMMA, TokenType.LT, TokenType.GT):
                    # Only type names and commas are valid inside <...> generics.
                    # Any operator (and, or, /, +, etc.) means this is a comparison, not generics.
                    self.pos = saved_pos
                    return False
                self.pos += 1

            self.pos = saved_pos
            return False
        except Exception:
            self.pos = saved_pos
            return False

    def error(self, message: str) -> ParserError:
        """Create parser error at current position."""
        token = self.current_token()
        return ParserError(message, token.line, token.column)

    # ========================================================================
    # Top-level parsing
    # ========================================================================

    def parse(self) -> Script:
        """Parse entire Pine Script source."""
        script = Script()

        self.skip_newlines()

        # Parse version annotation
        if self.match(TokenType.VERSION_ANNOTATION):
            version_token = self.advance()
            script.version = VersionAnnotation(version=version_token.value)
            self.skip_newlines()

        # Parse script declaration (indicator/strategy/library)
        if self.match(TokenType.IDENTIFIER):
            if self.current_token().value in ('indicator', 'strategy', 'library'):
                script.script_decl = self.parse_script_declaration()
                self.skip_newlines()

        # Parse imports
        while self.match(TokenType.IMPORT):
            script.imports.append(self.parse_import())
            self.skip_newlines()

        # Parse global declarations and body
        # Track actual source line numbers for ordering and blank line detection
        while not self.match(TokenType.EOF):
            self.skip_newlines()

            if self.match(TokenType.EOF):
                break

            # Handle VERSION_ANNOTATION tokens that appear in the body
            # (e.g., when pre-version declarations like DEFAULT_PYRAMIDING = 6
            # appear before //@version=6 in the source)
            if self.match(TokenType.VERSION_ANNOTATION):
                if not script.version:
                    version_token = self.advance()
                    script.version = VersionAnnotation(version=version_token.value)
                else:
                    self.advance()  # Skip duplicate version annotations
                continue

            # Handle COMPILER_ANNOTATION tokens in the body
            if self.match(TokenType.COMPILER_ANNOTATION):
                self.advance()  # Skip compiler annotations
                continue

            # Handle strategy/indicator/library declarations that appear after
            # pre-declaration code (when the initial top-level check missed them)
            if (not script.script_decl and
                self.match(TokenType.IDENTIFIER) and
                self.current_token().value in ('indicator', 'strategy', 'library')):
                script.script_decl = self.parse_script_declaration()
                self.skip_newlines()
                continue

            # Record the starting line of each declaration/statement
            start_line = self.current_token().line

            # Check for function declarations
            if self.is_function_declaration():
                node = self.parse_function_declaration()
                node.line = start_line
                script.declarations.append(node)
            # Check for var/varip declarations
            elif self.match(TokenType.VAR, TokenType.VARIP):
                node = self.parse_var_declaration()
                node.line = start_line
                script.declarations.append(node)
            # Check for type declarations
            elif self.match(TokenType.TYPE):
                node = self.parse_type_declaration()
                node.line = start_line
                script.declarations.append(node)
            # Check for enum declarations
            elif self.match(TokenType.ENUM):
                node = self.parse_enum_declaration()
                node.line = start_line
                script.declarations.append(node)
            # Handle import statements that appear in the body (not just at top)
            elif self.match(TokenType.IMPORT):
                import_node = self.parse_import()
                script.imports.append(import_node)
            # Check for input declarations
            elif self.is_input_declaration():
                saved_pos = self.pos
                try:
                    node = self.parse_input_declaration()
                    node.line = start_line
                    script.declarations.append(node)
                except Exception:
                    # Not a simple input declaration (e.g., input.string(...) == "On")
                    # Fall back to parsing as a regular statement
                    self.pos = saved_pos
                    stmt = self.parse_statement()
                    if stmt:
                        stmt.line = start_line
                        script.body.append(stmt)
            # Otherwise, parse as statement
            else:
                stmt = self.parse_statement()
                if stmt:
                    stmt.line = start_line
                    script.body.append(stmt)

            self.skip_newlines()

        return script

    def parse_script_declaration(self) -> Union[IndicatorDecl, StrategyDecl, LibraryDecl]:
        """Parse indicator/strategy/library declaration."""
        func_name = self.advance().value  # indicator, strategy, or library

        self.expect(TokenType.LPAREN)
        self.skip_newlines()  # Handle newlines after opening paren

        # Parse title - can be positional string or keyword argument
        title = None
        kwargs = {}

        # Check if first argument is keyword (IDENTIFIER = value) or positional (STRING_LITERAL)
        if self.match(TokenType.STRING_LITERAL):
            # Positional title argument
            title = self.advance().value
        elif self.match(TokenType.IDENTIFIER):
            # Could be keyword argument like title = "..."
            key = self.advance().value
            self.expect(TokenType.ASSIGN)
            value = self.parse_expression()
            if key == 'title':
                # Extract title value
                if isinstance(value, Literal) and value.literal_type == 'string':
                    title = value.value
                else:
                    title = str(value)
            else:
                kwargs[key] = value

        # Parse remaining keyword arguments
        while not self.match(TokenType.RPAREN):
            self.skip_newlines()  # Handle newlines before parameters

            if self.match(TokenType.RPAREN):  # Check again after skipping newlines
                break

            if self.match(TokenType.COMMA):
                self.advance()
                self.skip_newlines()  # Handle newlines after comma
                continue

            # Parse kwarg: name = value, or skip positional args
            if (self.match(TokenType.IDENTIFIER, TokenType.TYPE_IDENTIFIER) and
                self.peek_token().type == TokenType.ASSIGN):
                key = self.advance().value
                self.expect(TokenType.ASSIGN)
                value = self.parse_expression()
                if key == 'title' and title is None:
                    if isinstance(value, Literal) and value.literal_type == 'string':
                        title = value.value
                    else:
                        title = str(value)
                else:
                    kwargs[key] = value
            else:
                # Positional argument (e.g., "", true, 100) — skip it
                self.parse_expression()

        self.expect(TokenType.RPAREN)

        # Use default title if none provided
        if title is None:
            title = f"Untitled {func_name}"

        if func_name == 'indicator':
            return IndicatorDecl(title=title, kwargs=kwargs)
        elif func_name == 'strategy':
            return StrategyDecl(title=title, kwargs=kwargs)
        else:
            return LibraryDecl(title=title, kwargs=kwargs)

    def parse_import(self) -> ImportDecl:
        """Parse import statement: import user/library/version as alias"""
        self.expect(TokenType.IMPORT)

        # Parse user/library/version as separate tokens joined by DIV (/)
        # e.g., doqkhanh/tafirstlib/2 → IDENTIFIER DIV IDENTIFIER DIV INT_LITERAL
        user = self.expect(TokenType.IDENTIFIER).value
        library = ''
        version = None

        if self.match(TokenType.DIV):
            self.advance()  # consume /
            library = self.expect(TokenType.IDENTIFIER).value

        if self.match(TokenType.DIV):
            self.advance()  # consume /
            # Version can be an integer or identifier
            if self.match(TokenType.INT_LITERAL):
                version = self.advance().value
            elif self.match(TokenType.IDENTIFIER):
                version = self.advance().value

        # Parse optional 'as alias'
        alias = None
        if self.match(TokenType.IDENTIFIER) and self.current_token().value == 'as':
            self.advance()
            alias = self.expect(TokenType.IDENTIFIER).value

        return ImportDecl(user=user, library=library, version=version, alias=alias)

    def is_function_declaration(self) -> bool:
        """Check if current position is a function declaration."""
        # Look for: [export] [method] name(params) =>
        saved_pos = self.pos

        # Skip export/method keywords
        if self.match(TokenType.EXPORT):
            self.pos += 1
        if self.match(TokenType.METHOD):
            self.pos += 1

        # Check for identifier followed by ( and eventually =>
        if not (self.match(TokenType.IDENTIFIER) and self.peek_token().type == TokenType.LPAREN):
            self.pos = saved_pos
            return False

        # Scan ahead to find => arrow (function declarations must have this)
        # Skip past the parameter list
        paren_depth = 0
        while self.pos < len(self.tokens):
            token = self.tokens[self.pos]
            if token.type == TokenType.LPAREN:
                paren_depth += 1
            elif token.type == TokenType.RPAREN:
                paren_depth -= 1
                if paren_depth == 0:
                    # Found closing paren, check next token for =>
                    self.pos += 1
                    if self.pos < len(self.tokens) and self.tokens[self.pos].type == TokenType.ARROW:
                        self.pos = saved_pos
                        return True
                    else:
                        self.pos = saved_pos
                        return False
            self.pos += 1

        self.pos = saved_pos
        return False

    def parse_function_declaration(self) -> FuncDecl:
        """Parse function declaration."""
        is_export = False
        is_method = False

        if self.match(TokenType.EXPORT):
            is_export = True
            self.advance()

        if self.match(TokenType.METHOD):
            is_method = True
            self.advance()

        name = self.expect(TokenType.IDENTIFIER).value

        # Parse parameters
        self.expect(TokenType.LPAREN)
        self.skip_newlines()  # Skip newlines after opening paren
        params = []

        while not self.match(TokenType.RPAREN):
            self.skip_newlines()  # Skip newlines before each parameter

            if self.match(TokenType.RPAREN):
                break

            if self.match(TokenType.COMMA):
                self.advance()
                continue

            # Parse parameter: [type] name [= default]
            # Type can be: simple (float), generic (map<K,V>), or array (int[]), or custom type (TradeInfo)
            type_hint = None
            if self.match(TokenType.TYPE_IDENTIFIER):
                # Look ahead to disambiguate type hint vs param name:
                # TYPE_IDENTIFIER + IDENTIFIER → type hint (e.g. "float x")
                # TYPE_IDENTIFIER + LT → generic type hint (e.g. "map<string, int> this")
                # TYPE_IDENTIFIER + RPAREN/COMMA/ASSIGN → param name (e.g. "array" as name)
                next_tok = self.peek_token()
                if next_tok.type in (TokenType.IDENTIFIER, TokenType.LT, TokenType.TYPE_IDENTIFIER):
                    type_hint = self.parse_generic_type()
                # else: fall through, treat TYPE_IDENTIFIER as param name
            elif self.match(TokenType.IDENTIFIER) and self.peek_token().type == TokenType.IDENTIFIER:
                # For method parameters like "TradeInfo this" or "int val"
                # First IDENTIFIER is the type, second is the parameter name
                type_hint = self.advance().value

            # Param name can be IDENTIFIER or TYPE_IDENTIFIER used as a name
            if self.match(TokenType.IDENTIFIER):
                param_name = self.advance().value
            elif self.match(TokenType.TYPE_IDENTIFIER):
                param_name = self.advance().value
            else:
                param_name = self.expect(TokenType.IDENTIFIER).value

            default = None
            if self.match(TokenType.ASSIGN):
                self.advance()
                default = self.parse_expression()

            params.append(Parameter(name=param_name, type_hint=type_hint, default=default))
            self.skip_newlines()  # Skip newlines after parameter

        self.expect(TokenType.RPAREN)

        # Parse => and body
        self.expect(TokenType.ARROW)
        self.skip_newlines()

        # Check if single-line or multi-line body
        if self.match(TokenType.INDENT):
            # Multi-line body
            self.advance()
            body = self.parse_block()
            self.expect(TokenType.DEDENT)
        else:
            # Single-line body — may have comma-separated statements
            # e.g., fn(x) => a = expr1, b = expr2, return_expr
            body = self._parse_single_line_body()

        return FuncDecl(name=name, params=params, body=body,
                        is_method=is_method, is_export=is_export)

    def _parse_single_line_body(self):
        """Parse single-line function body with optional comma-separated statements.
        Pine Script allows: fn(x) => stmt1, stmt2, return_expr
        Each comma-separated item can be an assignment or expression.
        Returns a single expression (no commas) or a list of statements (commas present).
        """
        # Parse first item — could be var declaration, assignment, or expression
        first = self._parse_single_line_item()

        # Check for comma (multiple statements)
        if not self.match(TokenType.COMMA):
            return first  # Single expression, return as-is

        # Multiple comma-separated statements
        items = [first]
        while self.match(TokenType.COMMA):
            self.advance()  # consume comma
            self.skip_newlines()
            items.append(self._parse_single_line_item())

        return items

    def _parse_single_line_item(self):
        """Parse a single item in a comma-separated single-line function body.
        Can be: var type name = expr, name = expr, or just expr.
        """
        # Handle 'var' declarations: var type name = expr
        if self.match(TokenType.VAR):
            return self.parse_statement()

        # Try assignment: name = expr (look ahead for IDENTIFIER followed by ASSIGN)
        saved_pos = self.pos
        if self.match(TokenType.TYPE_IDENTIFIER):
            # Could be typed assignment: float x = expr
            type_hint = self.parse_generic_type()
            if self.match(TokenType.IDENTIFIER):
                var_name = self.advance().value
                if self.match(TokenType.ASSIGN):
                    self.advance()
                    value = self.parse_expression()
                    return Assignment(target=var_name, value=value, type_hint=type_hint)
            self.pos = saved_pos

        if self.match(TokenType.IDENTIFIER):
            maybe_name = self.current_token().value
            next_pos = self.pos + 1
            if next_pos < len(self.tokens) and self.tokens[next_pos].type == TokenType.ASSIGN:
                var_name = self.advance().value
                self.advance()  # consume =
                value = self.parse_expression()
                return Assignment(target=var_name, value=value)
            # Not an assignment — fall through to expression parse
            self.pos = saved_pos

        return self.parse_expression()

    def parse_generic_type(self) -> str:
        """
        Parse type syntax:
        - Modifier + type: simple int, series float, const string, input float
        - Generic: type<T> or map<K, V>
        - Array bracket: type[] (e.g., string[], int[])
        """
        # Pine Script type modifiers (no Python equivalent, consume and drop)
        TYPE_MODIFIERS = {'simple', 'series', 'const', 'input'}

        base_type = self.advance().value  # Get base type (map, array, string, int, etc.)

        # Handle type modifiers: simple int, series float, etc.
        # The modifier is consumed but dropped — only the actual type is kept
        if base_type in TYPE_MODIFIERS and self.match(TokenType.TYPE_IDENTIFIER):
            base_type = self.advance().value

        # Check for array bracket syntax: type[]
        if self.match(TokenType.LBRACKET):
            self.advance()  # consume [
            self.expect(TokenType.RBRACKET)  # consume ]
            # Convert to generic array syntax: string[] -> array<string>
            return f"array<{base_type}>"

        # Check for generic parameters: type<T>
        if self.match(TokenType.LT):
            self.advance()  # consume <

            generic_params = []
            while not self.match(TokenType.GT):
                if self.match(TokenType.COMMA):
                    self.advance()
                    continue

                # Parse type parameter (can be TYPE_IDENTIFIER or IDENTIFIER)
                if self.match(TokenType.TYPE_IDENTIFIER, TokenType.IDENTIFIER):
                    generic_params.append(self.advance().value)
                else:
                    raise self.error(f"Expected type parameter, got {self.current_token().type.name}")

            self.expect(TokenType.GT)  # consume >

            # Format as Python generic: map<string, float> -> dict[str, float]
            return f"{base_type}<{', '.join(generic_params)}>"

        return base_type

    def parse_var_declaration(self) -> Union[VarDecl, VaripDecl]:
        """Parse var/varip declaration."""
        is_var = self.match(TokenType.VAR)
        self.advance()  # var or varip

        # Parse optional type hint (with generic support or UDT name)
        type_hint = None
        if self.match(TokenType.TYPE_IDENTIFIER):
            type_hint = self.parse_generic_type()
        elif self.match(TokenType.IDENTIFIER):
            # Could be a UDT type name like Statistics, TradeInfo
            next_type = self.peek_token().type
            if next_type == TokenType.IDENTIFIER:
                # Simple UDT: TradeInfo myVar = ...
                type_hint = self.advance().value
            elif next_type in (TokenType.LBRACKET, TokenType.LT):
                # UDT array/generic: TestResult[] results = ..., MyType<T> x = ...
                type_hint = self.advance().value  # UDT name
                if self.match(TokenType.LBRACKET):
                    self.advance()  # consume [
                    self.expect(TokenType.RBRACKET)  # consume ]
                    type_hint = f"array<{type_hint}>"
                elif self.match(TokenType.LT):
                    # Parse generic params
                    self.advance()  # consume <
                    generic_params = []
                    while not self.match(TokenType.GT):
                        if self.match(TokenType.COMMA):
                            self.advance()
                            continue
                        if self.match(TokenType.TYPE_IDENTIFIER, TokenType.IDENTIFIER):
                            generic_params.append(self.advance().value)
                        else:
                            raise self.error(f"Expected type parameter, got {self.current_token().type.name}")
                    self.expect(TokenType.GT)
                    type_hint = f"{type_hint}<{', '.join(generic_params)}>"

        # Parse variable name
        name = self.expect(TokenType.IDENTIFIER).value

        # Parse = value
        self.expect(TokenType.ASSIGN)
        value = self.parse_expression()

        if is_var:
            return VarDecl(name=name, value=value, type_hint=type_hint)
        else:
            return VaripDecl(name=name, value=value, type_hint=type_hint)

    def parse_type_declaration(self) -> TypeDecl:
        """Parse type (UDT) declaration."""
        self.expect(TokenType.TYPE)
        name = self.expect(TokenType.IDENTIFIER).value

        self.skip_newlines()
        self.expect(TokenType.INDENT)

        fields = []
        while not self.match(TokenType.DEDENT):
            self.skip_newlines()
            if self.match(TokenType.DEDENT):
                break

            # Parse field: [var|varip] type name [= default_value]
            # Type can be a built-in type (TYPE_IDENTIFIER), generic (array<T>), or UDT (IDENTIFIER)
            # varip behaves like var since we don't use realtime candles
            field_is_var = False
            if self.match(TokenType.VAR, TokenType.VARIP):
                self.advance()  # consume var/varip
                field_is_var = True
            if self.match(TokenType.TYPE_IDENTIFIER):
                type_hint = self.parse_generic_type()
            elif self.match(TokenType.IDENTIFIER):
                type_hint = self.advance().value
            else:
                raise self.error(f"Expected type, got {self.current_token().type.name}")
            field_name = self.expect(TokenType.IDENTIFIER).value

            # Handle optional default value
            default_value = None
            if self.match(TokenType.ASSIGN):
                self.advance()  # consume =
                default_value = self.parse_expression()

            fields.append((field_name, type_hint, default_value))
            self.skip_newlines()

        self.expect(TokenType.DEDENT)

        return TypeDecl(name=name, fields=fields)

    def parse_enum_declaration(self) -> EnumDecl:
        """Parse enum declaration."""
        self.expect(TokenType.ENUM)
        # Enum name can be IDENTIFIER or TYPE_IDENTIFIER (e.g., "enum polyline", "enum ta")
        if self.match(TokenType.IDENTIFIER):
            name = self.advance().value
        elif self.match(TokenType.TYPE_IDENTIFIER):
            name = self.advance().value
        else:
            name = self.expect(TokenType.IDENTIFIER).value

        self.skip_newlines()
        self.expect(TokenType.INDENT)

        members = []
        while not self.match(TokenType.DEDENT):
            self.skip_newlines()
            if self.match(TokenType.DEDENT):
                break

            # Parse enum member: name [= value]; can also be TYPE_IDENTIFIER
            if self.match(TokenType.IDENTIFIER):
                member = self.advance().value
            elif self.match(TokenType.TYPE_IDENTIFIER):
                member = self.advance().value
            else:
                member = self.expect(TokenType.IDENTIFIER).value

            # Check for optional value assignment
            if self.match(TokenType.ASSIGN):
                self.advance()  # consume =
                # Parse the value (can be string, int, etc.)
                value = self.parse_expression()
                # Store as tuple (name, value) or just name
                members.append((member, value))
            else:
                members.append(member)

            self.skip_newlines()

        self.expect(TokenType.DEDENT)

        return EnumDecl(name=name, members=members)

    def is_input_declaration(self) -> bool:
        """Check if current position is an input declaration.
        Matches: name = input.*() or type name = input.*()
        """
        saved_pos = self.pos

        # Skip optional type hint (e.g., float, int, string)
        if self.match(TokenType.TYPE_IDENTIFIER):
            self.pos += 1

        # Must have an identifier (the variable name)
        if not self.match(TokenType.IDENTIFIER):
            self.pos = saved_pos
            return False

        self.pos += 1  # Skip identifier

        # Check for = input.
        result = (self.match(TokenType.ASSIGN) and
                  self.peek_token().type == TokenType.IDENTIFIER and
                  self.peek_token().value.startswith('input.'))

        self.pos = saved_pos
        return result

    def parse_input_declaration(self) -> InputDecl:
        """Parse input declaration. Handles: name = input.*() and type name = input.*()"""
        # Skip optional type hint
        if self.match(TokenType.TYPE_IDENTIFIER):
            self.advance()

        name = self.advance().value
        self.expect(TokenType.ASSIGN)

        # Parse input.*() call
        func_call = self.parse_expression()

        if not isinstance(func_call, FunctionCall):
            raise self.error("Expected input function call")

        # Extract function name and arguments
        if isinstance(func_call.func, str):
            func = func_call.func
        elif isinstance(func_call.func, MemberAccess):
            func = f"{func_call.func.object}.{func_call.func.member}"
        else:
            func = "input"

        return InputDecl(name=name, func=func, args=func_call.args, kwargs=func_call.kwargs)

    # ========================================================================
    # Statement parsing
    # ========================================================================

    def parse_statement(self) -> Optional[Statement]:
        """Parse a single statement."""
        self.skip_newlines()

        # If statement
        if self.match(TokenType.IF):
            return self.parse_if_statement()

        # For loop
        if self.match(TokenType.FOR):
            return self.parse_for_loop()

        # While loop
        if self.match(TokenType.WHILE):
            return self.parse_while_loop()

        # Switch statement
        if self.match(TokenType.SWITCH):
            return self.parse_switch_statement()

        # Break
        if self.match(TokenType.BREAK):
            self.advance()
            return BreakStatement()

        # Continue
        if self.match(TokenType.CONTINUE):
            self.advance()
            return ContinueStatement()

        # Var/varip declarations (can appear inside blocks in Pine Script v6)
        if self.match(TokenType.VAR, TokenType.VARIP):
            return self.parse_var_declaration()

        # Local variable declaration with explicit type: type name = value
        # Examples: float x = 10.0, string[] ids = array.new_string()
        # Also handles user-defined types: Signal signalState = ...
        # But NOT type conversion calls like bool(na), float(x) — those are expressions
        if self.match(TokenType.TYPE_IDENTIFIER) and self.peek_token().type != TokenType.LPAREN:
            type_hint = self.parse_generic_type()
            var_name = self.expect(TokenType.IDENTIFIER).value
            self.expect(TokenType.ASSIGN)
            value = self.parse_expression()
            # Return as a local variable assignment with type hint
            return Assignment(target=var_name, value=value, type_hint=type_hint)

        # Check for user-defined type declaration: UDT varName = value
        # Heuristic: IDENTIFIER IDENTIFIER ASSIGN (e.g., Signal signalState =)
        if (self.match(TokenType.IDENTIFIER) and
            self.pos + 1 < len(self.tokens) and
            self.tokens[self.pos + 1].type == TokenType.IDENTIFIER and
            self.pos + 2 < len(self.tokens) and
            self.tokens[self.pos + 2].type == TokenType.ASSIGN):
            type_name = self.advance().value
            var_name = self.advance().value
            self.expect(TokenType.ASSIGN)
            value = self.parse_expression()
            return Assignment(target=var_name, value=value, type_hint=type_name)

        # Check for UDT array declaration: UDT[] varName = value
        # Heuristic: IDENTIFIER LBRACKET RBRACKET IDENTIFIER ASSIGN
        if (self.match(TokenType.IDENTIFIER) and
            self.pos + 1 < len(self.tokens) and
            self.tokens[self.pos + 1].type == TokenType.LBRACKET and
            self.pos + 2 < len(self.tokens) and
            self.tokens[self.pos + 2].type == TokenType.RBRACKET and
            self.pos + 3 < len(self.tokens) and
            self.tokens[self.pos + 3].type == TokenType.IDENTIFIER and
            self.pos + 4 < len(self.tokens) and
            self.tokens[self.pos + 4].type == TokenType.ASSIGN):
            udt_name = self.advance().value
            self.advance()  # consume [
            self.advance()  # consume ]
            var_name = self.advance().value
            self.expect(TokenType.ASSIGN)
            value = self.parse_expression()
            return Assignment(target=var_name, value=value, type_hint=f"array<{udt_name}>")

        # Check for nested function declarations: name(params) =>
        if self.is_function_declaration():
            return self.parse_function_declaration()

        # Assignment or expression statement
        expr = self.parse_expression()

        # Check for assignment/reassignment
        if self.match(TokenType.ASSIGN):
            self.advance()
            value = self.parse_expression()

            if isinstance(expr, Identifier):
                return Assignment(target=expr.name, value=value)
            elif isinstance(expr, ArrayLiteral):
                # Tuple destructuring
                names = [e.name if isinstance(e, Identifier) else str(e) for e in expr.elements]
                return Assignment(target=TupleDestructure(names=names, value=value), value=value)

        elif self.match(TokenType.REASSIGN):
            self.advance()
            value = self.parse_expression()

            if isinstance(expr, Identifier):
                return Reassignment(target=expr.name, value=value)

        # Compound assignment operators: +=, -=, *=, /=, %=
        elif self.match(TokenType.PLUS_ASSIGN, TokenType.MINUS_ASSIGN, TokenType.MULT_ASSIGN,
                        TokenType.DIV_ASSIGN, TokenType.MOD_ASSIGN):
            op_token = self.advance()
            right_value = self.parse_expression()

            if isinstance(expr, Identifier):
                # Convert x += 5 to x := x + 5
                # Determine the operator symbol
                op_map = {
                    TokenType.PLUS_ASSIGN: '+',
                    TokenType.MINUS_ASSIGN: '-',
                    TokenType.MULT_ASSIGN: '*',
                    TokenType.DIV_ASSIGN: '/',
                    TokenType.MOD_ASSIGN: '%',
                }
                op_symbol = op_map[op_token.type]

                # Create binary expression: x + 5
                from .ast_nodes import BinaryOp
                new_value = BinaryOp(left=expr, op=op_symbol, right=right_value)

                # Return reassignment: x := (x + 5)
                return Reassignment(target=expr.name, value=new_value)

        # Expression statement
        return ExpressionStatement(expr=expr)

    def parse_block(self) -> List[Statement]:
        """Parse a block of statements (indented)."""
        statements = []

        while not self.match(TokenType.DEDENT, TokenType.EOF):
            self.skip_newlines()
            if self.match(TokenType.DEDENT, TokenType.EOF):
                break

            stmt = self.parse_statement()
            if stmt:
                statements.append(stmt)

            # Handle comma-separated statements on the same line
            # e.g., colNum = 8, rowNum = 8
            while self.match(TokenType.COMMA):
                self.advance()  # consume comma
                stmt = self.parse_statement()
                if stmt:
                    statements.append(stmt)

        return statements

    def parse_if_statement(self) -> IfStatement:
        """Parse if statement."""
        self.expect(TokenType.IF)
        condition = self.parse_expression()

        self.skip_newlines()
        self.expect(TokenType.INDENT)
        body = self.parse_block()
        self.expect(TokenType.DEDENT)

        # Parse else if clauses
        elseifs = []
        while self.match(TokenType.ELSE):
            self.advance()
            if self.match(TokenType.IF):
                self.advance()
                elif_condition = self.parse_expression()
                self.skip_newlines()
                self.expect(TokenType.INDENT)
                elif_body = self.parse_block()
                self.expect(TokenType.DEDENT)
                elseifs.append((elif_condition, elif_body))
            else:
                # Else clause
                self.skip_newlines()
                self.expect(TokenType.INDENT)
                else_body = self.parse_block()
                self.expect(TokenType.DEDENT)
                return IfStatement(condition=condition, body=body, elseifs=elseifs, else_body=else_body)

        return IfStatement(condition=condition, body=body, elseifs=elseifs)

    def parse_for_loop(self) -> Union[ForLoop, ForInLoop]:
        """Parse for loop (range or for-in style)."""
        self.expect(TokenType.FOR)

        # Check for tuple destructuring [a, b] in ...
        if self.match(TokenType.LBRACKET):
            return self.parse_for_in_loop()

        # Skip optional type hint: for int i = ... or for float x in ...
        if self.match(TokenType.TYPE_IDENTIFIER):
            self.advance()  # consume type hint (int, float, etc.)

        # Check for regular 'for var in iterable'
        var_name = self.expect(TokenType.IDENTIFIER).value

        if self.match(TokenType.IN):
            # for-in loop
            self.advance()
            iterable = self.parse_expression()

            self.skip_newlines()
            self.expect(TokenType.INDENT)
            body = self.parse_block()
            self.expect(TokenType.DEDENT)

            return ForInLoop(vars=[var_name], iterable=iterable, body=body)

        # for var = from to to_val [by step]
        self.expect(TokenType.ASSIGN)
        from_val = self.parse_expression()

        self.expect(TokenType.TO)
        to_val = self.parse_expression()

        step = None
        if self.match(TokenType.BY):
            self.advance()
            step = self.parse_expression()

        self.skip_newlines()
        self.expect(TokenType.INDENT)
        body = self.parse_block()
        self.expect(TokenType.DEDENT)

        return ForLoop(var=var_name, from_val=from_val, to_val=to_val, step=step, body=body)

    def parse_for_in_loop(self) -> ForInLoop:
        """Parse for-in loop with tuple destructuring."""
        self.expect(TokenType.LBRACKET)

        vars = []
        while not self.match(TokenType.RBRACKET):
            if self.match(TokenType.COMMA):
                self.advance()
                continue

            vars.append(self.expect(TokenType.IDENTIFIER).value)

        self.expect(TokenType.RBRACKET)
        self.expect(TokenType.IN)

        iterable = self.parse_expression()

        self.skip_newlines()
        self.expect(TokenType.INDENT)
        body = self.parse_block()
        self.expect(TokenType.DEDENT)

        return ForInLoop(vars=vars, iterable=iterable, body=body)

    def parse_while_loop(self) -> WhileLoop:
        """Parse while loop."""
        self.expect(TokenType.WHILE)
        condition = self.parse_expression()

        self.skip_newlines()
        self.expect(TokenType.INDENT)
        body = self.parse_block()
        self.expect(TokenType.DEDENT)

        return WhileLoop(condition=condition, body=body)

    def parse_switch_statement(self) -> SwitchStatement:
        """Parse switch statement."""
        self.expect(TokenType.SWITCH)

        # Optional expression
        expr = None
        if not self.match(TokenType.NEWLINE):
            expr = self.parse_expression()

        self.skip_newlines()
        self.expect(TokenType.INDENT)

        cases = []
        default = None

        while not self.match(TokenType.DEDENT):
            self.skip_newlines()
            if self.match(TokenType.DEDENT):
                break

            # Check for default case (starts with =>)
            if self.match(TokenType.ARROW):
                self.advance()  # consume =>
                self.skip_newlines()
                if self.match(TokenType.INDENT):
                    self.advance()
                    default = self.parse_block()
                    self.expect(TokenType.DEDENT)
                else:
                    case_stmt = self.parse_statement()
                    default = [case_stmt] if case_stmt else []
                self.skip_newlines()
                continue

            # Parse case expression
            case_expr = self.parse_expression()

            self.expect(TokenType.ARROW)
            self.skip_newlines()

            # Parse case body
            if self.match(TokenType.INDENT):
                self.advance()
                case_body = self.parse_block()
                self.expect(TokenType.DEDENT)
            else:
                # Single line
                case_stmt = self.parse_statement()
                case_body = [case_stmt] if case_stmt else []

            cases.append((case_expr, case_body))

        self.expect(TokenType.DEDENT)

        return SwitchStatement(expr=expr, cases=cases, default=default)

    # ========================================================================
    # Expression parsing
    # ========================================================================

    def parse_expression(self) -> Expression:
        """Parse expression (entry point for expression parsing)."""
        return self.parse_ternary()

    def parse_ternary(self) -> Expression:
        """Parse ternary operator: cond ? true_expr : false_expr"""
        expr = self.parse_or()

        if self.match(TokenType.TERNARY):
            self.advance()
            true_expr = self.parse_ternary()
            self.expect(TokenType.COLON)
            false_expr = self.parse_ternary()
            return TernaryOp(condition=expr, true_expr=true_expr, false_expr=false_expr)

        return expr

    def parse_or(self) -> Expression:
        """Parse logical OR."""
        left = self.parse_and()

        while self.match(TokenType.OR):
            op = self.advance().value
            right = self.parse_and()
            left = BinaryOp(left=left, op=op, right=right)

        return left

    def parse_and(self) -> Expression:
        """Parse logical AND."""
        left = self.parse_equality()

        while self.match(TokenType.AND):
            op = self.advance().value
            right = self.parse_equality()
            left = BinaryOp(left=left, op=op, right=right)

        return left

    def parse_equality(self) -> Expression:
        """Parse == and != operators."""
        left = self.parse_comparison()

        while self.match(TokenType.EQ, TokenType.NEQ):
            op = self.advance().value
            right = self.parse_comparison()
            left = BinaryOp(left=left, op=op, right=right)

        return left

    def parse_comparison(self) -> Expression:
        """Parse <, >, <=, >= operators."""
        left = self.parse_addition()

        while self.match(TokenType.LT, TokenType.GT, TokenType.LTE, TokenType.GTE):
            op = self.advance().value
            right = self.parse_addition()
            left = BinaryOp(left=left, op=op, right=right)

        return left

    def parse_addition(self) -> Expression:
        """Parse + and - operators."""
        left = self.parse_multiplication()

        while self.match(TokenType.PLUS, TokenType.MINUS):
            op = self.advance().value
            right = self.parse_multiplication()
            left = BinaryOp(left=left, op=op, right=right)

        return left

    def parse_multiplication(self) -> Expression:
        """Parse *, /, % operators."""
        left = self.parse_unary()

        while self.match(TokenType.MULT, TokenType.DIV, TokenType.MOD):
            op = self.advance().value
            right = self.parse_unary()
            left = BinaryOp(left=left, op=op, right=right)

        return left

    def parse_unary(self) -> Expression:
        """Parse unary operators (-, not)."""
        if self.match(TokenType.MINUS, TokenType.NOT):
            op = self.advance().value
            operand = self.parse_unary()
            return UnaryOp(op=op, operand=operand)

        return self.parse_postfix()

    def parse_postfix(self) -> Expression:
        """Parse postfix operations (function calls, member access, indexing)."""
        expr = self.parse_primary()

        # Block-level expressions (switch/if) already consumed their DEDENT;
        # don't greedily attach postfix [ or . from the NEXT statement
        if isinstance(expr, (SwitchExpression, IfExpression)):
            return expr

        while True:
            # Function call (may have generic parameters)
            # Parse as generics only for function/method names, not after index access
            # This prevents `arr[0] < 5` and `close < ema` from being parsed as generics
            # Heuristic: generics MUST be followed by ( for function call
            if (self.match(TokenType.LT) and
                self.peek_token().type in (TokenType.TYPE_IDENTIFIER, TokenType.IDENTIFIER) and
                isinstance(expr, (Identifier, MemberAccess)) and
                self._looks_like_generic_call()):  # Additional check
                # Generic function call: func<T>() or obj.method<T>()
                generics = self.parse_generic_params()
                # Add generics to the expression (store as string annotation)
                if isinstance(expr, Identifier):
                    expr.name = f"{expr.name}<{generics}>"
                elif isinstance(expr, MemberAccess):
                    expr.member = f"{expr.member}<{generics}>"
                # Continue to parse the actual function call
                if self.match(TokenType.LPAREN):
                    expr = self.parse_function_call(expr)
            elif self.match(TokenType.LPAREN):
                expr = self.parse_function_call(expr)
            # Member access
            elif self.match(TokenType.DOT):
                self.advance()
                member = self.expect(TokenType.IDENTIFIER).value

                # Check for generic parameters after member (support both built-in and user-defined types)
                if (self.match(TokenType.LT) and
                    self.peek_token().type in (TokenType.TYPE_IDENTIFIER, TokenType.IDENTIFIER) and
                    self._looks_like_generic_call()):
                    generics = self.parse_generic_params()
                    member = f"{member}<{generics}>"

                # Check if this is a method call
                if self.match(TokenType.LPAREN):
                    expr = self.parse_method_call(expr, member)
                else:
                    expr = MemberAccess(object=expr, member=member)
            # Index access
            elif self.match(TokenType.LBRACKET):
                self.advance()
                index = self.parse_expression()
                self.expect(TokenType.RBRACKET)
                expr = IndexAccess(object=expr, index=index)
            else:
                break

        return expr

    def parse_generic_params(self) -> str:
        """Parse generic parameters in angle brackets: <T> or <K, V>."""
        self.expect(TokenType.LT)  # consume <

        params = []
        while not self.match(TokenType.GT):
            if self.match(TokenType.COMMA):
                self.advance()
                continue

            # Parse type parameter
            if self.match(TokenType.TYPE_IDENTIFIER, TokenType.IDENTIFIER):
                params.append(self.advance().value)
            else:
                raise self.error(f"Expected type parameter, got {self.current_token().type.name}")

        self.expect(TokenType.GT)  # consume >

        return ', '.join(params)

    def parse_function_call(self, func: Expression) -> FunctionCall:
        """Parse function call arguments."""
        self.expect(TokenType.LPAREN)

        args = []
        kwargs = {}

        while not self.match(TokenType.RPAREN):
            if self.match(TokenType.COMMA):
                self.advance()
                continue

            self.skip_newlines()
            if self.match(TokenType.RPAREN):
                break

            # Check for keyword argument (can be IDENTIFIER or TYPE_IDENTIFIER like 'color')
            if self.match(TokenType.IDENTIFIER, TokenType.TYPE_IDENTIFIER) and self.peek_token().type == TokenType.ASSIGN:
                key = self.advance().value
                self.advance()  # =
                value = self.parse_expression()
                kwargs[key] = value
            else:
                # Positional argument
                args.append(self.parse_expression())

        self.expect(TokenType.RPAREN)

        # Convert func expression to string if it's an identifier or member access
        if isinstance(func, Identifier):
            func_name = func.name
        elif isinstance(func, MemberAccess):
            func_name = func
        else:
            func_name = func

        return FunctionCall(func=func_name, args=args, kwargs=kwargs)

    def parse_method_call(self, obj: Expression, method: str) -> MethodCall:
        """Parse method call arguments."""
        self.expect(TokenType.LPAREN)

        args = []
        kwargs = {}

        while not self.match(TokenType.RPAREN):
            if self.match(TokenType.COMMA):
                self.advance()
                continue

            # Check for keyword argument (can be IDENTIFIER or TYPE_IDENTIFIER like 'color')
            if self.match(TokenType.IDENTIFIER, TokenType.TYPE_IDENTIFIER) and self.peek_token().type == TokenType.ASSIGN:
                key = self.advance().value
                self.advance()  # =
                value = self.parse_expression()
                kwargs[key] = value
            else:
                # Positional argument
                args.append(self.parse_expression())

        self.expect(TokenType.RPAREN)

        return MethodCall(object=obj, method=method, args=args, kwargs=kwargs)

    def parse_primary(self) -> Expression:
        """Parse primary expressions (literals, identifiers, parenthesized expressions)."""
        # Literals
        if self.match(TokenType.INT_LITERAL):
            token = self.advance()
            return Literal(value=token.value, literal_type='int')

        if self.match(TokenType.FLOAT_LITERAL):
            token = self.advance()
            return Literal(value=token.value, literal_type='float')

        if self.match(TokenType.STRING_LITERAL):
            token = self.advance()
            return Literal(value=token.value, literal_type='string', is_double_quoted=token.is_double_quoted)

        if self.match(TokenType.TRUE):
            self.advance()
            return Literal(value=True, literal_type='bool')

        if self.match(TokenType.FALSE):
            self.advance()
            return Literal(value=False, literal_type='bool')

        if self.match(TokenType.COLOR_LITERAL):
            token = self.advance()
            return Literal(value=token.value, literal_type='color', is_double_quoted=token.is_double_quoted)

        if self.match(TokenType.NA_LITERAL):
            self.advance()
            return NaLiteral()

        # Array literal
        if self.match(TokenType.LBRACKET):
            self.advance()
            elements = []

            while not self.match(TokenType.RBRACKET):
                if self.match(TokenType.COMMA):
                    self.advance()
                    continue

                elements.append(self.parse_expression())

            self.expect(TokenType.RBRACKET)
            return ArrayLiteral(elements=elements)

        # Parenthesized expression
        if self.match(TokenType.LPAREN):
            self.advance()
            expr = self.parse_expression()
            self.expect(TokenType.RPAREN)
            return expr

        # Identifier
        if self.match(TokenType.IDENTIFIER, TokenType.TYPE_IDENTIFIER):
            token = self.advance()
            return Identifier(name=token.value)

        # If expression
        if self.match(TokenType.IF):
            return self.parse_if_expression()

        # Switch expression
        if self.match(TokenType.SWITCH):
            return self.parse_switch_expression()

        # For/for-in as value expression: x = for i in arr ...
        if self.match(TokenType.FOR):
            return self.parse_for_loop()

        raise self.error(f"Unexpected token in expression: {self.current_token().type.name}")

    def parse_if_expression(self) -> IfExpression:
        """Parse if expression (returns a value)."""
        self.expect(TokenType.IF)
        condition = self.parse_expression()

        self.skip_newlines()
        self.expect(TokenType.INDENT)

        # Parse true branch (can be expression or statements)
        true_expr = self.parse_expression_or_block()

        self.skip_newlines()
        self.expect(TokenType.DEDENT)

        # Parse else/else-if branch
        false_expr = None
        if self.match(TokenType.ELSE):
            self.advance()
            self.skip_newlines()
            if self.match(TokenType.IF):
                # else if → recurse into another if-expression
                false_expr = self.parse_if_expression()
            else:
                self.expect(TokenType.INDENT)
                false_expr = self.parse_expression_or_block()
                self.skip_newlines()
                self.expect(TokenType.DEDENT)

        return IfExpression(condition=condition, true_expr=true_expr, false_expr=false_expr)

    def parse_switch_expression(self) -> SwitchExpression:
        """Parse switch expression (returns a value)."""
        self.expect(TokenType.SWITCH)

        # Optional expression to switch on
        expr = None
        if not self.match(TokenType.NEWLINE):
            expr = self.parse_expression()

        self.skip_newlines()
        self.expect(TokenType.INDENT)

        cases = []
        default = None

        while not self.match(TokenType.DEDENT):
            self.skip_newlines()
            if self.match(TokenType.DEDENT):
                break

            # Check for default case (starts with =>)
            if self.match(TokenType.ARROW):
                self.advance()  # consume =>
                self.skip_newlines()
                # Parse default value — may be a multi-statement block
                if self.match(TokenType.INDENT):
                    self.advance()  # consume INDENT
                    default = self.parse_expression_or_block()
                    self.skip_newlines()
                    self.expect(TokenType.DEDENT)
                else:
                    default = self.parse_expression()
                self.skip_newlines()
                continue

            # Parse case condition
            case_condition = self.parse_expression()

            self.expect(TokenType.ARROW)
            self.skip_newlines()

            # Parse case value — may be a multi-statement block
            if self.match(TokenType.INDENT):
                self.advance()  # consume INDENT
                case_value = self.parse_expression_or_block()
                self.skip_newlines()
                self.expect(TokenType.DEDENT)
            else:
                case_value = self.parse_expression()

            cases.append((case_condition, case_value))
            self.skip_newlines()

        self.expect(TokenType.DEDENT)

        return SwitchExpression(expr=expr, cases=cases, default=default)

    def parse_expression_or_block(self) -> Union[Expression, List[Statement]]:
        """Parse either a single expression or a block of statements.
        For multi-statement blocks, returns a list of statements where the
        last expression is the return value.
        """
        # Try to parse as expression first
        start_pos = self.pos
        try:
            expr = self.parse_expression()
            # If next is DEDENT, it's a single expression (end of block)
            if self.match(TokenType.DEDENT):
                return expr
            # If next is NEWLINE, check if more statements follow
            if self.match(TokenType.NEWLINE):
                self.skip_newlines()
                # If DEDENT follows the newlines, it's a single expression
                if self.match(TokenType.DEDENT):
                    return expr
                # Otherwise there are more statements — fall through to block parse
        except:
            pass

        # Otherwise, parse as block
        self.pos = start_pos
        return self.parse_block()


def parse(tokens: List[Token]) -> Script:
    """Convenience function to parse Pine Script tokens into AST."""
    parser = Parser(tokens)
    return parser.parse()
