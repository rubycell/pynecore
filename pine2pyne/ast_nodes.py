"""
AST (Abstract Syntax Tree) node definitions for Pine Script v6.

All node types used by the parser to represent Pine Script syntax.
"""
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union


# Base class for all AST nodes
@dataclass
class ASTNode:
    """Base class for all AST nodes."""
    line: int = 0
    column: int = 0


# ============================================================================
# Top-level nodes
# ============================================================================

@dataclass
class Script(ASTNode):
    """Root node representing the entire Pine Script."""
    version: Optional['VersionAnnotation'] = None
    script_decl: Optional[Union['IndicatorDecl', 'StrategyDecl', 'LibraryDecl']] = None
    imports: List['ImportDecl'] = field(default_factory=list)
    declarations: List[Union['VarDecl', 'VaripDecl', 'TypeDecl', 'EnumDecl', 'FuncDecl', 'InputDecl']] = field(default_factory=list)
    body: List['Statement'] = field(default_factory=list)


@dataclass
class VersionAnnotation(ASTNode):
    """//@version=6 annotation."""
    version: str = ""  # "6" or "5"


@dataclass
class IndicatorDecl(ASTNode):
    """indicator("Title", ...) declaration."""
    title: str = ""
    kwargs: dict = field(default_factory=dict)


@dataclass
class StrategyDecl(ASTNode):
    """strategy("Title", ...) declaration."""
    title: str = ""
    kwargs: dict = field(default_factory=dict)


@dataclass
class LibraryDecl(ASTNode):
    """library("Title", ...) declaration."""
    title: str = ""
    kwargs: dict = field(default_factory=dict)


@dataclass
class ImportDecl(ASTNode):
    """import user/library/version as alias"""
    user: str = ""
    library: str = ""
    version: Optional[str] = None
    alias: Optional[str] = None


# ============================================================================
# Declarations
# ============================================================================

@dataclass
class VarDecl(ASTNode):
    """var [type] name = value"""
    name: str = ""
    value: 'Expression' = None
    type_hint: Optional[str] = None
    is_var: bool = True  # True for 'var', False for 'varip'


@dataclass
class VaripDecl(ASTNode):
    """varip [type] name = value"""
    name: str = ""
    value: 'Expression' = None
    type_hint: Optional[str] = None


@dataclass
class TypeDecl(ASTNode):
    """type MyType ... (User-defined type)"""
    name: str = ""
    fields: List[tuple] = field(default_factory=list)  # [(field_name, type_hint, default_value), ...] or [(field_name, type_hint), ...]


@dataclass
class EnumDecl(ASTNode):
    """enum MyEnum ... (Enum definition)"""
    name: str = ""
    members: List = field(default_factory=list)  # List[str] or List[tuple[str, Expression]]


@dataclass
class FuncDecl(ASTNode):
    """Function declaration: name(params) => body"""
    name: str = ""
    params: List['Parameter'] = field(default_factory=list)
    body: Union['Expression', List['Statement']] = None
    return_type: Optional[str] = None
    is_method: bool = False
    is_export: bool = False
    indexed_params: set = field(default_factory=set)  # Params that use [n] history access


@dataclass
class Parameter(ASTNode):
    """Function parameter: [type] name [= default]"""
    name: str = ""
    type_hint: Optional[str] = None
    default: Optional['Expression'] = None


@dataclass
class InputDecl(ASTNode):
    """Input declaration: name = input.*(...)"""
    name: str = ""
    func: str = ""  # e.g., "input.int"
    args: List['Expression'] = field(default_factory=list)
    kwargs: dict = field(default_factory=dict)


# ============================================================================
# Expressions
# ============================================================================

@dataclass
class Expression(ASTNode):
    """Base class for all expressions."""
    pass


@dataclass
class BinaryOp(Expression):
    """Binary operation: left op right"""
    left: Expression = None
    op: str = ""
    right: Expression = None


@dataclass
class UnaryOp(Expression):
    """Unary operation: op operand"""
    op: str = ""
    operand: Expression = None


@dataclass
class TernaryOp(Expression):
    """Ternary operation: condition ? true_expr : false_expr"""
    condition: Expression = None
    true_expr: Expression = None
    false_expr: Expression = None


@dataclass
class FunctionCall(Expression):
    """Function call: func(args, kwargs)"""
    func: Union[str, 'MemberAccess'] = ""  # Can be "ta.sma" or MemberAccess node
    args: List[Expression] = field(default_factory=list)
    kwargs: dict = field(default_factory=dict)


@dataclass
class MethodCall(Expression):
    """Method call: object.method(args)"""
    object: Expression = None
    method: str = ""
    args: List[Expression] = field(default_factory=list)
    kwargs: dict = field(default_factory=dict)


@dataclass
class MemberAccess(Expression):
    """Member access: object.member"""
    object: Union[str, Expression] = ""  # Can be identifier string or expression
    member: str = ""


@dataclass
class IndexAccess(Expression):
    """Index access: object[index]"""
    object: Expression = None
    index: Expression = None


@dataclass
class ArrayLiteral(Expression):
    """Array literal: [elem1, elem2, ...]"""
    elements: List[Expression] = field(default_factory=list)


@dataclass
class TupleDestructure(Expression):
    """Tuple destructuring: [a, b, c] = expr"""
    names: List[str] = field(default_factory=list)
    value: Expression = None


@dataclass
class Identifier(Expression):
    """Identifier (variable name)"""
    name: str = ""


@dataclass
class Literal(Expression):
    """Literal value (int, float, string, bool, color)"""
    value: Any = None
    literal_type: str = ""  # 'int', 'float', 'string', 'bool', 'color'
    is_double_quoted: bool = False  # Track if original source used double quotes


@dataclass
class NaLiteral(Expression):
    """The 'na' literal value"""
    pass


@dataclass
class IfExpression(Expression):
    """If expression (returns a value): if cond\n    val1\nelse\n    val2"""
    condition: Expression = None
    true_expr: Union[Expression, List['Statement']] = None
    false_expr: Optional[Union[Expression, List['Statement']]] = None


@dataclass
class SwitchExpression(Expression):
    """Switch expression (returns a value)"""
    expr: Optional[Expression] = None
    cases: List[tuple[Expression, Union[Expression, List['Statement']]]] = field(default_factory=list)
    default: Optional[Union[Expression, List['Statement']]] = None


# ============================================================================
# Statements
# ============================================================================

@dataclass
class Statement(ASTNode):
    """Base class for all statements."""
    pass


@dataclass
class Assignment(Statement):
    """Assignment statement: target = value"""
    target: Union[str, 'TupleDestructure'] = ""
    value: Expression = None
    type_hint: Optional[str] = None


@dataclass
class Reassignment(Statement):
    """Reassignment statement: target := value"""
    target: str = ""
    value: Expression = None


@dataclass
class IfStatement(Statement):
    """If statement: if cond\n    body\nelse if ...\nelse ..."""
    condition: Expression = None
    body: List[Statement] = field(default_factory=list)
    elseifs: List[tuple[Expression, List[Statement]]] = field(default_factory=list)
    else_body: Optional[List[Statement]] = None


@dataclass
class ForLoop(Statement):
    """For loop: for var = from to to_val [by step]"""
    var: str = ""
    from_val: Expression = None
    to_val: Expression = None
    step: Optional[Expression] = None
    body: List[Statement] = field(default_factory=list)


@dataclass
class ForInLoop(Statement):
    """For-in loop: for [vars] in iterable"""
    vars: List[str] = field(default_factory=list)
    iterable: Expression = None
    body: List[Statement] = field(default_factory=list)


@dataclass
class WhileLoop(Statement):
    """While loop: while condition\n    body"""
    condition: Expression = None
    body: List[Statement] = field(default_factory=list)


@dataclass
class SwitchStatement(Statement):
    """Switch statement (imperative form)"""
    expr: Optional[Expression] = None
    cases: List[tuple[Expression, List[Statement]]] = field(default_factory=list)
    default: Optional[List[Statement]] = None


@dataclass
class BreakStatement(Statement):
    """Break statement"""
    pass


@dataclass
class ContinueStatement(Statement):
    """Continue statement"""
    pass


@dataclass
class ExpressionStatement(Statement):
    """Expression as statement (e.g., function call on its own line)"""
    expr: Expression = None


@dataclass
class ReturnStatement(Statement):
    """Explicit return statement (for codegen)"""
    expr: Optional[Expression] = None


@dataclass
class RawCode(Statement):
    """Raw Python code to be emitted as-is by codegen."""
    code: str = ""


# ============================================================================
# Type annotations
# ============================================================================

# Forward references for type checking
Statement = Union[
    Assignment, Reassignment, IfStatement, ForLoop, ForInLoop,
    WhileLoop, SwitchStatement, BreakStatement, ContinueStatement,
    ExpressionStatement, ReturnStatement, VarDecl, VaripDecl,
    FuncDecl, InputDecl
]
