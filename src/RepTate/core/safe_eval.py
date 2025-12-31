"""Safe mathematical expression evaluation.

This module provides secure evaluation of mathematical expressions,
replacing unsafe eval() usage to eliminate code injection vulnerabilities.

The implementation:
- Uses ast.parse() with mode='eval' for parsing
- Recursively validates AST nodes against a whitelist
- Compiles to a callable for repeated evaluation
- Does NOT use eval() or exec() internally

Public API:
    safe_eval(expr, variables) -> float
    safe_eval_array(expr, variables) -> ndarray
    SafeExpression.parse(expr) -> SafeExpression
    SafeExpression.evaluate(bindings) -> float
    SafeArrayExpression.parse(expr) -> SafeArrayExpression
    SafeArrayExpression.evaluate(bindings) -> ndarray

Example:
    >>> from RepTate.core.safe_eval import safe_eval, SafeExpression
    >>> result = safe_eval("A * exp(-t / tau)", {"A": 1000.0, "t": 0.5, "tau": 0.1})
    >>> expr = SafeExpression.parse("sin(omega * t)")
    >>> result = expr.evaluate({"omega": 6.28, "t": 0.25})

Security:
    Rejects: import, exec, eval, compile, attribute access, subscript, dunder names
    Allows: +, -, *, /, **, unary -, sin, cos, tan, exp, log, log10, sqrt, abs

Contract: contracts/safe_eval.md
"""

from __future__ import annotations

import ast
import math
import operator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Union

import numpy as np
from numpy import ndarray

if TYPE_CHECKING:
    from collections.abc import Callable

# Type alias for array-compatible values
ArrayLike = Union[float, ndarray]

# =============================================================================
# Constants
# =============================================================================

# Allowed binary operators
_BINARY_OPS: Final[dict[type[ast.operator], Callable[[float, float], float]]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}

# Allowed unary operators
_UNARY_OPS: Final[dict[type[ast.unaryop], Callable[[float], float]]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Allowed mathematical functions (scalar)
_SAFE_FUNCTIONS: Final[dict[str, Callable[[float], float]]] = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "sqrt": math.sqrt,
    "abs": abs,
}

# Allowed mathematical functions (array-compatible)
# These numpy functions work with both scalars and arrays
_SAFE_ARRAY_FUNCTIONS: Final[dict[str, Callable[[ArrayLike], ArrayLike]]] = {
    # Trigonometric
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "arccos": np.arccos,
    "arcsin": np.arcsin,
    "arctan": np.arctan,
    # Hyperbolic
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "arcsinh": np.arcsinh,
    "arccosh": np.arccosh,
    "arctanh": np.arctanh,
    # Exponential and logarithmic
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    # Power and roots
    "sqrt": np.sqrt,
    "power": np.power,
    # Rounding
    "around": np.around,
    "round": np.round,
    "rint": np.rint,
    "floor": np.floor,
    "ceil": np.ceil,
    "trunc": np.trunc,
    # Absolute
    "fabs": np.fabs,
    "abs": np.abs,
    # Angle conversion
    "deg2rad": np.deg2rad,
    "rad2deg": np.rad2deg,
    # Modulo (two-argument but handled specially)
    "mod": np.mod,
    # Two-argument arctan
    "arctan2": np.arctan2,
}

# Allowed constants for array expressions
_SAFE_CONSTANTS: Final[dict[str, float]] = {
    "e": np.e,
    "pi": np.pi,
}

# Names that are explicitly disallowed
_DISALLOWED_NAMES: Final[frozenset[str]] = frozenset({
    "__import__",
    "__builtins__",
    "__name__",
    "__doc__",
    "__package__",
    "__loader__",
    "__spec__",
    "__file__",
    "__cached__",
    "eval",
    "exec",
    "compile",
    "open",
    "input",
    "print",
    "globals",
    "locals",
    "getattr",
    "setattr",
    "delattr",
    "type",
    "vars",
    "dir",
    "hasattr",
    "isinstance",
    "issubclass",
    "callable",
    "object",
    "classmethod",
    "staticmethod",
    "property",
    "super",
    "breakpoint",
    "memoryview",
    "bytearray",
    "bytes",
    "format",
    "help",
    "id",
    "hash",
    "len",
    "repr",
    "str",
    "chr",
    "ord",
    "range",
    "enumerate",
    "zip",
    "map",
    "filter",
    "sorted",
    "reversed",
    "iter",
    "next",
    "slice",
    "list",
    "dict",
    "set",
    "tuple",
    "frozenset",
})


# =============================================================================
# AST Validation (Scalar)
# =============================================================================

class _ASTValidator(ast.NodeVisitor):
    """Validates AST nodes against the whitelist.

    Collects variable names and raises ValueError for unsafe operations.
    """

    def __init__(self) -> None:
        self.variables: set[str] = set()

    def visit_Expression(self, node: ast.Expression) -> None:
        """Visit the root Expression node.

        Args:
            node (ast.Expression): The root expression AST node to validate.
        """
        self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        """Visit binary operation (e.g., a + b).

        Validates that the binary operator is in the allowed whitelist
        and recursively validates left and right operands.

        Args:
            node (ast.BinOp): The binary operation AST node to validate.

        Raises:
            ValueError: If the operator is not in the allowed set.
        """
        if type(node.op) not in _BINARY_OPS:
            raise ValueError(f"Disallowed operator: {type(node.op).__name__}")
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        """Visit unary operation (e.g., -a).

        Validates that the unary operator is in the allowed whitelist
        and recursively validates the operand.

        Args:
            node (ast.UnaryOp): The unary operation AST node to validate.

        Raises:
            ValueError: If the operator is not in the allowed set.
        """
        if type(node.op) not in _UNARY_OPS:
            raise ValueError(f"Disallowed operator: {type(node.op).__name__}")
        self.visit(node.operand)

    def visit_Call(self, node: ast.Call) -> None:
        """Visit function call (e.g., sin(x)).

        Validates that the function is a named function in the safe functions
        whitelist, validates all arguments, and rejects keyword arguments.

        Args:
            node (ast.Call): The function call AST node to validate.

        Raises:
            ValueError: If the function is not a simple named call, not in the
                safe functions whitelist, or uses keyword arguments.
        """
        if not isinstance(node.func, ast.Name):
            raise ValueError("Disallowed operation: only named function calls allowed")

        func_name = node.func.id
        if func_name not in _SAFE_FUNCTIONS:
            raise ValueError(f"Disallowed function: {func_name}")

        # Validate arguments
        for arg in node.args:
            self.visit(arg)

        # Reject keyword arguments
        if node.keywords:
            raise ValueError("Disallowed operation: keyword arguments not allowed")

    def visit_Name(self, node: ast.Name) -> None:
        """Visit variable name.

        Validates that the name is not a dunder name, not in the disallowed
        names list, and collects it as a variable if not a safe function.

        Args:
            node (ast.Name): The name AST node to validate.

        Raises:
            ValueError: If the name is a dunder name or in the disallowed list.
        """
        name = node.id

        # Check for dunder names
        if name.startswith("__") and name.endswith("__"):
            raise ValueError(f"Disallowed name: {name}")

        # Check for explicitly disallowed names
        if name in _DISALLOWED_NAMES:
            raise ValueError(f"Disallowed name: {name}")

        # Check for names starting with double underscore
        if name.startswith("__"):
            raise ValueError(f"Disallowed name: {name}")

        # If it's not a safe function, it's a variable
        if name not in _SAFE_FUNCTIONS:
            self.variables.add(name)

    def visit_Constant(self, node: ast.Constant) -> None:
        """Visit constant (number or string literal).

        Validates that the constant is a numeric type (int or float).

        Args:
            node (ast.Constant): The constant AST node to validate.

        Raises:
            ValueError: If the constant is not an int or float.
        """
        if not isinstance(node.value, (int, float)):
            raise ValueError(f"Disallowed constant type: {type(node.value).__name__}")

    def visit_Num(self, node: ast.Num) -> None:
        """Visit number (legacy, for compatibility).

        Handles ast.Num nodes for Python 3.7 compatibility.
        ast.Num is deprecated in Python 3.8+ but still exists.

        Args:
            node (ast.Num): The number AST node to validate.
        """
        # ast.Num is deprecated in Python 3.8+ but still exists
        pass

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Visit attribute access (e.g., x.attr).

        Rejects all attribute access as a security measure.

        Args:
            node (ast.Attribute): The attribute access AST node.

        Raises:
            ValueError: Always raises to disallow attribute access.
        """
        raise ValueError(f"Disallowed operation: attribute access ({node.attr})")

    def visit_Subscript(self, node: ast.Subscript) -> None:
        """Visit subscript access (e.g., x[0]).

        Rejects all subscript access as a security measure.

        Args:
            node (ast.Subscript): The subscript access AST node.

        Raises:
            ValueError: Always raises to disallow subscript access.
        """
        raise ValueError("Disallowed operation: subscript access")

    def visit_Lambda(self, node: ast.Lambda) -> None:
        """Visit lambda expression.

        Rejects lambda expressions as a security measure.

        Args:
            node (ast.Lambda): The lambda expression AST node.

        Raises:
            ValueError: Always raises to disallow lambda expressions.
        """
        raise ValueError("Disallowed operation: lambda expression")

    def visit_ListComp(self, node: ast.ListComp) -> None:
        """Visit list comprehension.

        Rejects list comprehensions as a security measure.

        Args:
            node (ast.ListComp): The list comprehension AST node.

        Raises:
            ValueError: Always raises to disallow list comprehensions.
        """
        raise ValueError("Disallowed operation: list comprehension")

    def visit_SetComp(self, node: ast.SetComp) -> None:
        """Visit set comprehension.

        Rejects set comprehensions as a security measure.

        Args:
            node (ast.SetComp): The set comprehension AST node.

        Raises:
            ValueError: Always raises to disallow set comprehensions.
        """
        raise ValueError("Disallowed operation: set comprehension")

    def visit_DictComp(self, node: ast.DictComp) -> None:
        """Visit dict comprehension.

        Rejects dict comprehensions as a security measure.

        Args:
            node (ast.DictComp): The dict comprehension AST node.

        Raises:
            ValueError: Always raises to disallow dict comprehensions.
        """
        raise ValueError("Disallowed operation: dict comprehension")

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        """Visit generator expression.

        Rejects generator expressions as a security measure.

        Args:
            node (ast.GeneratorExp): The generator expression AST node.

        Raises:
            ValueError: Always raises to disallow generator expressions.
        """
        raise ValueError("Disallowed operation: generator expression")

    def visit_IfExp(self, node: ast.IfExp) -> None:
        """Visit conditional expression (ternary).

        Rejects conditional expressions as a security measure.

        Args:
            node (ast.IfExp): The conditional expression AST node.

        Raises:
            ValueError: Always raises to disallow conditional expressions.
        """
        raise ValueError("Disallowed operation: conditional expression")

    def visit_List(self, node: ast.List) -> None:
        """Visit list literal.

        Rejects list literals as a security measure.

        Args:
            node (ast.List): The list literal AST node.

        Raises:
            ValueError: Always raises to disallow list literals.
        """
        raise ValueError("Disallowed operation: list literal")

    def visit_Dict(self, node: ast.Dict) -> None:
        """Visit dict literal.

        Rejects dict literals as a security measure.

        Args:
            node (ast.Dict): The dict literal AST node.

        Raises:
            ValueError: Always raises to disallow dict literals.
        """
        raise ValueError("Disallowed operation: dict literal")

    def visit_Set(self, node: ast.Set) -> None:
        """Visit set literal.

        Rejects set literals as a security measure.

        Args:
            node (ast.Set): The set literal AST node.

        Raises:
            ValueError: Always raises to disallow set literals.
        """
        raise ValueError("Disallowed operation: set literal")

    def visit_Tuple(self, node: ast.Tuple) -> None:
        """Visit tuple literal.

        Rejects tuple literals as a security measure.

        Args:
            node (ast.Tuple): The tuple literal AST node.

        Raises:
            ValueError: Always raises to disallow tuple literals.
        """
        raise ValueError("Disallowed operation: tuple literal")

    def visit_Compare(self, node: ast.Compare) -> None:
        """Visit comparison (e.g., a < b).

        Rejects comparison operations as a security measure.

        Args:
            node (ast.Compare): The comparison AST node.

        Raises:
            ValueError: Always raises to disallow comparisons.
        """
        raise ValueError("Disallowed operation: comparison")

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        """Visit boolean operation (e.g., and, or).

        Rejects boolean operations as a security measure.

        Args:
            node (ast.BoolOp): The boolean operation AST node.

        Raises:
            ValueError: Always raises to disallow boolean operations.
        """
        raise ValueError("Disallowed operation: boolean operation")

    def generic_visit(self, node: ast.AST) -> None:
        """Default visitor for unhandled node types.

        Rejects all node types except Expression and Load for safety.

        Args:
            node (ast.AST): The AST node being visited.

        Raises:
            ValueError: If the node type is not explicitly allowed.
        """
        # For safety, reject any node type not explicitly handled
        node_type = type(node).__name__
        if node_type not in {"Expression", "Load"}:
            raise ValueError(f"Disallowed operation: {node_type}")


# =============================================================================
# AST Validation (Array)
# =============================================================================

class _ArrayASTValidator(ast.NodeVisitor):
    """Validates AST nodes against the array-compatible whitelist.

    Collects variable names and raises ValueError for unsafe operations.
    Allows a broader set of numpy functions compared to the scalar validator.
    """

    def __init__(self) -> None:
        self.variables: set[str] = set()

    def visit_Expression(self, node: ast.Expression) -> None:
        """Visit the root Expression node.

        Args:
            node (ast.Expression): The root expression AST node to validate.
        """
        self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        """Visit binary operation (e.g., a + b).

        Validates that the binary operator is in the allowed whitelist
        and recursively validates left and right operands.

        Args:
            node (ast.BinOp): The binary operation AST node to validate.

        Raises:
            ValueError: If the operator is not in the allowed set.
        """
        if type(node.op) not in _BINARY_OPS:
            raise ValueError(f"Disallowed operator: {type(node.op).__name__}")
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        """Visit unary operation (e.g., -a).

        Validates that the unary operator is in the allowed whitelist
        and recursively validates the operand.

        Args:
            node (ast.UnaryOp): The unary operation AST node to validate.

        Raises:
            ValueError: If the operator is not in the allowed set.
        """
        if type(node.op) not in _UNARY_OPS:
            raise ValueError(f"Disallowed operator: {type(node.op).__name__}")
        self.visit(node.operand)

    def visit_Call(self, node: ast.Call) -> None:
        """Visit function call (e.g., sin(x)).

        Validates that the function is a named function in the safe array
        functions whitelist, validates all arguments, and rejects keyword
        arguments.

        Args:
            node (ast.Call): The function call AST node to validate.

        Raises:
            ValueError: If the function is not a simple named call, not in the
                safe array functions whitelist, or uses keyword arguments.
        """
        if not isinstance(node.func, ast.Name):
            raise ValueError("Disallowed operation: only named function calls allowed")

        func_name = node.func.id
        if func_name not in _SAFE_ARRAY_FUNCTIONS:
            raise ValueError(f"Disallowed function: {func_name}")

        # Validate arguments
        for arg in node.args:
            self.visit(arg)

        # Reject keyword arguments
        if node.keywords:
            raise ValueError("Disallowed operation: keyword arguments not allowed")

    def visit_Name(self, node: ast.Name) -> None:
        """Visit variable name.

        Validates that the name is not a dunder name, not in the disallowed
        names list, and collects it as a variable if not a safe function or
        constant.

        Args:
            node (ast.Name): The name AST node to validate.

        Raises:
            ValueError: If the name is a dunder name or in the disallowed list.
        """
        name = node.id

        # Check for dunder names
        if name.startswith("__") and name.endswith("__"):
            raise ValueError(f"Disallowed name: {name}")

        # Check for explicitly disallowed names
        if name in _DISALLOWED_NAMES:
            raise ValueError(f"Disallowed name: {name}")

        # Check for names starting with double underscore
        if name.startswith("__"):
            raise ValueError(f"Disallowed name: {name}")

        # If it's not a safe function or constant, it's a variable
        if name not in _SAFE_ARRAY_FUNCTIONS and name not in _SAFE_CONSTANTS:
            self.variables.add(name)

    def visit_Constant(self, node: ast.Constant) -> None:
        """Visit constant (number or string literal).

        Validates that the constant is a numeric type (int or float).

        Args:
            node (ast.Constant): The constant AST node to validate.

        Raises:
            ValueError: If the constant is not an int or float.
        """
        if not isinstance(node.value, (int, float)):
            raise ValueError(f"Disallowed constant type: {type(node.value).__name__}")

    def visit_Num(self, node: ast.Num) -> None:
        """Visit number (legacy, for compatibility).

        Handles ast.Num nodes for Python 3.7 compatibility.
        ast.Num is deprecated in Python 3.8+ but still exists.

        Args:
            node (ast.Num): The number AST node to validate.
        """
        pass

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Visit attribute access (e.g., x.attr).

        Rejects all attribute access as a security measure.

        Args:
            node (ast.Attribute): The attribute access AST node.

        Raises:
            ValueError: Always raises to disallow attribute access.
        """
        raise ValueError(f"Disallowed operation: attribute access ({node.attr})")

    def visit_Subscript(self, node: ast.Subscript) -> None:
        """Visit subscript access (e.g., x[0]).

        Rejects all subscript access as a security measure.

        Args:
            node (ast.Subscript): The subscript access AST node.

        Raises:
            ValueError: Always raises to disallow subscript access.
        """
        raise ValueError("Disallowed operation: subscript access")

    def visit_Lambda(self, node: ast.Lambda) -> None:
        """Visit lambda expression.

        Rejects lambda expressions as a security measure.

        Args:
            node (ast.Lambda): The lambda expression AST node.

        Raises:
            ValueError: Always raises to disallow lambda expressions.
        """
        raise ValueError("Disallowed operation: lambda expression")

    def visit_ListComp(self, node: ast.ListComp) -> None:
        """Visit list comprehension.

        Rejects list comprehensions as a security measure.

        Args:
            node (ast.ListComp): The list comprehension AST node.

        Raises:
            ValueError: Always raises to disallow list comprehensions.
        """
        raise ValueError("Disallowed operation: list comprehension")

    def visit_SetComp(self, node: ast.SetComp) -> None:
        """Visit set comprehension.

        Rejects set comprehensions as a security measure.

        Args:
            node (ast.SetComp): The set comprehension AST node.

        Raises:
            ValueError: Always raises to disallow set comprehensions.
        """
        raise ValueError("Disallowed operation: set comprehension")

    def visit_DictComp(self, node: ast.DictComp) -> None:
        """Visit dict comprehension.

        Rejects dict comprehensions as a security measure.

        Args:
            node (ast.DictComp): The dict comprehension AST node.

        Raises:
            ValueError: Always raises to disallow dict comprehensions.
        """
        raise ValueError("Disallowed operation: dict comprehension")

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        """Visit generator expression.

        Rejects generator expressions as a security measure.

        Args:
            node (ast.GeneratorExp): The generator expression AST node.

        Raises:
            ValueError: Always raises to disallow generator expressions.
        """
        raise ValueError("Disallowed operation: generator expression")

    def visit_IfExp(self, node: ast.IfExp) -> None:
        """Visit conditional expression (ternary).

        Rejects conditional expressions as a security measure.

        Args:
            node (ast.IfExp): The conditional expression AST node.

        Raises:
            ValueError: Always raises to disallow conditional expressions.
        """
        raise ValueError("Disallowed operation: conditional expression")

    def visit_List(self, node: ast.List) -> None:
        """Visit list literal.

        Rejects list literals as a security measure.

        Args:
            node (ast.List): The list literal AST node.

        Raises:
            ValueError: Always raises to disallow list literals.
        """
        raise ValueError("Disallowed operation: list literal")

    def visit_Dict(self, node: ast.Dict) -> None:
        """Visit dict literal.

        Rejects dict literals as a security measure.

        Args:
            node (ast.Dict): The dict literal AST node.

        Raises:
            ValueError: Always raises to disallow dict literals.
        """
        raise ValueError("Disallowed operation: dict literal")

    def visit_Set(self, node: ast.Set) -> None:
        """Visit set literal.

        Rejects set literals as a security measure.

        Args:
            node (ast.Set): The set literal AST node.

        Raises:
            ValueError: Always raises to disallow set literals.
        """
        raise ValueError("Disallowed operation: set literal")

    def visit_Tuple(self, node: ast.Tuple) -> None:
        """Visit tuple literal.

        Rejects tuple literals as a security measure.

        Args:
            node (ast.Tuple): The tuple literal AST node.

        Raises:
            ValueError: Always raises to disallow tuple literals.
        """
        raise ValueError("Disallowed operation: tuple literal")

    def visit_Compare(self, node: ast.Compare) -> None:
        """Visit comparison (e.g., a < b).

        Rejects comparison operations as a security measure.

        Args:
            node (ast.Compare): The comparison AST node.

        Raises:
            ValueError: Always raises to disallow comparisons.
        """
        raise ValueError("Disallowed operation: comparison")

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        """Visit boolean operation (e.g., and, or).

        Rejects boolean operations as a security measure.

        Args:
            node (ast.BoolOp): The boolean operation AST node.

        Raises:
            ValueError: Always raises to disallow boolean operations.
        """
        raise ValueError("Disallowed operation: boolean operation")

    def generic_visit(self, node: ast.AST) -> None:
        """Default visitor for unhandled node types.

        Rejects all node types except Expression and Load for safety.

        Args:
            node (ast.AST): The AST node being visited.

        Raises:
            ValueError: If the node type is not explicitly allowed.
        """
        node_type = type(node).__name__
        if node_type not in {"Expression", "Load"}:
            raise ValueError(f"Disallowed operation: {node_type}")


# =============================================================================
# AST Evaluator (Scalar)
# =============================================================================

class _ASTEvaluator(ast.NodeVisitor):
    """Evaluates a validated AST with provided variable bindings.

    This is a safe evaluator that does NOT use eval() or exec().
    """

    def __init__(self, bindings: dict[str, float]) -> None:
        self.bindings = bindings

    def visit_Expression(self, node: ast.Expression) -> float:
        """Visit the root Expression node.

        Args:
            node (ast.Expression): The root expression AST node to evaluate.

        Returns:
            float: The evaluated result of the expression.
        """
        return self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp) -> float:
        """Evaluate binary operation.

        Evaluates the left and right operands and applies the binary operator.

        Args:
            node (ast.BinOp): The binary operation AST node to evaluate.

        Returns:
            float: The result of applying the operator to the operands.
        """
        left = self.visit(node.left)
        right = self.visit(node.right)
        op_func = _BINARY_OPS[type(node.op)]
        return op_func(left, right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> float:
        """Evaluate unary operation.

        Evaluates the operand and applies the unary operator.

        Args:
            node (ast.UnaryOp): The unary operation AST node to evaluate.

        Returns:
            float: The result of applying the operator to the operand.
        """
        operand = self.visit(node.operand)
        op_func = _UNARY_OPS[type(node.op)]
        return op_func(operand)

    def visit_Call(self, node: ast.Call) -> float:
        """Evaluate function call.

        Evaluates all arguments and calls the safe function.

        Args:
            node (ast.Call): The function call AST node to evaluate.

        Returns:
            float: The result of the function call.

        Raises:
            ValueError: If the function does not take exactly 1 argument.
        """
        # We know from validation that node.func is ast.Name
        # and the function is in _SAFE_FUNCTIONS
        func_name = node.func.id  # type: ignore[union-attr]
        func = _SAFE_FUNCTIONS[func_name]
        args = [self.visit(arg) for arg in node.args]

        # All safe functions take exactly one argument
        if len(args) != 1:
            raise ValueError(f"Function {func_name} takes exactly 1 argument")

        return func(args[0])

    def visit_Name(self, node: ast.Name) -> float:
        """Evaluate variable reference.

        Looks up the variable name in the bindings dictionary.

        Args:
            node (ast.Name): The name AST node to evaluate.

        Returns:
            float: The value of the variable from bindings.

        Raises:
            ValueError: If the variable is missing from bindings or is a function
                name used as a variable.
        """
        name = node.id

        # Check if it's a safe function (should not happen in evaluated context)
        if name in _SAFE_FUNCTIONS:
            raise ValueError(f"Function {name} used as variable")

        # Look up in bindings
        if name not in self.bindings:
            raise ValueError(f"Missing variable: {name}")

        return self.bindings[name]

    def visit_Constant(self, node: ast.Constant) -> float:
        """Evaluate constant.

        Converts the constant value to a float.

        Args:
            node (ast.Constant): The constant AST node to evaluate.

        Returns:
            float: The constant value as a float.
        """
        return float(node.value)

    def visit_Num(self, node: ast.Num) -> float:
        """Evaluate number (legacy).

        Handles ast.Num nodes for Python 3.7 compatibility.
        ast.Num is deprecated in Python 3.8+ but still exists.

        Args:
            node (ast.Num): The number AST node to evaluate.

        Returns:
            float: The numeric value as a float.
        """
        return float(node.n)  # type: ignore[attr-defined]

    def generic_visit(self, node: ast.AST) -> float:
        """Default visitor - should never be reached if validation passed.

        Args:
            node (ast.AST): The AST node being visited.

        Returns:
            float: Never returns, always raises.

        Raises:
            ValueError: Always raises for unexpected node types.
        """
        raise ValueError(f"Unexpected node type: {type(node).__name__}")


# =============================================================================
# AST Evaluator (Array)
# =============================================================================

class _ArrayASTEvaluator(ast.NodeVisitor):
    """Evaluates a validated AST with provided variable bindings (array-compatible).

    This is a safe evaluator that does NOT use eval() or exec().
    Supports numpy arrays and broadcasting.
    """

    def __init__(self, bindings: dict[str, Any]) -> None:
        self.bindings = bindings

    def visit_Expression(self, node: ast.Expression) -> ArrayLike:
        """Visit the root Expression node.

        Args:
            node (ast.Expression): The root expression AST node to evaluate.

        Returns:
            ArrayLike: The evaluated result (scalar or array).
        """
        return self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp) -> ArrayLike:
        """Evaluate binary operation.

        Evaluates the left and right operands and applies the binary operator
        using numpy's array broadcasting rules.

        Args:
            node (ast.BinOp): The binary operation AST node to evaluate.

        Returns:
            ArrayLike: The result of applying the operator (scalar or array).

        Raises:
            ValueError: If an unexpected operator type is encountered.
        """
        left = self.visit(node.left)
        right = self.visit(node.right)
        op_type = type(node.op)
        if op_type == ast.Add:
            return left + right
        elif op_type == ast.Sub:
            return left - right
        elif op_type == ast.Mult:
            return left * right
        elif op_type == ast.Div:
            return left / right
        elif op_type == ast.Pow:
            return left ** right
        else:
            raise ValueError(f"Unexpected operator: {op_type.__name__}")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ArrayLike:
        """Evaluate unary operation.

        Evaluates the operand and applies the unary operator.

        Args:
            node (ast.UnaryOp): The unary operation AST node to evaluate.

        Returns:
            ArrayLike: The result of applying the operator (scalar or array).

        Raises:
            ValueError: If an unexpected operator type is encountered.
        """
        operand = self.visit(node.operand)
        op_type = type(node.op)
        if op_type == ast.UAdd:
            return +operand
        elif op_type == ast.USub:
            return -operand
        else:
            raise ValueError(f"Unexpected unary operator: {op_type.__name__}")

    def visit_Call(self, node: ast.Call) -> ArrayLike:
        """Evaluate function call.

        Evaluates all arguments and calls the safe array function.
        Handles both single-argument and two-argument numpy functions.

        Args:
            node (ast.Call): The function call AST node to evaluate.

        Returns:
            ArrayLike: The result of the function call (scalar or array).

        Raises:
            ValueError: If the function receives the wrong number of arguments.
        """
        func_name = node.func.id  # type: ignore[union-attr]
        func = _SAFE_ARRAY_FUNCTIONS[func_name]
        args = [self.visit(arg) for arg in node.args]

        # Handle functions with 1 or 2 arguments
        if func_name in {"arctan2", "mod", "power"}:
            if len(args) != 2:
                raise ValueError(f"Function {func_name} requires exactly 2 arguments")
            return func(args[0], args[1])
        else:
            if len(args) != 1:
                raise ValueError(f"Function {func_name} requires exactly 1 argument")
            return func(args[0])

    def visit_Name(self, node: ast.Name) -> ArrayLike:
        """Evaluate variable reference.

        Looks up the variable name in constants or bindings dictionary.

        Args:
            node (ast.Name): The name AST node to evaluate.

        Returns:
            ArrayLike: The value of the constant or variable (scalar or array).

        Raises:
            ValueError: If the variable is missing from bindings or is a function
                name used as a variable.
        """
        name = node.id

        # Check if it's a constant
        if name in _SAFE_CONSTANTS:
            return _SAFE_CONSTANTS[name]

        # Check if it's a safe function (should not happen in evaluated context)
        if name in _SAFE_ARRAY_FUNCTIONS:
            raise ValueError(f"Function {name} used as variable")

        # Look up in bindings
        if name not in self.bindings:
            raise ValueError(f"Missing variable: {name}")

        return self.bindings[name]

    def visit_Constant(self, node: ast.Constant) -> float:
        """Evaluate constant.

        Converts the constant value to a float.

        Args:
            node (ast.Constant): The constant AST node to evaluate.

        Returns:
            float: The constant value as a float.
        """
        return float(node.value)

    def visit_Num(self, node: ast.Num) -> float:
        """Evaluate number (legacy).

        Handles ast.Num nodes for Python 3.7 compatibility.
        ast.Num is deprecated in Python 3.8+ but still exists.

        Args:
            node (ast.Num): The number AST node to evaluate.

        Returns:
            float: The numeric value as a float.
        """
        return float(node.n)  # type: ignore[attr-defined]

    def generic_visit(self, node: ast.AST) -> ArrayLike:
        """Default visitor - should never be reached if validation passed.

        Args:
            node (ast.AST): The AST node being visited.

        Returns:
            ArrayLike: Never returns, always raises.

        Raises:
            ValueError: Always raises for unexpected node types.
        """
        raise ValueError(f"Unexpected node type: {type(node).__name__}")


# =============================================================================
# Public API (Scalar)
# =============================================================================

@dataclass(frozen=True)
class SafeExpression:
    """A validated mathematical expression safe for evaluation.

    This class represents a parsed and validated expression that can be
    evaluated multiple times with different variable bindings.

    Attributes:
        raw: Original expression string provided by user
        ast: Parsed AST tree (validated at construction)
        variables: Set of variable names referenced in expression
        evaluator: Internal evaluator reference (not public)

    Example:
        >>> expr = SafeExpression.parse("sin(omega * t)")
        >>> print(expr.variables)  # frozenset({'omega', 't'})
        >>> result = expr.evaluate({"omega": 6.28, "t": 0.25})
    """

    raw: str
    ast: ast.Expression
    variables: frozenset[str]
    _evaluator: object = None  # Reserved for future compiled evaluator

    @classmethod
    def parse(cls, expr: str) -> SafeExpression:
        """Parse and validate expression, raising ValueError if unsafe.

        Args:
            expr: Mathematical expression string

        Returns:
            SafeExpression: Validated expression object

        Raises:
            ValueError: If expression is unsafe or invalid

        Example:
            >>> expr = SafeExpression.parse("A * exp(-t / tau)")
            >>> print(expr.variables)
            frozenset({'A', 't', 'tau'})
        """
        # Strip whitespace
        expr = expr.strip()

        # Check for empty expression
        if not expr:
            raise ValueError("Invalid expression: empty expression")

        # Parse the expression
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as e:
            raise ValueError(f"Invalid expression syntax: {e}") from e

        # Validate the AST
        validator = _ASTValidator()
        try:
            validator.visit(tree)
        except ValueError:
            raise

        return cls(
            raw=expr,
            ast=tree,
            variables=frozenset(validator.variables),
        )

    def evaluate(self, bindings: dict[str, float]) -> float:
        """Evaluate with provided variable bindings.

        Args:
            bindings: Dict mapping all referenced variables to values

        Returns:
            float: Evaluated result

        Raises:
            ValueError: If any referenced variable is missing from bindings

        Example:
            >>> expr = SafeExpression.parse("x ** 2")
            >>> expr.evaluate({"x": 3.0})
            9.0
        """
        # Check for missing variables
        missing = self.variables - set(bindings.keys())
        if missing:
            raise ValueError(f"Missing variable(s): {', '.join(sorted(missing))}")

        # Evaluate using the AST evaluator
        evaluator = _ASTEvaluator(bindings)
        return evaluator.visit(self.ast)


# =============================================================================
# Public API (Array)
# =============================================================================

@dataclass(frozen=True)
class SafeArrayExpression:
    """A validated mathematical expression safe for array evaluation.

    This class represents a parsed and validated expression that can be
    evaluated with numpy arrays for vectorized operations.

    Supports a broader set of numpy functions including:
    - Trigonometric: sin, cos, tan, arccos, arcsin, arctan, arctan2
    - Hyperbolic: sinh, cosh, tanh, arcsinh, arccosh, arctanh
    - Exponential: exp, log, log10
    - Power: sqrt, power
    - Rounding: around, round_, rint, floor, ceil, trunc
    - Absolute: fabs, abs
    - Angle: deg2rad, rad2deg
    - Other: mod

    Constants: e, pi

    Attributes:
        raw: Original expression string provided by user
        ast: Parsed AST tree (validated at construction)
        variables: Set of variable names referenced in expression

    Example:
        >>> expr = SafeArrayExpression.parse("A0 + A1 * x")
        >>> x = np.linspace(0, 10, 100)
        >>> result = expr.evaluate({"A0": 1.0, "A1": 2.0, "x": x})
    """

    raw: str
    ast: ast.Expression
    variables: frozenset[str]

    @classmethod
    def parse(cls, expr: str) -> SafeArrayExpression:
        """Parse and validate expression, raising ValueError if unsafe.

        Args:
            expr: Mathematical expression string

        Returns:
            SafeArrayExpression: Validated expression object

        Raises:
            ValueError: If expression is unsafe or invalid
        """
        # Strip whitespace
        expr = expr.strip()

        # Check for empty expression
        if not expr:
            raise ValueError("Invalid expression: empty expression")

        # Parse the expression
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as e:
            raise ValueError(f"Invalid expression syntax: {e}") from e

        # Validate the AST
        validator = _ArrayASTValidator()
        try:
            validator.visit(tree)
        except ValueError:
            raise

        return cls(
            raw=expr,
            ast=tree,
            variables=frozenset(validator.variables),
        )

    def evaluate(self, bindings: dict[str, Any]) -> ArrayLike:
        """Evaluate with provided variable bindings.

        Args:
            bindings: Dict mapping variables to values (float or ndarray)

        Returns:
            ArrayLike: Evaluated result (scalar or array depending on inputs)

        Raises:
            ValueError: If any referenced variable is missing from bindings
        """
        # Check for missing variables
        missing = self.variables - set(bindings.keys())
        if missing:
            raise ValueError(f"Missing variable(s): {', '.join(sorted(missing))}")

        # Evaluate using the array AST evaluator
        evaluator = _ArrayASTEvaluator(bindings)
        return evaluator.visit(self.ast)


def safe_eval(expr: str, variables: dict[str, float]) -> float:
    """Safely evaluate a mathematical expression with provided variable bindings.

    This function parses, validates, and evaluates a mathematical expression
    in a single call. For repeated evaluation of the same expression with
    different variables, use SafeExpression.parse() instead.

    Args:
        expr: Mathematical expression string (e.g., "sin(omega*t) + A*exp(-t/tau)")
        variables: Dict mapping variable names to float values

    Returns:
        float: Evaluated result

    Raises:
        ValueError: If expression contains disallowed operations
        ValueError: If referenced variable is not in bindings
        ValueError: If expression is syntactically invalid

    Example:
        >>> result = safe_eval(
        ...     "A * exp(-t / tau)",
        ...     {"A": 1000.0, "t": 0.5, "tau": 0.1}
        ... )
        >>> print(f"{result:.6f}")
        6.737947
    """
    parsed = SafeExpression.parse(expr)
    return parsed.evaluate(variables)


def safe_eval_array(expr: str, variables: dict[str, Any]) -> ArrayLike:
    """Safely evaluate a mathematical expression with array support.

    This function parses, validates, and evaluates a mathematical expression
    supporting numpy arrays for vectorized operations.

    Args:
        expr: Mathematical expression string
        variables: Dict mapping variable names to values (float or ndarray)

    Returns:
        ArrayLike: Evaluated result (scalar or array)

    Raises:
        ValueError: If expression contains disallowed operations
        ValueError: If referenced variable is not in bindings
        ValueError: If expression is syntactically invalid

    Example:
        >>> import numpy as np
        >>> x = np.linspace(0, 10, 100)
        >>> result = safe_eval_array("sin(x) + A * exp(-x)", {"x": x, "A": 1.0})
    """
    parsed = SafeArrayExpression.parse(expr)
    return parsed.evaluate(variables)
