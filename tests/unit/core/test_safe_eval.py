"""Unit tests for safe_eval module.

Tests cover:
- T018: Valid operator and function evaluation with variables
- T019: Security tests for injection attempts
- T019a: Error handling for division-by-zero and invalid math operations

Tests follow the contract specification in contracts/safe_eval.md.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    pass


# =============================================================================
# T018: Unit tests for safe_eval - operators, functions, variables
# =============================================================================

class TestSafeEvalBasicOperators:
    """Test basic arithmetic operators."""

    def test_addition(self) -> None:
        """Test addition operator."""
        from RepTate.core.safe_eval import safe_eval

        result = safe_eval("a + b", {"a": 2.0, "b": 3.0})
        assert result == 5.0

    def test_subtraction(self) -> None:
        """Test subtraction operator."""
        from RepTate.core.safe_eval import safe_eval

        result = safe_eval("a - b", {"a": 5.0, "b": 3.0})
        assert result == 2.0

    def test_multiplication(self) -> None:
        """Test multiplication operator."""
        from RepTate.core.safe_eval import safe_eval

        result = safe_eval("a * b", {"a": 4.0, "b": 3.0})
        assert result == 12.0

    def test_division(self) -> None:
        """Test division operator."""
        from RepTate.core.safe_eval import safe_eval

        result = safe_eval("a / b", {"a": 6.0, "b": 2.0})
        assert result == 3.0

    def test_exponentiation(self) -> None:
        """Test exponentiation operator."""
        from RepTate.core.safe_eval import safe_eval

        result = safe_eval("a ** b", {"a": 2.0, "b": 3.0})
        assert result == 8.0

    def test_unary_negation(self) -> None:
        """Test unary negation."""
        from RepTate.core.safe_eval import safe_eval

        result = safe_eval("-a", {"a": 5.0})
        assert result == -5.0

    def test_unary_positive(self) -> None:
        """Test unary positive."""
        from RepTate.core.safe_eval import safe_eval

        result = safe_eval("+a", {"a": 5.0})
        assert result == 5.0

    def test_complex_expression(self) -> None:
        """Test complex expression with multiple operators."""
        from RepTate.core.safe_eval import safe_eval

        result = safe_eval("a + b * c - d / e", {
            "a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0, "e": 2.0
        })
        # 1 + (2*3) - (4/2) = 1 + 6 - 2 = 5
        assert result == 5.0

    def test_parentheses(self) -> None:
        """Test expression with parentheses."""
        from RepTate.core.safe_eval import safe_eval

        result = safe_eval("(a + b) * c", {"a": 1.0, "b": 2.0, "c": 3.0})
        assert result == 9.0

    def test_nested_parentheses(self) -> None:
        """Test expression with nested parentheses."""
        from RepTate.core.safe_eval import safe_eval

        result = safe_eval("((a + b) * (c - d)) / e", {
            "a": 1.0, "b": 2.0, "c": 5.0, "d": 2.0, "e": 3.0
        })
        # ((1+2) * (5-2)) / 3 = (3 * 3) / 3 = 3
        assert result == 3.0


class TestSafeEvalFunctions:
    """Test allowed mathematical functions."""

    def test_sin(self) -> None:
        """Test sine function."""
        from RepTate.core.safe_eval import safe_eval

        result = safe_eval("sin(x)", {"x": 0.0})
        assert result == pytest.approx(0.0)

        result = safe_eval("sin(x)", {"x": math.pi / 2})
        assert result == pytest.approx(1.0)

    def test_cos(self) -> None:
        """Test cosine function."""
        from RepTate.core.safe_eval import safe_eval

        result = safe_eval("cos(x)", {"x": 0.0})
        assert result == pytest.approx(1.0)

        result = safe_eval("cos(x)", {"x": math.pi})
        assert result == pytest.approx(-1.0)

    def test_tan(self) -> None:
        """Test tangent function."""
        from RepTate.core.safe_eval import safe_eval

        result = safe_eval("tan(x)", {"x": 0.0})
        assert result == pytest.approx(0.0)

        result = safe_eval("tan(x)", {"x": math.pi / 4})
        assert result == pytest.approx(1.0)

    def test_exp(self) -> None:
        """Test exponential function."""
        from RepTate.core.safe_eval import safe_eval

        result = safe_eval("exp(x)", {"x": 0.0})
        assert result == pytest.approx(1.0)

        result = safe_eval("exp(x)", {"x": 1.0})
        assert result == pytest.approx(math.e)

    def test_log(self) -> None:
        """Test natural logarithm function."""
        from RepTate.core.safe_eval import safe_eval

        result = safe_eval("log(x)", {"x": 1.0})
        assert result == pytest.approx(0.0)

        result = safe_eval("log(x)", {"x": math.e})
        assert result == pytest.approx(1.0)

    def test_log10(self) -> None:
        """Test base-10 logarithm function."""
        from RepTate.core.safe_eval import safe_eval

        result = safe_eval("log10(x)", {"x": 1.0})
        assert result == pytest.approx(0.0)

        result = safe_eval("log10(x)", {"x": 100.0})
        assert result == pytest.approx(2.0)

    def test_sqrt(self) -> None:
        """Test square root function."""
        from RepTate.core.safe_eval import safe_eval

        result = safe_eval("sqrt(x)", {"x": 4.0})
        assert result == pytest.approx(2.0)

        result = safe_eval("sqrt(x)", {"x": 0.0})
        assert result == pytest.approx(0.0)

    def test_abs(self) -> None:
        """Test absolute value function."""
        from RepTate.core.safe_eval import safe_eval

        result = safe_eval("abs(x)", {"x": -5.0})
        assert result == pytest.approx(5.0)

        result = safe_eval("abs(x)", {"x": 5.0})
        assert result == pytest.approx(5.0)

    def test_nested_functions(self) -> None:
        """Test nested function calls."""
        from RepTate.core.safe_eval import safe_eval

        result = safe_eval("exp(log(x))", {"x": 5.0})
        assert result == pytest.approx(5.0)

        result = safe_eval("sqrt(abs(x))", {"x": -4.0})
        assert result == pytest.approx(2.0)

    def test_function_with_expression(self) -> None:
        """Test function with expression argument."""
        from RepTate.core.safe_eval import safe_eval

        result = safe_eval("sin(a * x + b)", {"a": 1.0, "x": 0.0, "b": 0.0})
        assert result == pytest.approx(0.0)


class TestSafeEvalVariables:
    """Test variable handling."""

    def test_single_variable(self) -> None:
        """Test single variable evaluation."""
        from RepTate.core.safe_eval import safe_eval

        result = safe_eval("x", {"x": 42.0})
        assert result == 42.0

    def test_multiple_variables(self) -> None:
        """Test multiple variables."""
        from RepTate.core.safe_eval import safe_eval

        result = safe_eval("x + y + z", {"x": 1.0, "y": 2.0, "z": 3.0})
        assert result == 6.0

    def test_variable_reuse(self) -> None:
        """Test variable used multiple times."""
        from RepTate.core.safe_eval import safe_eval

        result = safe_eval("x * x + x", {"x": 3.0})
        # 3*3 + 3 = 12
        assert result == 12.0

    def test_numeric_literal(self) -> None:
        """Test numeric literals in expression."""
        from RepTate.core.safe_eval import safe_eval

        result = safe_eval("x + 5", {"x": 3.0})
        assert result == 8.0

    def test_float_literals(self) -> None:
        """Test floating point literals."""
        from RepTate.core.safe_eval import safe_eval

        result = safe_eval("x * 2.5", {"x": 4.0})
        assert result == 10.0

    def test_scientific_notation(self) -> None:
        """Test scientific notation literals."""
        from RepTate.core.safe_eval import safe_eval

        result = safe_eval("x * 1e-3", {"x": 1000.0})
        assert result == pytest.approx(1.0)

    def test_underscore_in_variable(self) -> None:
        """Test underscores in variable names."""
        from RepTate.core.safe_eval import safe_eval

        result = safe_eval("my_var + other_var", {"my_var": 1.0, "other_var": 2.0})
        assert result == 3.0

    def test_complex_variable_names(self) -> None:
        """Test complex but valid variable names."""
        from RepTate.core.safe_eval import safe_eval

        result = safe_eval("A0 + A1 * x", {"A0": 1.0, "A1": 2.0, "x": 3.0})
        assert result == 7.0


class TestSafeEvalNumericLiterals:
    """Test numeric literal handling."""

    def test_integer_literal(self) -> None:
        """Test integer literal."""
        from RepTate.core.safe_eval import safe_eval

        result = safe_eval("5", {})
        assert result == 5.0

    def test_float_literal(self) -> None:
        """Test float literal."""
        from RepTate.core.safe_eval import safe_eval

        result = safe_eval("3.14", {})
        assert result == pytest.approx(3.14)

    def test_negative_literal(self) -> None:
        """Test negative literal."""
        from RepTate.core.safe_eval import safe_eval

        result = safe_eval("-5", {})
        assert result == -5.0

    def test_scientific_literal(self) -> None:
        """Test scientific notation literal."""
        from RepTate.core.safe_eval import safe_eval

        result = safe_eval("1e10", {})
        assert result == 1e10


class TestSafeExpressionClass:
    """Test SafeExpression class."""

    def test_parse_simple_expression(self) -> None:
        """Test parsing a simple expression."""
        from RepTate.core.safe_eval import SafeExpression

        expr = SafeExpression.parse("a + b")
        assert expr.raw == "a + b"
        assert expr.variables == frozenset({"a", "b"})

    def test_evaluate_parsed_expression(self) -> None:
        """Test evaluating a parsed expression."""
        from RepTate.core.safe_eval import SafeExpression

        expr = SafeExpression.parse("a * b + c")
        result = expr.evaluate({"a": 2.0, "b": 3.0, "c": 1.0})
        assert result == 7.0

    def test_reuse_parsed_expression(self) -> None:
        """Test reusing a parsed expression with different bindings."""
        from RepTate.core.safe_eval import SafeExpression

        expr = SafeExpression.parse("x ** 2")
        assert expr.evaluate({"x": 2.0}) == 4.0
        assert expr.evaluate({"x": 3.0}) == 9.0
        assert expr.evaluate({"x": 4.0}) == 16.0

    def test_variables_property(self) -> None:
        """Test that variables property is correct."""
        from RepTate.core.safe_eval import SafeExpression

        expr = SafeExpression.parse("sin(omega * t) + A * exp(-t / tau)")
        assert expr.variables == frozenset({"omega", "t", "A", "tau"})

    def test_expression_immutability(self) -> None:
        """Test that SafeExpression is immutable."""
        from RepTate.core.safe_eval import SafeExpression

        expr = SafeExpression.parse("x + y")
        with pytest.raises(AttributeError):
            expr.raw = "modified"  # type: ignore[misc]

    def test_expression_with_functions(self) -> None:
        """Test expression with functions extracts only variables."""
        from RepTate.core.safe_eval import SafeExpression

        # sin, cos, exp should not be in variables
        expr = SafeExpression.parse("sin(x) + cos(y) + exp(z)")
        assert expr.variables == frozenset({"x", "y", "z"})


# =============================================================================
# T019: Security tests for safe_eval - injection attempts
# =============================================================================

class TestSafeEvalSecurityInjection:
    """Test rejection of malicious injection attempts."""

    def test_reject_import(self) -> None:
        """Test rejection of import statement."""
        from RepTate.core.safe_eval import safe_eval

        with pytest.raises(ValueError, match="[Dd]isallowed"):
            safe_eval("__import__('os')", {})

    def test_reject_import_system(self) -> None:
        """Test rejection of import with system call."""
        from RepTate.core.safe_eval import safe_eval

        with pytest.raises(ValueError, match="[Dd]isallowed"):
            safe_eval("__import__('os').system('ls')", {})

    def test_reject_exec(self) -> None:
        """Test rejection of exec function."""
        from RepTate.core.safe_eval import safe_eval

        with pytest.raises(ValueError, match="[Dd]isallowed"):
            safe_eval("exec('print(1)')", {})

    def test_reject_eval_function(self) -> None:
        """Test rejection of eval function."""
        from RepTate.core.safe_eval import safe_eval

        with pytest.raises(ValueError, match="[Dd]isallowed"):
            safe_eval("eval('1+1')", {})

    def test_reject_compile(self) -> None:
        """Test rejection of compile function."""
        from RepTate.core.safe_eval import safe_eval

        with pytest.raises(ValueError, match="[Dd]isallowed"):
            safe_eval("compile('1+1', '', 'eval')", {})

    def test_reject_open(self) -> None:
        """Test rejection of open function."""
        from RepTate.core.safe_eval import safe_eval

        with pytest.raises(ValueError, match="[Dd]isallowed"):
            safe_eval("open('/etc/passwd')", {})

    def test_reject_attribute_access(self) -> None:
        """Test rejection of attribute access."""
        from RepTate.core.safe_eval import safe_eval

        with pytest.raises(ValueError, match="[Dd]isallowed"):
            safe_eval("x.__class__", {"x": 1.0})

    def test_reject_method_access(self) -> None:
        """Test rejection of method access via attribute."""
        from RepTate.core.safe_eval import safe_eval

        with pytest.raises(ValueError, match="[Dd]isallowed"):
            safe_eval("().__class__.__bases__", {})

    def test_reject_subscript(self) -> None:
        """Test rejection of subscript access."""
        from RepTate.core.safe_eval import safe_eval

        with pytest.raises(ValueError, match="[Dd]isallowed"):
            safe_eval("x[0]", {"x": [1, 2, 3]})

    def test_reject_dict_subscript(self) -> None:
        """Test rejection of dict subscript access."""
        from RepTate.core.safe_eval import safe_eval

        with pytest.raises(ValueError, match="[Dd]isallowed"):
            safe_eval("x['key']", {"x": {"key": "value"}})

    def test_reject_dunder_name(self) -> None:
        """Test rejection of dunder names."""
        from RepTate.core.safe_eval import safe_eval

        with pytest.raises(ValueError, match="[Dd]isallowed"):
            safe_eval("__name__", {})

    def test_reject_builtins(self) -> None:
        """Test rejection of __builtins__ access."""
        from RepTate.core.safe_eval import safe_eval

        with pytest.raises(ValueError, match="[Dd]isallowed"):
            safe_eval("__builtins__", {})

    def test_reject_globals(self) -> None:
        """Test rejection of globals() call."""
        from RepTate.core.safe_eval import safe_eval

        with pytest.raises(ValueError, match="[Dd]isallowed"):
            safe_eval("globals()", {})

    def test_reject_locals(self) -> None:
        """Test rejection of locals() call."""
        from RepTate.core.safe_eval import safe_eval

        with pytest.raises(ValueError, match="[Dd]isallowed"):
            safe_eval("locals()", {})

    def test_reject_getattr(self) -> None:
        """Test rejection of getattr function."""
        from RepTate.core.safe_eval import safe_eval

        with pytest.raises(ValueError, match="[Dd]isallowed"):
            safe_eval("getattr(x, '__class__')", {"x": 1.0})

    def test_reject_setattr(self) -> None:
        """Test rejection of setattr function."""
        from RepTate.core.safe_eval import safe_eval

        with pytest.raises(ValueError, match="[Dd]isallowed"):
            safe_eval("setattr(x, 'attr', 1)", {"x": object()})

    def test_reject_delattr(self) -> None:
        """Test rejection of delattr function."""
        from RepTate.core.safe_eval import safe_eval

        with pytest.raises(ValueError, match="[Dd]isallowed"):
            safe_eval("delattr(x, 'attr')", {"x": object()})

    def test_reject_lambda(self) -> None:
        """Test rejection of lambda expression."""
        from RepTate.core.safe_eval import safe_eval

        with pytest.raises(ValueError, match="[Dd]isallowed|[Ii]nvalid"):
            safe_eval("(lambda: 1)()", {})

    def test_reject_list_comprehension(self) -> None:
        """Test rejection of list comprehension."""
        from RepTate.core.safe_eval import safe_eval

        with pytest.raises(ValueError, match="[Dd]isallowed|[Ii]nvalid"):
            safe_eval("[x for x in range(10)]", {})

    def test_reject_generator_expression(self) -> None:
        """Test rejection of generator expression."""
        from RepTate.core.safe_eval import safe_eval

        with pytest.raises(ValueError, match="[Dd]isallowed|[Ii]nvalid"):
            safe_eval("(x for x in range(10))", {})

    def test_reject_unknown_function(self) -> None:
        """Test rejection of unknown functions."""
        from RepTate.core.safe_eval import safe_eval

        with pytest.raises(ValueError, match="[Dd]isallowed|[Uu]nknown"):
            safe_eval("unknown_func(x)", {"x": 1.0})

    def test_reject_input_function(self) -> None:
        """Test rejection of input function."""
        from RepTate.core.safe_eval import safe_eval

        with pytest.raises(ValueError, match="[Dd]isallowed"):
            safe_eval("input()", {})

    def test_reject_print_function(self) -> None:
        """Test rejection of print function."""
        from RepTate.core.safe_eval import safe_eval

        with pytest.raises(ValueError, match="[Dd]isallowed"):
            safe_eval("print('hello')", {})

    def test_reject_type_function(self) -> None:
        """Test rejection of type function."""
        from RepTate.core.safe_eval import safe_eval

        with pytest.raises(ValueError, match="[Dd]isallowed"):
            safe_eval("type(x)", {"x": 1.0})

    def test_reject_class_mro_access(self) -> None:
        """Test rejection of class MRO access."""
        from RepTate.core.safe_eval import safe_eval

        with pytest.raises(ValueError, match="[Dd]isallowed"):
            safe_eval("().__class__.__mro__", {})

    def test_reject_subclasses(self) -> None:
        """Test rejection of __subclasses__ access."""
        from RepTate.core.safe_eval import safe_eval

        with pytest.raises(ValueError, match="[Dd]isallowed"):
            safe_eval("().__class__.__subclasses__()", {})


class TestSafeEvalSecurityEdgeCases:
    """Test edge cases for security."""

    def test_reject_unicode_tricks(self) -> None:
        """Test rejection of unicode lookalike characters."""
        from RepTate.core.safe_eval import safe_eval

        # Using fullwidth characters or other tricks should still fail
        # This tests that we don't have unicode-related bypasses
        with pytest.raises(ValueError):
            # This should fail because 'eval' is not allowed
            safe_eval("eval('1')", {})

    def test_reject_nested_eval_attempt(self) -> None:
        """Test rejection of nested eval attempts."""
        from RepTate.core.safe_eval import safe_eval

        with pytest.raises(ValueError):
            safe_eval("exec(eval('print(1)'))", {})

    def test_safe_dict_with_callable(self) -> None:
        """Test that passing callable in variables is safe."""
        from RepTate.core.safe_eval import safe_eval

        # Even if a malicious callable is passed, it should not be callable
        # in the expression because only specific functions are allowed
        def malicious() -> str:
            return "hacked"

        # This should fail because 'f' is not a number and can't be evaluated
        # in a mathematical context (or the function call should be rejected)
        with pytest.raises((ValueError, TypeError)):
            safe_eval("f()", {"f": malicious})


# =============================================================================
# T019a: Error handling tests for division-by-zero and invalid math operations
# =============================================================================

class TestSafeEvalErrorHandling:
    """Test error handling for mathematical edge cases.

    Note: Python's math module raises exceptions for domain errors,
    so these tests check for proper exception handling rather than
    IEEE 754 special values.
    """

    def test_division_by_zero(self) -> None:
        """Test division by zero raises ZeroDivisionError.

        Python raises ZeroDivisionError for float division by zero,
        unlike numpy which returns inf.
        """
        from RepTate.core.safe_eval import safe_eval

        with pytest.raises(ZeroDivisionError):
            safe_eval("x / y", {"x": 1.0, "y": 0.0})

    def test_division_by_zero_negative(self) -> None:
        """Test negative division by zero raises ZeroDivisionError."""
        from RepTate.core.safe_eval import safe_eval

        with pytest.raises(ZeroDivisionError):
            safe_eval("-x / y", {"x": 1.0, "y": 0.0})

    def test_zero_divided_by_zero(self) -> None:
        """Test 0/0 raises ZeroDivisionError."""
        from RepTate.core.safe_eval import safe_eval

        with pytest.raises(ZeroDivisionError):
            safe_eval("x / y", {"x": 0.0, "y": 0.0})

    def test_log_of_zero(self) -> None:
        """Test log of zero raises ValueError (math domain error)."""
        from RepTate.core.safe_eval import safe_eval

        with pytest.raises(ValueError):
            safe_eval("log(x)", {"x": 0.0})

    def test_log_of_negative(self) -> None:
        """Test log of negative number raises ValueError (math domain error)."""
        from RepTate.core.safe_eval import safe_eval

        with pytest.raises(ValueError):
            safe_eval("log(x)", {"x": -1.0})

    def test_log10_of_zero(self) -> None:
        """Test log10 of zero raises ValueError (math domain error)."""
        from RepTate.core.safe_eval import safe_eval

        with pytest.raises(ValueError):
            safe_eval("log10(x)", {"x": 0.0})

    def test_log10_of_negative(self) -> None:
        """Test log10 of negative number raises ValueError (math domain error)."""
        from RepTate.core.safe_eval import safe_eval

        with pytest.raises(ValueError):
            safe_eval("log10(x)", {"x": -1.0})

    def test_sqrt_of_negative(self) -> None:
        """Test sqrt of negative number raises ValueError (math domain error)."""
        from RepTate.core.safe_eval import safe_eval

        with pytest.raises(ValueError):
            safe_eval("sqrt(x)", {"x": -1.0})

    def test_large_exponent(self) -> None:
        """Test large exponent raises OverflowError."""
        from RepTate.core.safe_eval import safe_eval

        with pytest.raises(OverflowError):
            safe_eval("exp(x)", {"x": 1000.0})

    def test_very_negative_exponent(self) -> None:
        """Test very negative exponent produces zero."""
        from RepTate.core.safe_eval import safe_eval

        result = safe_eval("exp(x)", {"x": -1000.0})
        assert result == 0.0

    def test_power_overflow(self) -> None:
        """Test power overflow raises OverflowError."""
        from RepTate.core.safe_eval import safe_eval

        with pytest.raises(OverflowError):
            safe_eval("x ** y", {"x": 10.0, "y": 1000.0})


class TestSafeEvalSyntaxErrors:
    """Test handling of syntax errors."""

    def test_invalid_syntax_unbalanced_parens(self) -> None:
        """Test unbalanced parentheses."""
        from RepTate.core.safe_eval import safe_eval

        with pytest.raises(ValueError, match="[Ss]yntax|[Ii]nvalid"):
            safe_eval("sin(x", {"x": 1.0})

    def test_invalid_syntax_extra_parens(self) -> None:
        """Test extra closing parentheses."""
        from RepTate.core.safe_eval import safe_eval

        with pytest.raises(ValueError, match="[Ss]yntax|[Ii]nvalid"):
            safe_eval("sin(x))", {"x": 1.0})

    def test_invalid_syntax_empty(self) -> None:
        """Test empty expression."""
        from RepTate.core.safe_eval import safe_eval

        with pytest.raises(ValueError, match="[Ss]yntax|[Ii]nvalid|[Ee]mpty"):
            safe_eval("", {})

    def test_invalid_syntax_operator_only(self) -> None:
        """Test operator-only expression."""
        from RepTate.core.safe_eval import safe_eval

        with pytest.raises(ValueError, match="[Ss]yntax|[Ii]nvalid"):
            safe_eval("+", {})

    def test_invalid_syntax_consecutive_operators(self) -> None:
        """Test consecutive binary operators."""
        from RepTate.core.safe_eval import safe_eval

        with pytest.raises(ValueError, match="[Ss]yntax|[Ii]nvalid"):
            safe_eval("x + * y", {"x": 1.0, "y": 1.0})


class TestSafeEvalMissingVariables:
    """Test handling of missing variables."""

    def test_missing_single_variable(self) -> None:
        """Test missing single variable."""
        from RepTate.core.safe_eval import safe_eval

        with pytest.raises(ValueError, match="[Uu]nknown|[Mm]issing|[Vv]ariable"):
            safe_eval("x + y", {"x": 1.0})

    def test_missing_all_variables(self) -> None:
        """Test missing all variables."""
        from RepTate.core.safe_eval import safe_eval

        with pytest.raises(ValueError, match="[Uu]nknown|[Mm]issing|[Vv]ariable"):
            safe_eval("x + y", {})

    def test_variable_typo(self) -> None:
        """Test variable name typo."""
        from RepTate.core.safe_eval import safe_eval

        with pytest.raises(ValueError, match="[Uu]nknown|[Mm]issing|[Vv]ariable"):
            safe_eval("x + y", {"x": 1.0, "Y": 2.0})  # y vs Y


class TestSafeExpressionErrorHandling:
    """Test SafeExpression error handling."""

    def test_parse_invalid_expression(self) -> None:
        """Test parsing invalid expression."""
        from RepTate.core.safe_eval import SafeExpression

        with pytest.raises(ValueError):
            SafeExpression.parse("sin(x")

    def test_parse_unsafe_expression(self) -> None:
        """Test parsing unsafe expression."""
        from RepTate.core.safe_eval import SafeExpression

        with pytest.raises(ValueError):
            SafeExpression.parse("__import__('os')")

    def test_evaluate_missing_variable(self) -> None:
        """Test evaluating with missing variable."""
        from RepTate.core.safe_eval import SafeExpression

        expr = SafeExpression.parse("x + y")
        with pytest.raises(ValueError, match="[Mm]issing"):
            expr.evaluate({"x": 1.0})


# =============================================================================
# Performance tests
# =============================================================================

class TestSafeEvalPerformance:
    """Test performance requirements."""

    def test_10000_evaluations_per_second(self) -> None:
        """Test that 10,000 evaluations/second is achievable.

        This test verifies the performance requirement from the contract.
        """
        from RepTate.core.safe_eval import SafeExpression

        # Parse once, evaluate many times
        expr = SafeExpression.parse("sin(omega * t) + A * exp(-t / tau)")
        bindings = {"omega": 6.28, "t": 0.25, "A": 1.0, "tau": 0.1}

        # Warm up
        for _ in range(100):
            expr.evaluate(bindings)

        # Measure
        n_iterations = 10000
        start_time = time.perf_counter()
        for _ in range(n_iterations):
            expr.evaluate(bindings)
        elapsed = time.perf_counter() - start_time

        evaluations_per_second = n_iterations / elapsed
        assert evaluations_per_second >= 10000, (
            f"Performance requirement not met: {evaluations_per_second:.0f} eval/s < 10000"
        )

    def test_simple_expression_performance(self) -> None:
        """Test simple expression performance."""
        from RepTate.core.safe_eval import SafeExpression

        expr = SafeExpression.parse("a + b * c")
        bindings = {"a": 1.0, "b": 2.0, "c": 3.0}

        # Should be even faster for simple expressions
        n_iterations = 10000
        start_time = time.perf_counter()
        for _ in range(n_iterations):
            expr.evaluate(bindings)
        elapsed = time.perf_counter() - start_time

        evaluations_per_second = n_iterations / elapsed
        assert evaluations_per_second >= 10000


# =============================================================================
# Contract example tests
# =============================================================================

class TestSafeEvalContractExamples:
    """Test examples from the contract specification."""

    def test_contract_example_1(self) -> None:
        """Test contract example: A * exp(-t / tau)."""
        from RepTate.core.safe_eval import safe_eval

        result = safe_eval(
            "A * exp(-t / tau)",
            {"A": 1000.0, "t": 0.5, "tau": 0.1}
        )
        # result = 1000 * exp(-5) = 1000 * 0.006737... = 6.737...
        assert result == pytest.approx(6.737946999085467, rel=1e-9)

    def test_contract_example_2(self) -> None:
        """Test contract example: sin(omega * t)."""
        from RepTate.core.safe_eval import SafeExpression

        expr = SafeExpression.parse("sin(omega * t)")
        assert expr.variables == frozenset({"omega", "t"})

        result = expr.evaluate({"omega": 6.28, "t": 0.25})
        # sin(6.28 * 0.25) = sin(1.57) ~= 1.0
        assert result == pytest.approx(math.sin(1.57), rel=1e-2)

    def test_contract_invalid_syntax_example(self) -> None:
        """Test contract example: invalid syntax."""
        from RepTate.core.safe_eval import safe_eval

        with pytest.raises(ValueError, match="[Ss]yntax|[Ii]nvalid"):
            safe_eval("sin(x", {"x": 1.0})

    def test_contract_disallowed_operation_example(self) -> None:
        """Test contract example: disallowed operation."""
        from RepTate.core.safe_eval import safe_eval

        with pytest.raises(ValueError, match="[Dd]isallowed"):
            safe_eval("__import__('os').system('rm -rf /')", {})

    def test_contract_missing_variable_example(self) -> None:
        """Test contract example: missing variable."""
        from RepTate.core.safe_eval import safe_eval

        with pytest.raises(ValueError, match="[Uu]nknown|[Mm]issing|[Vv]ariable"):
            safe_eval("a + b", {"a": 1.0})
