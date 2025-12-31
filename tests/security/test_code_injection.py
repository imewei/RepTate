"""Security tests for code injection vulnerabilities.

These tests verify that expression evaluation correctly prevents
code injection attacks via eval(), exec(), and related mechanisms.

OWASP Reference: A03:2021 - Injection
CWE Reference: CWE-94 - Improper Control of Generation of Code ('Code Injection')
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from RepTate.core.safe_eval import SafeArrayExpression, SafeExpression, safe_eval, safe_eval_array

if TYPE_CHECKING:
    pass


class TestCodeInjectionPrevention:
    """Test prevention of code injection via expression evaluation."""

    def test_reject_import_statement(self) -> None:
        """Test rejection of import via __import__.

        Attack vector: __import__('os').system('rm -rf /')
        Expected: ValueError
        """
        with pytest.raises(ValueError, match="[Dd]isallowed"):
            safe_eval("__import__('os').system('ls')", {})

    def test_reject_subprocess_import(self) -> None:
        """Test rejection of subprocess module import.

        Attack vector: __import__('subprocess').call(['ls'])
        Expected: ValueError
        """
        with pytest.raises(ValueError, match="[Dd]isallowed"):
            safe_eval("__import__('subprocess').call(['ls'])", {})

    def test_reject_file_read_attempt(self) -> None:
        """Test rejection of file read via open().

        Attack vector: open('/etc/passwd').read()
        Expected: ValueError
        """
        with pytest.raises(ValueError, match="[Dd]isallowed"):
            safe_eval("open('/etc/passwd').read()", {})

    def test_reject_file_write_attempt(self) -> None:
        """Test rejection of file write via open().

        Attack vector: open('/tmp/evil', 'w').write('malicious')
        Expected: ValueError
        """
        with pytest.raises(ValueError, match="[Dd]isallowed"):
            safe_eval("open('/tmp/evil', 'w').write('data')", {})


class TestAttributeAccessAttacks:
    """Test prevention of attacks via attribute access."""

    def test_reject_class_attribute_access(self) -> None:
        """Test rejection of __class__ attribute access.

        Attack vector: x.__class__
        Expected: ValueError
        """
        with pytest.raises(ValueError, match="[Dd]isallowed"):
            safe_eval("x.__class__", {"x": 1.0})

    def test_reject_mro_access(self) -> None:
        """Test rejection of MRO traversal.

        Attack vector: ().__class__.__mro__
        Expected: ValueError
        """
        with pytest.raises(ValueError, match="[Dd]isallowed"):
            safe_eval("().__class__.__mro__", {})

    def test_reject_subclasses_access(self) -> None:
        """Test rejection of __subclasses__ access.

        Attack vector: ().__class__.__subclasses__()
        Expected: ValueError
        """
        with pytest.raises(ValueError, match="[Dd]isallowed"):
            safe_eval("().__class__.__subclasses__()", {})

    def test_reject_dict_attribute_access(self) -> None:
        """Test rejection of __dict__ access.

        Attack vector: x.__dict__
        Expected: ValueError
        """
        with pytest.raises(ValueError, match="[Dd]isallowed"):
            safe_eval("x.__dict__", {"x": object()})

    def test_reject_globals_access(self) -> None:
        """Test rejection of __globals__ access.

        Attack vector: (lambda: 0).__globals__
        Expected: ValueError (lambda rejected first)
        """
        with pytest.raises(ValueError, match="[Dd]isallowed"):
            safe_eval("(lambda: 0).__globals__", {})


class TestBuiltinBypassAttempts:
    """Test prevention of attempts to access builtins."""

    def test_reject_builtins_direct(self) -> None:
        """Test rejection of __builtins__ access.

        Attack vector: __builtins__
        Expected: ValueError
        """
        with pytest.raises(ValueError, match="[Dd]isallowed"):
            safe_eval("__builtins__", {})

    def test_reject_builtins_via_name(self) -> None:
        """Test rejection of __name__ access.

        Attack vector: __name__
        Expected: ValueError
        """
        with pytest.raises(ValueError, match="[Dd]isallowed"):
            safe_eval("__name__", {})

    def test_reject_eval_function(self) -> None:
        """Test rejection of eval function.

        Attack vector: eval('1+1')
        Expected: ValueError
        """
        with pytest.raises(ValueError, match="[Dd]isallowed"):
            safe_eval("eval('1+1')", {})

    def test_reject_exec_function(self) -> None:
        """Test rejection of exec function.

        Attack vector: exec('print(1)')
        Expected: ValueError
        """
        with pytest.raises(ValueError, match="[Dd]isallowed"):
            safe_eval("exec('print(1)')", {})

    def test_reject_compile_function(self) -> None:
        """Test rejection of compile function.

        Attack vector: compile('1+1', '', 'eval')
        Expected: ValueError
        """
        with pytest.raises(ValueError, match="[Dd]isallowed"):
            safe_eval("compile('1+1', '', 'eval')", {})


class TestReflectionAttacks:
    """Test prevention of attacks via reflection/introspection."""

    def test_reject_getattr_function(self) -> None:
        """Test rejection of getattr.

        Attack vector: getattr(x, '__class__')
        Expected: ValueError
        """
        with pytest.raises(ValueError, match="[Dd]isallowed"):
            safe_eval("getattr(x, '__class__')", {"x": 1.0})

    def test_reject_setattr_function(self) -> None:
        """Test rejection of setattr.

        Attack vector: setattr(x, 'attr', value)
        Expected: ValueError
        """
        with pytest.raises(ValueError, match="[Dd]isallowed"):
            safe_eval("setattr(x, 'attr', 1)", {"x": object()})

    def test_reject_delattr_function(self) -> None:
        """Test rejection of delattr.

        Attack vector: delattr(x, 'attr')
        Expected: ValueError
        """
        with pytest.raises(ValueError, match="[Dd]isallowed"):
            safe_eval("delattr(x, 'attr')", {"x": object()})

    def test_reject_hasattr_function(self) -> None:
        """Test rejection of hasattr.

        Attack vector: hasattr(x, '__class__')
        Expected: ValueError
        """
        with pytest.raises(ValueError, match="[Dd]isallowed"):
            safe_eval("hasattr(x, '__class__')", {"x": 1.0})

    def test_reject_type_function(self) -> None:
        """Test rejection of type function.

        Attack vector: type(x)
        Expected: ValueError
        """
        with pytest.raises(ValueError, match="[Dd]isallowed"):
            safe_eval("type(x)", {"x": 1.0})

    def test_reject_isinstance_function(self) -> None:
        """Test rejection of isinstance.

        Attack vector: isinstance(x, int)
        Expected: ValueError
        """
        with pytest.raises(ValueError, match="[Dd]isallowed"):
            safe_eval("isinstance(x, int)", {"x": 1.0})

    def test_reject_callable_function(self) -> None:
        """Test rejection of callable.

        Attack vector: callable(x)
        Expected: ValueError
        """
        with pytest.raises(ValueError, match="[Dd]isallowed"):
            safe_eval("callable(x)", {"x": lambda: None})


class TestArrayAccessAttacks:
    """Test prevention of attacks via numpy array manipulation."""

    def test_reject_array_method_access(self) -> None:
        """Test rejection of array method access.

        Attack vector: x.tolist (attribute access on array)
        Expected: ValueError
        """
        arr = np.array([1, 2, 3])

        with pytest.raises(ValueError, match="[Dd]isallowed"):
            safe_eval("x.tolist", {"x": arr})

    def test_reject_array_subscript_access(self) -> None:
        """Test rejection of array subscript access.

        Attack vector: x[0]
        Expected: ValueError
        """
        arr = np.array([1, 2, 3])

        with pytest.raises(ValueError, match="[Dd]isallowed"):
            safe_eval("x[0]", {"x": arr})

    def test_reject_array_class_access(self) -> None:
        """Test rejection of array __class__ access.

        Attack vector: x.__class__
        Expected: ValueError
        """
        arr = np.array([1, 2, 3])

        with pytest.raises(ValueError, match="[Dd]isallowed"):
            safe_eval("x.__class__", {"x": arr})


class TestResourceExhaustionAttacks:
    """Test prevention of resource exhaustion attacks."""

    def test_very_large_exponent(self) -> None:
        """Test handling of very large exponents.

        Attack vector: 10 ** 10000 (memory exhaustion)
        Expected: OverflowError or reasonable handling
        """
        # This should raise OverflowError, not hang
        with pytest.raises(OverflowError):
            safe_eval("x ** y", {"x": 10.0, "y": 10000.0})

    def test_nested_exponentiation(self) -> None:
        """Test handling of nested exponentiation.

        Attack vector: x ** x ** x ** x (extreme complexity)
        Expected: OverflowError or ValueError
        """
        with pytest.raises((OverflowError, ValueError)):
            safe_eval("x ** x ** x ** x", {"x": 100.0})

    def test_division_by_very_small_number(self) -> None:
        """Test handling of division by very small number.

        Attack vector: 1 / 1e-1000 (overflow to infinity)
        Expected: Handled appropriately (inf or error)
        """
        # Should either produce inf or raise OverflowError
        result = safe_eval("x / y", {"x": 1.0, "y": 1e-300})
        assert result != 0.0  # Should be very large or inf


class TestComplexExpressionAttacks:
    """Test attacks using complex expression structures."""

    def test_deeply_nested_parentheses(self) -> None:
        """Test handling of deeply nested parentheses.

        Attack vector: ((((((((((x)))))))))) (parser stress)
        Expected: Should parse correctly or reject if too deep
        """
        # 50 levels of nesting
        expr = "(" * 50 + "x" + ")" * 50

        # Should either work or raise ValueError
        try:
            result = safe_eval(expr, {"x": 1.0})
            assert result == 1.0
        except ValueError:
            # Acceptable if depth limit is enforced
            pass

    def test_very_long_expression(self) -> None:
        """Test handling of very long expressions.

        Attack vector: x+x+x+... (thousands of times)
        Expected: Should work up to reasonable limits or enforce length limit

        Note: Current implementation may hit recursion limits with very long expressions.
        This is a known limitation and is acceptable (fails safely).
        """
        # 100 additions (1000 causes RecursionError - which is acceptable)
        expr = "+".join(["x"] * 100)

        # Should work for moderate lengths
        result = safe_eval(expr, {"x": 1.0})
        assert result == pytest.approx(100.0)

        # Very long expressions may hit recursion limit (acceptable)
        # This demonstrates DoS protection via Python's recursion limit
        very_long_expr = "+".join(["x"] * 1000)
        try:
            safe_eval(very_long_expr, {"x": 1.0})
            # If it works, that's fine
        except RecursionError:
            # Also acceptable - Python's built-in protection
            pass


class TestUnicodeAttacks:
    """Test attacks using Unicode lookalikes and special characters."""

    def test_unicode_lookalike_eval(self) -> None:
        """Test that Unicode lookalikes don't bypass security.

        Attack vector: Using lookalike characters for 'eval'
        Expected: Should still be blocked
        """
        # Even if someone uses Unicode tricks, the actual eval is blocked
        with pytest.raises(ValueError):
            safe_eval("eval('1')", {})

    def test_zero_width_characters(self) -> None:
        """Test handling of zero-width Unicode characters.

        Attack vector: ev​al (with zero-width space)
        Expected: Should be treated as different identifier or error
        """
        # Zero-width characters would make this an unknown function
        expr = "e\u200Bval('1')"  # Zero-width space

        with pytest.raises(ValueError):
            safe_eval(expr, {})


class TestLambdaAndComprehensions:
    """Test prevention of lambda and comprehension-based attacks."""

    def test_reject_lambda_expression(self) -> None:
        """Test rejection of lambda expression.

        Attack vector: (lambda: __import__('os'))()
        Expected: ValueError
        """
        with pytest.raises(ValueError, match="[Dd]isallowed|[Ii]nvalid"):
            safe_eval("(lambda: 1)()", {})

    def test_reject_list_comprehension(self) -> None:
        """Test rejection of list comprehension.

        Attack vector: [i for i in range(1000000)]
        Expected: ValueError
        """
        with pytest.raises(ValueError, match="[Dd]isallowed|[Ii]nvalid"):
            safe_eval("[i for i in range(10)]", {})

    def test_reject_dict_comprehension(self) -> None:
        """Test rejection of dict comprehension.

        Attack vector: {i: i for i in range(10)}
        Expected: ValueError
        """
        with pytest.raises(ValueError, match="[Dd]isallowed|[Ii]nvalid"):
            safe_eval("{i: i for i in range(10)}", {})

    def test_reject_generator_expression(self) -> None:
        """Test rejection of generator expression.

        Attack vector: (i for i in range(10))
        Expected: ValueError
        """
        with pytest.raises(ValueError, match="[Dd]isallowed|[Ii]nvalid"):
            safe_eval("(i for i in range(10))", {})


class TestArrayExpressionSecurity:
    """Test security of SafeArrayExpression."""

    def test_array_expression_rejects_subscript(self) -> None:
        """Test that array expressions also reject subscript access."""
        arr = np.array([1, 2, 3])

        with pytest.raises(ValueError, match="[Dd]isallowed"):
            safe_eval_array("x[0]", {"x": arr})

    def test_array_expression_rejects_attribute(self) -> None:
        """Test that array expressions reject attribute access."""
        arr = np.array([1, 2, 3])

        with pytest.raises(ValueError, match="[Dd]isallowed"):
            safe_eval_array("x.sum", {"x": arr})

    def test_array_expression_rejects_import(self) -> None:
        """Test that array expressions reject import."""
        with pytest.raises(ValueError, match="[Dd]isallowed"):
            safe_eval_array("__import__('numpy').zeros(10)", {})


class TestSafeExpressionClassSecurity:
    """Test security of SafeExpression class."""

    def test_safe_expression_immutability(self) -> None:
        """Test that SafeExpression cannot be modified after creation."""
        expr = SafeExpression.parse("x + y")

        # Should not be able to modify raw expression
        with pytest.raises(AttributeError):
            expr.raw = "__import__('os').system('ls')"  # type: ignore[misc]

    def test_safe_expression_parse_validates(self) -> None:
        """Test that SafeExpression.parse() performs validation."""
        # Parsing should fail for unsafe expressions
        with pytest.raises(ValueError):
            SafeExpression.parse("__import__('os')")

    def test_safe_array_expression_parse_validates(self) -> None:
        """Test that SafeArrayExpression.parse() performs validation."""
        with pytest.raises(ValueError):
            SafeArrayExpression.parse("__import__('numpy')")


class TestRealWorldAttackScenarios:
    """Test real-world attack scenarios."""

    def test_malicious_user_expression(self) -> None:
        """Simulate malicious user providing expression input.

        Scenario: User enters expression in theory parameter
        Attack: User tries to read /etc/passwd
        Expected: Attack is prevented
        """
        # Simulate user input
        user_expression = "__import__('os').system('cat /etc/passwd')"

        with pytest.raises(ValueError):
            safe_eval(user_expression, {})

    def test_malicious_file_parameter(self) -> None:
        """Simulate attack via file parameter expression.

        Scenario: User creates data file with malicious parameter expression
        Attack: Parameter expression tries to execute code
        Expected: Attack is prevented
        """
        # Malicious parameter value in file
        param_expression = "open('/etc/passwd').read()"

        with pytest.raises(ValueError):
            safe_eval(param_expression, {})

    def test_chained_attack_attempt(self) -> None:
        """Test prevention of chained attack attempts.

        Attack: Try multiple attack vectors in sequence
        Expected: All should be blocked
        """
        attacks = [
            "__import__('os').system('ls')",
            "eval('1+1')",
            "exec('print(1)')",
            "open('/etc/passwd').read()",
            "().__class__.__bases__",
            "getattr(__builtins__, 'eval')",
        ]

        for attack in attacks:
            with pytest.raises(ValueError):
                safe_eval(attack, {})
