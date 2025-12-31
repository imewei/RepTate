"""Security tests for path traversal vulnerabilities.

These tests verify that the SafePath utility correctly prevents
directory traversal attacks using various attack vectors.

OWASP Reference: A01:2021 - Broken Access Control
CWE Reference: CWE-22 - Improper Limitation of a Pathname to a Restricted Directory
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from RepTate.core.path_utils import PathTraversalError, SafePath

if TYPE_CHECKING:
    pass


class TestPathTraversalAttacks:
    """Test rejection of various path traversal attack vectors."""

    def test_reject_parent_directory_attack(self) -> None:
        """Test rejection of ../ parent directory traversal.

        Attack vector: ../../../etc/passwd
        Expected: PathTraversalError
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            safe = SafePath(tmpdir)

            with pytest.raises(PathTraversalError):
                safe.validate("../../../etc/passwd")

    def test_reject_relative_parent_traversal(self) -> None:
        """Test rejection of relative paths escaping base.

        Attack vector: subdir/../../outside
        Expected: PathTraversalError
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            safe = SafePath(tmpdir)

            with pytest.raises(PathTraversalError):
                safe.validate("subdir/../../outside")

    def test_reject_absolute_path_outside_base(self) -> None:
        """Test rejection of absolute paths outside base directory.

        Attack vector: /etc/passwd
        Expected: PathTraversalError
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            safe = SafePath(tmpdir)

            with pytest.raises(PathTraversalError):
                safe.validate("/etc/passwd")

    def test_reject_mixed_separator_attack(self) -> None:
        """Test rejection of attacks using mixed path separators.

        Attack vector: ..\\..\\..\\etc\\passwd (on Unix)
        Expected: PathTraversalError or treated as literal filename
        """
        import platform

        with tempfile.TemporaryDirectory() as tmpdir:
            safe = SafePath(tmpdir)

            if platform.system() == "Windows":
                # On Windows, backslashes work as separators
                with pytest.raises(PathTraversalError):
                    safe.validate("..\\..\\..\\windows\\system32")
            else:
                # On Unix, backslashes are literal characters in filenames
                # This creates a file with backslashes in the name, which is valid
                result = safe.validate("..\\..\\..\\windows\\system32")
                # Should be within base directory (backslashes are literal)
                assert result.parent.resolve() == Path(tmpdir).resolve()

    def test_reject_url_encoded_traversal(self) -> None:
        """Test handling of URL-encoded path components.

        Attack vector: %2e%2e%2f%2e%2e%2fetc%2fpasswd
        Note: URL encoding is NOT automatically decoded by filesystem operations.
        The percent signs are treated as literal characters in the filename.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            safe = SafePath(tmpdir)

            # URL encoding is NOT decoded - treated as literal filename
            # This is actually safe (percent signs are just filename characters)
            result = safe.validate("%2e%2e/%2e%2e/etc/passwd")
            # Should be within base directory (percent signs are literal)
            assert str(result).startswith(tmpdir)

    def test_reject_null_byte_injection(self) -> None:
        """Test rejection of null byte injection.

        Attack vector: allowed_file.txt\\x00../../etc/passwd
        Expected: Exception or rejection
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            safe = SafePath(tmpdir)

            # Null bytes should cause an error or be rejected
            with pytest.raises((ValueError, PathTraversalError)):
                safe.validate("allowed_file.txt\x00../../etc/passwd")

    def test_allow_valid_relative_path(self) -> None:
        """Test that valid relative paths within base are allowed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            safe = SafePath(tmpdir)

            # Create subdirectory
            subdir = Path(tmpdir) / "subdir"
            subdir.mkdir()

            # This should be allowed
            validated = safe.validate("subdir/file.txt")
            assert validated.parent == subdir

    def test_allow_valid_nested_path(self) -> None:
        """Test that valid nested paths are allowed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            safe = SafePath(tmpdir)

            # Create nested structure
            nested = Path(tmpdir) / "a" / "b" / "c"
            nested.mkdir(parents=True)

            # This should be allowed
            validated = safe.validate("a/b/c/file.txt")
            assert validated.parent == nested


class TestSymlinkAttacks:
    """Test symlink-based path traversal attacks."""

    def test_reject_symlink_to_outside_directory(self) -> None:
        """Test rejection of symlinks pointing outside base directory.

        Attack vector: Create symlink to /etc, then access via symlink
        Expected: PathTraversalError
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            safe = SafePath(tmpdir)
            base = Path(tmpdir)

            # Create symlink to /etc (or /tmp if /etc doesn't exist)
            target = Path("/tmp")  # Use /tmp for portability
            symlink = base / "link_to_tmp"

            try:
                symlink.symlink_to(target)

                # Accessing via symlink should be rejected
                with pytest.raises(PathTraversalError):
                    safe.validate("link_to_tmp/test.txt")

            except OSError:
                # Symlinks not supported on this platform
                pytest.skip("Symlinks not supported")

    def test_allow_symlink_within_base(self) -> None:
        """Test that symlinks within base directory are allowed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            safe = SafePath(tmpdir)
            base = Path(tmpdir)

            # Create subdirectory and symlink within base
            subdir = base / "subdir"
            subdir.mkdir()

            target_file = subdir / "target.txt"
            target_file.write_text("test")

            symlink = base / "link.txt"

            try:
                symlink.symlink_to(target_file)

                # This should be allowed (symlink stays within base)
                validated = safe.validate("link.txt")
                assert validated.resolve() == target_file.resolve()

            except OSError:
                # Symlinks not supported
                pytest.skip("Symlinks not supported")


class TestJoinMethod:
    """Test SafePath.join() method security."""

    def test_join_reject_parent_traversal(self) -> None:
        """Test that join() rejects parent directory traversal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            safe = SafePath(tmpdir)

            with pytest.raises(PathTraversalError):
                safe.join("..", "..", "etc", "passwd")

    def test_join_reject_absolute_path_component(self) -> None:
        """Test that join() rejects absolute path components."""
        with tempfile.TemporaryDirectory() as tmpdir:
            safe = SafePath(tmpdir)

            # Joining with absolute path should escape base
            with pytest.raises(PathTraversalError):
                safe.join("subdir", "/etc/passwd")

    def test_join_allows_safe_paths(self) -> None:
        """Test that join() allows safe path combinations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            safe = SafePath(tmpdir)

            # This should work
            result = safe.join("a", "b", "c", "file.txt")
            assert "file.txt" in str(result)


class TestEdgeCases:
    """Test edge cases and unusual inputs."""

    def test_empty_path_component(self) -> None:
        """Test handling of empty path components."""
        with tempfile.TemporaryDirectory() as tmpdir:
            safe = SafePath(tmpdir)

            # Empty string should resolve to base directory
            validated = safe.validate("")
            assert validated == Path(tmpdir).resolve()

    def test_dot_path_component(self) -> None:
        """Test handling of '.' path component."""
        with tempfile.TemporaryDirectory() as tmpdir:
            safe = SafePath(tmpdir)

            # Single dot should resolve to base directory
            validated = safe.validate(".")
            assert validated == Path(tmpdir).resolve()

    def test_multiple_slashes(self) -> None:
        """Test handling of multiple consecutive slashes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            safe = SafePath(tmpdir)

            # Multiple slashes should be normalized
            validated = safe.validate("subdir//file.txt")
            assert str(validated).count("//") == 0

    def test_unicode_path_components(self) -> None:
        """Test handling of Unicode in path components."""
        with tempfile.TemporaryDirectory() as tmpdir:
            safe = SafePath(tmpdir)

            # Unicode should be handled correctly
            validated = safe.validate("файл.txt")  # Russian
            assert validated.parent == Path(tmpdir).resolve()

    def test_very_long_path(self) -> None:
        """Test handling of very long path names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            safe = SafePath(tmpdir)

            # Very long path (but still valid)
            long_component = "a" * 200
            result = safe.join(long_component, "file.txt")

            # Should succeed or raise OSError (platform limit)
            assert result is not None


class TestExistsMethods:
    """Test SafePath.exists(), is_file(), is_dir() methods."""

    def test_exists_rejects_traversal(self) -> None:
        """Test that exists() rejects path traversal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            safe = SafePath(tmpdir)

            with pytest.raises(PathTraversalError):
                safe.exists("../../../etc/passwd")

    def test_is_file_rejects_traversal(self) -> None:
        """Test that is_file() rejects path traversal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            safe = SafePath(tmpdir)

            with pytest.raises(PathTraversalError):
                safe.is_file("../../../etc/passwd")

    def test_is_dir_rejects_traversal(self) -> None:
        """Test that is_dir() rejects path traversal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            safe = SafePath(tmpdir)

            with pytest.raises(PathTraversalError):
                safe.is_dir("../../../etc")

    def test_list_dir_rejects_traversal(self) -> None:
        """Test that list_dir() rejects path traversal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            safe = SafePath(tmpdir)

            with pytest.raises(PathTraversalError):
                safe.list_dir("../../../etc")


class TestRealWorldScenarios:
    """Test real-world attack scenarios."""

    def test_user_controlled_filename_attack(self) -> None:
        """Simulate attack via user-controlled filename input.

        Scenario: User uploads file with malicious name like "../../etc/passwd"
        Expected: Attack should be prevented
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            safe = SafePath(tmpdir)

            # Simulate user input
            user_filename = "../../etc/passwd"

            with pytest.raises(PathTraversalError):
                safe.validate(user_filename)

    def test_combined_attack_vector(self) -> None:
        """Test combined attack using multiple techniques.

        Attack: subdir/../../../etc/./passwd
        Expected: PathTraversalError
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            safe = SafePath(tmpdir)

            with pytest.raises(PathTraversalError):
                safe.validate("subdir/../../../etc/./passwd")

    def test_safe_usage_pattern(self) -> None:
        """Demonstrate correct safe usage pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            safe = SafePath(tmpdir)

            # Create allowed directory structure
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            test_file = data_dir / "test.txt"
            test_file.write_text("content")

            # Safe access pattern
            validated = safe.validate("data/test.txt")
            assert validated.exists()
            assert validated.is_file()

            # Read file safely
            content = validated.read_text()
            assert content == "content"
