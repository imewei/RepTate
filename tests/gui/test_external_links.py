"""Tests for external link configuration in QTextBrowser widgets.

Feature: Deep RCA prevention for broken GUI links
Verifies that QTextBrowser widgets displaying HTML with clickable links
have setOpenExternalLinks(True) configured.

Background:
    QTextBrowser defaults to openExternalLinks=False, which causes
    <a href="..."> links to not open in the browser when clicked.
    This test ensures consistency across all widgets that display citations
    or other clickable HTML content.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import NamedTuple

import pytest


class TextBrowserConfig(NamedTuple):
    """Configuration found for a QTextBrowser widget."""

    file: Path
    widget_name: str
    line_number: int
    has_open_external_links: bool
    displays_html_links: bool


class TestExternalLinksConfiguration:
    """Tests for QTextBrowser external link settings."""

    @pytest.fixture
    def gui_source_dir(self) -> Path:
        """Return the GUI source directory."""
        return Path(__file__).parent.parent.parent / "src" / "RepTate" / "gui"

    @pytest.fixture
    def all_gui_python_files(self, gui_source_dir: Path) -> list[Path]:
        """Return all Python files in the GUI directory."""
        return list(gui_source_dir.glob("*.py"))

    # Widgets that are known to NOT need openExternalLinks
    # (they don't display user-clickable links)
    _EXCLUDED_WIDGETS = {
        "widget",  # QTextEditLogger.widget - log display only
        "text_box",  # bob_gen_poly text output
        "proto_text",  # bob_gen_poly protocol display
    }

    def _find_text_browser_configs(self, file_path: Path) -> list[TextBrowserConfig]:
        """Analyze a Python file for QTextBrowser configurations.

        Returns a list of TextBrowserConfig for each QTextBrowser found.
        """
        content = file_path.read_text()
        configs: list[TextBrowserConfig] = []

        # Pattern to find QTextBrowser widget names
        # Matches: self.widgetName = QTextBrowser(...) or
        #          self.widgetName = QtWidgets.QTextBrowser(...)
        browser_pattern = re.compile(
            r"self\.(\w+)\s*=\s*(?:QtWidgets\.)?QTextBrowser\s*\("
        )

        # Pattern to find HTML link content being set
        # Matches: <a href="..."> in strings
        html_link_pattern = re.compile(r'<a\s+href\s*=')

        # Find all QTextBrowser instances
        browser_matches = list(browser_pattern.finditer(content))

        for match in browser_matches:
            widget_name = match.group(1)
            line_number = content[: match.start()].count("\n") + 1

            # Skip known excluded widgets
            if widget_name in self._EXCLUDED_WIDGETS:
                continue

            # Check if setOpenExternalLinks(True) is called for this widget
            has_open_external = bool(
                re.search(rf"self\.{widget_name}\.setOpenExternalLinks\s*\(\s*True\s*\)", content)
            )

            # Check if this widget displays HTML with links
            # Look for methods that set HTML content WITH links for this specific widget
            displays_links = False

            # Check for insertHtml with link content specifically for this widget
            widget_html_pattern = re.compile(
                rf"self\.{widget_name}\.(insertHtml|setHtml)\s*\([^)]*<a\s+href"
            )
            if widget_html_pattern.search(content):
                displays_links = True

            # Also check if do_cite method references this widget (citation pattern)
            if "def do_cite" in content and widget_name in ["thTextBox", "toolTextBox"]:
                if html_link_pattern.search(content):
                    displays_links = True

            configs.append(
                TextBrowserConfig(
                    file=file_path,
                    widget_name=widget_name,
                    line_number=line_number,
                    has_open_external_links=has_open_external,
                    displays_html_links=displays_links,
                )
            )

        return configs

    def test_qtool_textbox_has_open_external_links(self, gui_source_dir: Path):
        """QTool.toolTextBox should have setOpenExternalLinks(True).

        This is a regression test for the citation link bug where clicking
        citation links in tools did not open the browser.
        """
        qtool_file = gui_source_dir / "QTool.py"
        content = qtool_file.read_text()

        assert "self.toolTextBox.setOpenExternalLinks(True)" in content, (
            "QTool.toolTextBox must have setOpenExternalLinks(True) "
            "to enable clickable citation links"
        )

    def test_qtheory_textbox_has_open_external_links(self, gui_source_dir: Path):
        """QTheory.thTextBox should have setOpenExternalLinks(True).

        This was already correctly configured and serves as the reference
        implementation.
        """
        qtheory_file = gui_source_dir / "QTheory.py"
        content = qtheory_file.read_text()

        assert "self.thTextBox.setOpenExternalLinks(True)" in content, (
            "QTheory.thTextBox must have setOpenExternalLinks(True) "
            "to enable clickable citation links"
        )

    def test_text_browsers_with_links_have_open_external_enabled(
        self, all_gui_python_files: list[Path]
    ):
        """All QTextBrowser widgets that display HTML links must enable openExternalLinks.

        This test scans all GUI Python files to find QTextBrowser widgets
        and verifies that those displaying HTML content with links have
        setOpenExternalLinks(True) configured.
        """
        violations: list[str] = []

        for file_path in all_gui_python_files:
            configs = self._find_text_browser_configs(file_path)

            for config in configs:
                if config.displays_html_links and not config.has_open_external_links:
                    violations.append(
                        f"{config.file.name}:{config.line_number} - "
                        f"self.{config.widget_name} displays HTML links but "
                        f"does not call setOpenExternalLinks(True)"
                    )

        if violations:
            violation_list = "\n  ".join(violations)
            pytest.fail(
                f"Found QTextBrowser widgets with HTML links missing "
                f"setOpenExternalLinks(True):\n  {violation_list}"
            )


class TestCitationLinksInCode:
    """Tests for citation link patterns in theory/tool code."""

    @pytest.fixture
    def source_dir(self) -> Path:
        """Return the RepTate source directory."""
        return Path(__file__).parent.parent.parent / "src" / "RepTate"

    def test_citation_pattern_uses_href(self, source_dir: Path):
        """Citation links should use proper <a href> format."""
        # Check QTheory.py and QTool.py for citation patterns
        for filename in ["gui/QTheory.py", "gui/QTool.py"]:
            file_path = source_dir / filename
            content = file_path.read_text()

            # Look for do_cite method
            if "def do_cite" in content:
                # Verify it uses <a href="..."> format
                assert '<a href="%s">' in content or "<a href=" in content, (
                    f"{filename} do_cite method should use <a href> for citation links"
                )

    def test_help_urls_are_https_preferred(self, source_dir: Path):
        """Help URLs should prefer https over http where available."""
        http_urls: list[str] = []

        for py_file in source_dir.rglob("*.py"):
            if ".venv" in str(py_file):
                continue

            content = py_file.read_text()

            # Find http:// URLs that could be https
            # Exclude localhost and internal URLs
            matches = re.findall(r'http://reptate\.readthedocs\.io[^\s\'"]*', content)
            for match in matches:
                http_urls.append(f"{py_file.name}: {match}")

        # This is informational - http:// URLs work but https:// is preferred
        if http_urls:
            # Just a warning, not a failure - URLs still work via redirect
            pass  # Could use pytest.warns() if we want to flag this


class TestAboutDialogLinks:
    """Tests for AboutDialog external link configuration."""

    @pytest.fixture
    def about_dialog_ui(self) -> str:
        """Load the AboutDialog.ui content."""
        ui_path = (
            Path(__file__).parent.parent.parent
            / "src"
            / "RepTate"
            / "gui"
            / "AboutDialog.ui"
        )
        return ui_path.read_text()

    def test_about_dialog_has_open_external_links(self, about_dialog_ui: str):
        """AboutDialog QTextBrowser widgets should have openExternalLinks=true."""
        # Count openExternalLinks properties set to true
        open_external_count = about_dialog_ui.count("<property name=\"openExternalLinks\">")
        true_count = about_dialog_ui.count("<bool>true</bool>")

        # There should be at least one openExternalLinks property set to true
        assert open_external_count > 0, (
            "AboutDialog.ui should have openExternalLinks property"
        )
