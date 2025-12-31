#!/usr/bin/env python3
"""AST-based documentation gap analysis for RepTate."""

import ast
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Set
from collections import defaultdict


@dataclass
class DocItem:
    """Represents a documentable code item."""
    name: str
    type: str  # 'class', 'function', 'method'
    file: str
    line: int
    has_docstring: bool
    docstring: str = ""
    params: List[str] = field(default_factory=list)
    returns: bool = False
    is_public: bool = True
    parent: str = ""


class DocAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze documentation coverage."""

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.items: List[DocItem] = []
        self.current_class = None

    def _is_public(self, name: str) -> bool:
        """Check if name is public (doesn't start with _)."""
        return not name.startswith('_')

    def _extract_params(self, node: ast.FunctionDef) -> List[str]:
        """Extract parameter names from function definition."""
        params = []
        for arg in node.args.args:
            if arg.arg != 'self' and arg.arg != 'cls':
                params.append(arg.arg)
        return params

    def _has_return(self, node: ast.FunctionDef) -> bool:
        """Check if function has return statement."""
        for child in ast.walk(node):
            if isinstance(child, ast.Return) and child.value is not None:
                return True
        return False

    def _check_docstring_completeness(self, docstring: str, params: List[str], has_return: bool) -> Dict[str, bool]:
        """Check if docstring documents params and returns."""
        if not docstring:
            return {"has_params": False, "has_returns": False}

        doc_lower = docstring.lower()

        # Check for common parameter documentation patterns
        has_params_doc = False
        if params:
            # Look for common param documentation patterns
            param_indicators = ['parameters:', 'params:', 'args:', 'arguments:']
            has_params_doc = any(indicator in doc_lower for indicator in param_indicators)

            # Also check if individual params are mentioned
            if not has_params_doc:
                mentioned_params = sum(1 for p in params if p in doc_lower)
                has_params_doc = mentioned_params >= len(params) * 0.5  # At least 50% mentioned
        else:
            has_params_doc = True  # No params to document

        # Check for return documentation
        has_returns_doc = False
        if has_return:
            return_indicators = ['returns:', 'return:', 'yields:', 'yield:']
            has_returns_doc = any(indicator in doc_lower for indicator in return_indicators)
        else:
            has_returns_doc = True  # No return to document

        return {"has_params": has_params_doc, "has_returns": has_returns_doc}

    def visit_ClassDef(self, node: ast.ClassDef):
        """Visit class definition."""
        docstring = ast.get_docstring(node) or ""
        is_public = self._is_public(node.name)

        item = DocItem(
            name=node.name,
            type='class',
            file=str(self.filepath),
            line=node.lineno,
            has_docstring=bool(docstring),
            docstring=docstring,
            is_public=is_public
        )
        self.items.append(item)

        # Visit methods
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Visit function/method definition."""
        docstring = ast.get_docstring(node) or ""
        is_public = self._is_public(node.name)
        params = self._extract_params(node)
        has_return = self._has_return(node)

        item = DocItem(
            name=node.name,
            type='method' if self.current_class else 'function',
            file=str(self.filepath),
            line=node.lineno,
            has_docstring=bool(docstring),
            docstring=docstring,
            params=params,
            returns=has_return,
            is_public=is_public,
            parent=self.current_class or ""
        )
        self.items.append(item)

        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Visit async function definition."""
        # Treat same as regular function
        self.visit_FunctionDef(node)


def analyze_file(filepath: Path) -> List[DocItem]:
    """Analyze a single Python file for documentation gaps."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        tree = ast.parse(content, filename=str(filepath))
        analyzer = DocAnalyzer(filepath)
        analyzer.visit(tree)
        return analyzer.items
    except Exception as e:
        print(f"Error analyzing {filepath}: {e}", file=sys.stderr)
        return []


def analyze_directory(base_path: Path, pattern: str = "**/*.py") -> Dict[str, List[DocItem]]:
    """Analyze all Python files in directory."""
    results = defaultdict(list)

    for filepath in base_path.glob(pattern):
        if '__pycache__' in str(filepath) or 'test' in str(filepath):
            continue

        items = analyze_file(filepath)
        if items:
            # Group by module
            module_name = str(filepath.relative_to(base_path.parent))
            results[module_name] = items

    return dict(results)


def generate_report(all_items: Dict[str, List[DocItem]], output_path: Path):
    """Generate comprehensive documentation gap report."""

    report = []
    report.append("# RepTate Documentation Gap Analysis")
    report.append(f"\nGenerated: 2025-12-31\n")

    # Overall statistics
    total_items = 0
    documented_items = 0
    public_items = 0
    public_documented = 0

    undocumented_classes = []
    undocumented_functions = []
    incomplete_docs = []

    # Categorize items by module group
    core_items = {}
    gui_items = {}
    theory_items = {}
    tool_items = {}
    other_items = {}

    for module, items in all_items.items():
        if '/core/' in module:
            core_items[module] = items
        elif '/gui/' in module:
            gui_items[module] = items
        elif '/theories/' in module:
            theory_items[module] = items
        elif '/tools/' in module:
            tool_items[module] = items
        else:
            other_items[module] = items

    # Analyze each category
    def analyze_category(category_name: str, category_items: Dict[str, List[DocItem]]):
        nonlocal total_items, documented_items, public_items, public_documented

        if not category_items:
            return

        report.append(f"\n## {category_name}\n")

        cat_total = 0
        cat_documented = 0
        cat_public = 0
        cat_public_documented = 0

        for module, items in sorted(category_items.items()):
            module_undoc = []
            module_incomplete = []

            for item in items:
                cat_total += 1
                total_items += 1

                if item.is_public:
                    cat_public += 1
                    public_items += 1

                if item.has_docstring:
                    cat_documented += 1
                    documented_items += 1

                    if item.is_public:
                        cat_public_documented += 1
                        public_documented += 1

                    # Check completeness
                    if item.type in ('function', 'method'):
                        analyzer = DocAnalyzer(Path(item.file))
                        completeness = analyzer._check_docstring_completeness(
                            item.docstring, item.params, item.returns
                        )

                        issues = []
                        if not completeness['has_params'] and item.params:
                            issues.append(f"missing params: {', '.join(item.params)}")
                        if not completeness['has_returns'] and item.returns:
                            issues.append("missing return documentation")

                        if issues and item.is_public:
                            module_incomplete.append((item, issues))
                            incomplete_docs.append((module, item, issues))
                else:
                    if item.is_public:
                        module_undoc.append(item)
                        if item.type == 'class':
                            undocumented_classes.append((module, item))
                        else:
                            undocumented_functions.append((module, item))

            # Report per module
            if module_undoc or module_incomplete:
                report.append(f"\n### `{module}`\n")

                if module_undoc:
                    report.append("**UNDOCUMENTED:**\n")
                    for item in module_undoc:
                        prefix = f"{item.parent}." if item.parent else ""
                        report.append(f"- `{prefix}{item.name}` ({item.type}, line {item.line})")
                    report.append("")

                if module_incomplete:
                    report.append("**INCOMPLETE:**\n")
                    for item, issues in module_incomplete:
                        prefix = f"{item.parent}." if item.parent else ""
                        report.append(f"- `{prefix}{item.name}` ({item.type}, line {item.line}): {'; '.join(issues)}")
                    report.append("")

        # Category summary
        coverage = (cat_documented / cat_total * 100) if cat_total > 0 else 0
        public_coverage = (cat_public_documented / cat_public * 100) if cat_public > 0 else 0

        report.append(f"\n**{category_name} Summary:**")
        report.append(f"- Total items: {cat_total}")
        report.append(f"- Documented: {cat_documented} ({coverage:.1f}%)")
        report.append(f"- Public items: {cat_public}")
        report.append(f"- Public documented: {cat_public_documented} ({public_coverage:.1f}%)")
        report.append("")

    # Analyze each category
    analyze_category("Core Modules (src/RepTate/core/)", core_items)
    analyze_category("GUI Modules (src/RepTate/gui/)", gui_items)
    analyze_category("Theory Modules (src/RepTate/theories/)", theory_items)
    analyze_category("Tool Modules (src/RepTate/tools/)", tool_items)

    # Overall summary
    report.append("\n## Overall Summary\n")
    overall_coverage = (documented_items / total_items * 100) if total_items > 0 else 0
    public_coverage = (public_documented / public_items * 100) if public_items > 0 else 0

    report.append(f"- **Total items analyzed:** {total_items}")
    report.append(f"- **Documented items:** {documented_items} ({overall_coverage:.1f}%)")
    report.append(f"- **Public items:** {public_items}")
    report.append(f"- **Public documented:** {public_documented} ({public_coverage:.1f}%)")
    report.append(f"- **Undocumented public classes:** {len(undocumented_classes)}")
    report.append(f"- **Undocumented public functions/methods:** {len(undocumented_functions)}")
    report.append(f"- **Incomplete documentation:** {len(incomplete_docs)}")

    # Priority recommendations
    report.append("\n## Priority Recommendations\n")
    report.append("### High Priority (New Modernization Infrastructure)\n")

    priority_modules = [
        'src/RepTate/core/safe_eval.py',
        'src/RepTate/core/serialization.py',
        'src/RepTate/core/feature_flags.py',
        'src/RepTate/core/interfaces.py',
        'src/RepTate/core/native_loader.py',
        'src/RepTate/gui/DatasetManager.py',
        'src/RepTate/gui/FileIOController.py',
        'src/RepTate/gui/MenuManager.py',
        'src/RepTate/gui/ParameterController.py',
        'src/RepTate/gui/TheoryCompute.py',
        'src/RepTate/gui/ViewCoordinator.py',
    ]

    for module in priority_modules:
        found = False
        for undoc_module, item in undocumented_classes + undocumented_functions:
            if module in undoc_module:
                if not found:
                    report.append(f"\n**`{module}`:**")
                    found = True
                prefix = f"{item.parent}." if item.parent else ""
                report.append(f"- Add docstring to `{prefix}{item.name}` ({item.type}, line {item.line})")

        for undoc_module, item, issues in incomplete_docs:
            if module in undoc_module:
                if not found:
                    report.append(f"\n**`{module}`:**")
                    found = True
                prefix = f"{item.parent}." if item.parent else ""
                report.append(f"- Complete docstring for `{prefix}{item.name}` ({item.type}, line {item.line}): {'; '.join(issues)}")

    # Write report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"Report generated: {output_path}")
    print(f"\nSummary:")
    print(f"  Total items: {total_items}")
    print(f"  Overall coverage: {overall_coverage:.1f}%")
    print(f"  Public API coverage: {public_coverage:.1f}%")
    print(f"  Undocumented public items: {len(undocumented_classes) + len(undocumented_functions)}")
    print(f"  Incomplete documentation: {len(incomplete_docs)}")


if __name__ == '__main__':
    base_path = Path('/home/wei/Documents/GitHub/RepTate/src/RepTate')
    output_path = Path('/home/wei/Documents/GitHub/RepTate/DOCUMENTATION_GAP_REPORT.md')

    print("Analyzing RepTate codebase for documentation gaps...")
    all_items = analyze_directory(base_path)

    print(f"Found {sum(len(items) for items in all_items.values())} documentable items")
    print(f"Across {len(all_items)} modules")

    generate_report(all_items, output_path)
