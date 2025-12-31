#!/usr/bin/env python3
"""Focused documentation gap analysis for new modernization modules."""

import ast
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class DocGap:
    """Documentation gap item."""
    name: str
    type: str  # 'class', 'function', 'method'
    line: int
    issue: str  # 'missing_docstring', 'missing_params', 'missing_returns'
    details: str = ""
    parent: str = ""


def analyze_file_focused(filepath: Path) -> List[DocGap]:
    """Analyze a file and return documentation gaps."""
    gaps = []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        tree = ast.parse(content, filename=str(filepath))
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return gaps

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.current_class = None

        def visit_ClassDef(self, node):
            docstring = ast.get_docstring(node)
            if not docstring:
                gaps.append(DocGap(
                    name=node.name,
                    type='class',
                    line=node.lineno,
                    issue='missing_docstring',
                    details='Class needs docstring explaining purpose and usage'
                ))

            old_class = self.current_class
            self.current_class = node.name
            self.generic_visit(node)
            self.current_class = old_class

        def visit_FunctionDef(self, node):
            # Skip private methods
            if node.name.startswith('_') and node.name != '__init__':
                self.generic_visit(node)
                return

            docstring = ast.get_docstring(node) or ""
            params = [arg.arg for arg in node.args.args if arg.arg not in ('self', 'cls')]
            has_return = any(isinstance(n, ast.Return) and n.value for n in ast.walk(node))

            item_type = 'method' if self.current_class else 'function'

            if not docstring:
                gaps.append(DocGap(
                    name=node.name,
                    type=item_type,
                    line=node.lineno,
                    issue='missing_docstring',
                    details=f"{'Method' if self.current_class else 'Function'} needs docstring",
                    parent=self.current_class or ""
                ))
            else:
                # Check for parameter documentation
                doc_lower = docstring.lower()
                if params:
                    has_params_doc = any(ind in doc_lower for ind in ['parameters:', 'params:', 'args:', 'arguments:'])
                    if not has_params_doc:
                        gaps.append(DocGap(
                            name=node.name,
                            type=item_type,
                            line=node.lineno,
                            issue='missing_params',
                            details=f"Parameters not documented: {', '.join(params)}",
                            parent=self.current_class or ""
                        ))

                # Check for return documentation
                if has_return:
                    has_return_doc = any(ind in doc_lower for ind in ['returns:', 'return:', 'yields:', 'yield:'])
                    if not has_return_doc:
                        gaps.append(DocGap(
                            name=node.name,
                            type=item_type,
                            line=node.lineno,
                            issue='missing_returns',
                            details="Return value not documented",
                            parent=self.current_class or ""
                        ))

            self.generic_visit(node)

    visitor = Visitor()
    visitor.visit(tree)
    return gaps


def generate_focused_report():
    """Generate focused report on new modernization modules."""

    base_path = Path('/home/wei/Documents/GitHub/RepTate/src/RepTate')

    # Priority modules to analyze
    modules = {
        'Core Infrastructure': [
            'core/safe_eval.py',
            'core/serialization.py',
            'core/feature_flags.py',
            'core/interfaces.py',
            'core/native_loader.py',
            'core/path_utils.py',
            'core/temp_utils.py',
        ],
        'GUI Controllers': [
            'gui/DatasetManager.py',
            'gui/FileIOController.py',
            'gui/MenuManager.py',
            'gui/ParameterController.py',
            'gui/TheoryCompute.py',
            'gui/ViewCoordinator.py',
        ],
    }

    report = []
    report.append("# RepTate Documentation Gap Report - New Modernization Modules\n")
    report.append("**Generated:** 2025-12-31\n")
    report.append("**Focus:** New infrastructure added in 003-reptate-modernization\n")

    total_gaps = 0
    total_files = 0
    files_with_gaps = 0

    for category, module_list in modules.items():
        report.append(f"\n## {category}\n")

        category_gaps = 0

        for module_rel_path in module_list:
            module_path = base_path / module_rel_path

            if not module_path.exists():
                report.append(f"### {module_rel_path}\n")
                report.append("**Status:** File does not exist (may need to be created)\n")
                continue

            gaps = analyze_file_focused(module_path)
            total_files += 1

            if gaps:
                files_with_gaps += 1
                category_gaps += len(gaps)
                total_gaps += len(gaps)

                report.append(f"### `{module_rel_path}`\n")
                report.append(f"**Documentation Gaps:** {len(gaps)}\n")

                # Group by issue type
                missing_docstrings = [g for g in gaps if g.issue == 'missing_docstring']
                missing_params = [g for g in gaps if g.issue == 'missing_params']
                missing_returns = [g for g in gaps if g.issue == 'missing_returns']

                if missing_docstrings:
                    report.append("\n**Missing Docstrings:**\n")
                    for gap in missing_docstrings:
                        prefix = f"{gap.parent}." if gap.parent else ""
                        report.append(f"- Line {gap.line}: `{prefix}{gap.name}` ({gap.type})")
                        report.append(f"  - {gap.details}")

                if missing_params:
                    report.append("\n**Missing Parameter Documentation:**\n")
                    for gap in missing_params:
                        prefix = f"{gap.parent}." if gap.parent else ""
                        report.append(f"- Line {gap.line}: `{prefix}{gap.name}` ({gap.type})")
                        report.append(f"  - {gap.details}")

                if missing_returns:
                    report.append("\n**Missing Return Documentation:**\n")
                    for gap in missing_returns:
                        prefix = f"{gap.parent}." if gap.parent else ""
                        report.append(f"- Line {gap.line}: `{prefix}{gap.name}` ({gap.type})")
                        report.append(f"  - {gap.details}")

                report.append("")
            else:
                report.append(f"### `{module_rel_path}`\n")
                report.append("**Status:** Fully documented ✓\n")

        report.append(f"\n**{category} Summary:** {category_gaps} documentation gaps\n")

    # Overall summary
    report.append("\n## Overall Summary\n")
    report.append(f"- **Total files analyzed:** {total_files}")
    report.append(f"- **Files with documentation gaps:** {files_with_gaps}")
    report.append(f"- **Total documentation gaps:** {total_gaps}")

    if total_files > 0:
        coverage_pct = ((total_files - files_with_gaps) / total_files) * 100
        report.append(f"- **Files fully documented:** {total_files - files_with_gaps} ({coverage_pct:.1f}%)")

    # Recommendations
    report.append("\n## Recommended Actions\n")
    report.append("\n### Immediate Priority\n")
    report.append("1. **Add class docstrings** - All public classes should have docstrings explaining:")
    report.append("   - Purpose and responsibility")
    report.append("   - Key attributes")
    report.append("   - Usage examples (for complex classes)")
    report.append("\n2. **Document public methods** - All public methods should have:")
    report.append("   - Brief description")
    report.append("   - Parameters section (with types)")
    report.append("   - Returns section (with type)")
    report.append("   - Raises section (if applicable)")
    report.append("\n### Documentation Style Guide\n")
    report.append("```python")
    report.append('"""Brief one-line summary.')
    report.append("")
    report.append("Detailed description if needed.")
    report.append("")
    report.append("Parameters")
    report.append("----------")
    report.append("param_name : type")
    report.append("    Description")
    report.append("")
    report.append("Returns")
    report.append("-------")
    report.append("return_type")
    report.append("    Description")
    report.append("")
    report.append("Raises")
    report.append("------")
    report.append("ExceptionType")
    report.append("    When and why")
    report.append('"""')
    report.append("```")
    report.append("\n### Tool Support\n")
    report.append("- Use `pydocstyle` to check docstring conventions")
    report.append("- Use `sphinx` to generate documentation from docstrings")
    report.append("- Consider using `interrogate` to measure documentation coverage")

    # Write report
    output_path = Path('/home/wei/Documents/GitHub/RepTate/MODERNIZATION_DOC_GAPS.md')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"Focused report generated: {output_path}")
    print(f"\nNew Modernization Modules Summary:")
    print(f"  Files analyzed: {total_files}")
    print(f"  Files with gaps: {files_with_gaps}")
    print(f"  Total gaps: {total_gaps}")
    if total_files > 0:
        print(f"  Documentation coverage: {((total_files - files_with_gaps) / total_files) * 100:.1f}%")


if __name__ == '__main__':
    generate_focused_report()
