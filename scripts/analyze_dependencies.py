#!/usr/bin/env python3
"""Analyze module dependencies and create dependency graph.

This script analyzes RepTate's internal dependencies to identify:
- Module coupling (import counts)
- Circular dependencies
- Integration points
- Tight coupling that needs facades/adapters
"""

import ast
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, List, Tuple


class DependencyAnalyzer:
    """Analyze Python module dependencies."""

    def __init__(self, root_path: Path):
        self.root = root_path
        self.dependencies: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_deps: Dict[str, Set[str]] = defaultdict(set)
        self.external_deps: Dict[str, Set[str]] = defaultdict(set)

    def get_module_name(self, filepath: Path) -> str:
        """Convert file path to module name."""
        try:
            rel_path = filepath.relative_to(self.root)
            parts = list(rel_path.parts)
            if parts[-1] == "__init__.py":
                parts = parts[:-1]
            else:
                parts[-1] = parts[-1].replace(".py", "")
            return ".".join(parts)
        except ValueError:
            return str(filepath)

    def extract_imports(self, filepath: Path) -> Tuple[Set[str], Set[str]]:
        """Extract internal and external imports from a Python file.

        Returns:
            (internal_imports, external_imports)
        """
        internal = set()
        external = set()

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("RepTate."):
                            internal.add(alias.name)
                        else:
                            # Extract top-level package
                            pkg = alias.name.split(".")[0]
                            external.add(pkg)

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        if node.module.startswith("RepTate."):
                            internal.add(node.module)
                        elif not node.module.startswith("."):
                            # Extract top-level package
                            pkg = node.module.split(".")[0]
                            external.add(pkg)

        except SyntaxError as e:
            print(f"Syntax error in {filepath}: {e}")
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")

        return internal, external

    def analyze_directory(self, pattern: str = "**/*.py") -> None:
        """Analyze all Python files in directory."""
        files = list(self.root.glob(pattern))
        print(f"Analyzing {len(files)} Python files...")

        for filepath in files:
            # Skip auto-generated resource files
            if filepath.name.endswith("_rc.py") or filepath.name.endswith("_ui.py"):
                continue

            module = self.get_module_name(filepath)
            internal, external = self.extract_imports(filepath)

            self.dependencies[module] = internal
            self.external_deps[module] = external

            # Build reverse dependency graph
            for dep in internal:
                self.reverse_deps[dep].add(module)

    def find_circular_dependencies(self) -> List[List[str]]:
        """Find circular dependency chains."""
        cycles = []
        visited = set()
        rec_stack = set()

        def dfs(node: str, path: List[str]) -> None:
            if node in rec_stack:
                # Found cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                if cycle not in cycles:
                    cycles.append(cycle)
                return

            if node in visited:
                return

            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for dep in self.dependencies.get(node, []):
                dfs(dep, path[:])

            rec_stack.remove(node)

        for module in self.dependencies.keys():
            if module not in visited:
                dfs(module, [])

        return cycles

    def get_coupling_metrics(self) -> List[Tuple[str, int, int]]:
        """Get modules sorted by coupling (efferent + afferent).

        Returns:
            List of (module, efferent_count, afferent_count) tuples
        """
        metrics = []
        for module in self.dependencies.keys():
            efferent = len(self.dependencies[module])  # Dependencies on others
            afferent = len(self.reverse_deps[module])   # Modules depending on this
            metrics.append((module, efferent, afferent))

        return sorted(metrics, key=lambda x: x[1] + x[2], reverse=True)

    def identify_god_modules(self, threshold: int = 15) -> List[str]:
        """Identify highly coupled modules (god modules)."""
        god_modules = []
        for module, efferent, afferent in self.get_coupling_metrics():
            if efferent >= threshold:
                god_modules.append(module)
        return god_modules

    def get_layer_violations(self) -> List[Tuple[str, str]]:
        """Identify architectural layer violations.

        Expected: GUI → Applications → Theories → Core
        Violations: Core → GUI, Theories → Applications, etc.
        """
        violations = []

        for module, deps in self.dependencies.items():
            # Core should not depend on gui, applications, theories, tools
            if ".core." in module:
                for dep in deps:
                    if any(x in dep for x in [".gui.", ".applications.", ".theories.", ".tools."]):
                        violations.append((module, dep))

            # Theories should not depend on applications
            elif ".theories." in module:
                for dep in deps:
                    if ".applications." in dep:
                        violations.append((module, dep))

        return violations

    def analyze_integration_points(self) -> Dict[str, List[str]]:
        """Identify key integration points requiring facades/adapters."""
        integration_points = {
            "serialization": [],
            "scipy_legacy": [],
            "native_libs": [],
            "gui_theory": [],
        }

        for module, deps in self.dependencies.items():
            # Modules still using SciPy
            if any("scipy" in ext for ext in self.external_deps.get(module, [])):
                integration_points["scipy_legacy"].append(module)

            # Modules using ctypes (native libraries)
            if "ctypes" in self.external_deps.get(module, []):
                integration_points["native_libs"].append(module)

            # Direct GUI → Theory coupling (should use interfaces)
            if ".gui." in module:
                theory_deps = [d for d in deps if ".theories." in d]
                if theory_deps:
                    integration_points["gui_theory"].append((module, theory_deps))

        return integration_points


def main():
    """Run dependency analysis."""
    import sys

    # Find RepTate root
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    reptate_root = repo_root / "src" / "RepTate"

    if not reptate_root.exists():
        print(f"Error: RepTate source not found at {reptate_root}")
        sys.exit(1)

    analyzer = DependencyAnalyzer(reptate_root)
    analyzer.analyze_directory()

    print("\n" + "=" * 80)
    print("DEPENDENCY ANALYSIS REPORT")
    print("=" * 80)

    # 1. Coupling metrics
    print("\n### TOP 20 MOST COUPLED MODULES ###")
    print(f"{'Module':<50} {'Out':>5} {'In':>5} {'Total':>5}")
    print("-" * 70)
    for module, efferent, afferent in analyzer.get_coupling_metrics()[:20]:
        total = efferent + afferent
        print(f"{module:<50} {efferent:>5} {afferent:>5} {total:>5}")

    # 2. God modules
    print("\n### GOD MODULES (>= 15 DEPENDENCIES) ###")
    god_modules = analyzer.identify_god_modules(threshold=15)
    for module in god_modules:
        efferent = len(analyzer.dependencies[module])
        print(f"  - {module}: {efferent} dependencies")

    # 3. Circular dependencies
    print("\n### CIRCULAR DEPENDENCIES ###")
    cycles = analyzer.find_circular_dependencies()
    if cycles:
        for i, cycle in enumerate(cycles[:10], 1):  # Show first 10
            print(f"\nCycle {i}:")
            for module in cycle:
                print(f"  → {module}")
    else:
        print("  ✓ No circular dependencies detected")

    # 4. Layer violations
    print("\n### ARCHITECTURAL LAYER VIOLATIONS ###")
    violations = analyzer.get_layer_violations()
    if violations:
        for src, dst in violations[:20]:  # Show first 20
            print(f"  {src} → {dst}")
    else:
        print("  ✓ No layer violations detected")

    # 5. Integration points
    print("\n### INTEGRATION POINTS REQUIRING ADAPTERS ###")
    integration = analyzer.analyze_integration_points()

    print(f"\nSciPy Legacy ({len(integration['scipy_legacy'])} modules):")
    for module in integration['scipy_legacy']:
        print(f"  - {module}")

    print(f"\nNative Libraries ({len(integration['native_libs'])} modules):")
    for module in integration['native_libs'][:10]:  # Show first 10
        print(f"  - {module}")

    print(f"\nDirect GUI → Theory Coupling ({len(integration['gui_theory'])} modules):")
    for module, deps in integration['gui_theory'][:10]:  # Show first 10
        print(f"  - {module} → {', '.join(deps)}")

    # 6. External dependencies summary
    print("\n### TOP 15 EXTERNAL DEPENDENCIES ###")
    ext_dep_count = defaultdict(int)
    for module, deps in analyzer.external_deps.items():
        for dep in deps:
            ext_dep_count[dep] += 1

    sorted_ext = sorted(ext_dep_count.items(), key=lambda x: x[1], reverse=True)
    for dep, count in sorted_ext[:15]:
        print(f"  {dep:<20} {count:>3} modules")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
