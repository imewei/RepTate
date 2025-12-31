"""Characterization tests for RepTate god classes.

Characterization tests capture the current behavior of large, complex classes
before refactoring. They serve as a safety net during decomposition work.

Target classes:
    - QApplicationWindow: ~2000 LOC main application window
    - QTheory: ~1500 LOC theory UI component

These tests should be written BEFORE any refactoring begins to ensure
behavioral equivalence after changes.
"""
