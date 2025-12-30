# Contributing

## Development Environment

- Python 3.12+
- Install project dependencies using the repository tooling.

## Project Layout

- Source code: `src/RepTate/`
- Tests: `tests/`
- Documentation: `docs/`

## Guidelines

- Keep computation in JAX and avoid NumPy/SciPy in core code.
- Maintain full type annotations for public APIs.
- Use explicit imports; avoid wildcard imports.
- Update documentation when changing public behavior.

## Validation

- Run the test suite before submitting changes.
- Run validation scripts for numerical routines when touching core computation.
- Check `docs/logging.md` if you need to adjust log locations or levels.
