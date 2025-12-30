---

description: "Task list for JAX-First Curve Fitting Modernization"
---

# Tasks: JAX-First Curve Fitting Modernization

**Input**: Design documents from `/specs/001-jax-nlsq-modernization/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/

**Tests**: Not requested in the feature specification; no test tasks included.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Constitution Guardrails *(mandatory)*

- Include tasks that keep all numerical computation in JAX (no NumPy/SciPy compute).
- Include tasks that verify full-precision correctness (no approximation shortcuts).
- Include tasks for Python 3.12+ typing and explicit-import hygiene where APIs change.
- Include tasks for NLSQ-based curve fitting and removal of scipy.optimize usage.
- Include tasks that keep NumPyro NUTS integration viable when inference is involved.
- Include tasks that preserve a PyQt UI boundary separate from computation.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- Single desktop application under `RepTate/` with tests in `tests/`

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

**Cross-cutting note**: Tasks in this phase establish shared scaffolding and constitution-mandated structure, so they are intentionally not tied to a single user story.

- [X] T001 Create core package stubs in `RepTate/core/data/__init__.py`, `RepTate/core/models/__init__.py`, `RepTate/core/fitting/__init__.py`, `RepTate/core/inference/__init__.py`, `RepTate/core/compute/__init__.py` (supports FR-001 to FR-006 scaffolding)
- [X] T002 Create GUI package stubs in `RepTate/gui/views/__init__.py`, `RepTate/gui/controllers/__init__.py`, `RepTate/gui/exports/__init__.py` (supports FR-007, FR-010 scaffolding)
- [X] T003 Add module boundary docs in `RepTate/core/__init__.py` and `RepTate/gui/__init__.py` (supports constitution: Modern Python Contracts)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T004 Implement dataset I/O and validation in `RepTate/core/data/dataset_io.py`
- [X] T005 [P] Implement model registry for existing RepTate models in `RepTate/core/models/model_registry.py`
- [X] T006 [P] Inventory and replace scipy.optimize usage with NLSQ in `RepTate/` (identify targets, update code paths)
- [X] T007 [P] Implement result records (Dataset, Model, FitResult, PosteriorResult) in `RepTate/core/models/results.py`
- [X] T008 Implement result persistence for fits/posteriors/traces in `RepTate/core/data/result_store.py`
- [X] T009 Implement hardware detection + CPU fallback warning in `RepTate/core/compute/hardware.py`
- [X] T010 Implement computation service boundary for UI in `RepTate/core/compute/service_api.py`
- [X] T011 Implement deterministic-fit correctness validation script in `tests/regression/validate_fit_precision.py`
- [X] T012 Implement inference consistency validation script in `tests/regression/validate_inference_precision.py`

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Deterministic Curve Fit (Priority: P1) 🎯 MVP

**Goal**: Deterministic fit of datasets using existing models with diagnostics and residuals.

**Independent Test**: Load a reference dataset, run a deterministic fit, and verify parameters, residuals, and diagnostics are produced.

### Implementation for User Story 1

- [X] T013 [P] [US1] Implement dataset import mapping in `RepTate/core/data/importer.py`
- [X] T014 [P] [US1] Implement NLSQ deterministic fit runner in `RepTate/core/fitting/nlsq_fit.py`
- [X] T015 [US1] Implement fit diagnostics assembly in `RepTate/core/fitting/fit_results.py`
- [X] T016 [US1] Add deterministic fit controller in `RepTate/gui/controllers/fit_controller.py`
- [X] T017 [US1] Add fit configuration/results view in `RepTate/gui/views/fit_view.py`

**Checkpoint**: User Story 1 is fully functional and testable independently

---

## Phase 4: User Story 2 - Bayesian Inference with Warm-Start (Priority: P2)

**Goal**: Bayesian inference from deterministic fits with posterior summaries and resumable runs.

**Independent Test**: Run deterministic fit, start inference, confirm posterior samples and credible intervals are produced, interrupt and resume.

### Implementation for User Story 2

- [X] T018 [P] [US2] Implement warm-start preparation from FitResult in `RepTate/core/inference/warm_start.py`
- [X] T019 [P] [US2] Implement NumPyro NUTS runner in `RepTate/core/inference/nuts_runner.py`
- [X] T020 [US2] Implement resume state storage in `RepTate/core/inference/resume_store.py`
- [X] T021 [US2] Add inference controller in `RepTate/gui/controllers/inference_controller.py`
- [X] T022 [US2] Add inference results view in `RepTate/gui/views/inference_view.py`

**Checkpoint**: User Stories 1 and 2 are independently functional

---

## Phase 5: User Story 3 - Visual Analysis and Reporting (Priority: P3)

**Goal**: Visualize data, fits, residuals, posteriors, and export plots/results/traces.

**Independent Test**: Load dataset, run fit and inference, open views, and export plots + numeric results + raw traces.

### Implementation for User Story 3

- [X] T023 [P] [US3] Implement visualization state model in `RepTate/core/models/visualization_state.py`
- [X] T024 [P] [US3] Implement plot rendering views in `RepTate/gui/views/plot_views.py`
- [X] T025 [US3] Implement export pipeline for plots/results/traces in `RepTate/gui/exports/export_service.py`
- [X] T026 [US3] Add export controller wiring in `RepTate/gui/controllers/export_controller.py`
- [X] T027 [US3] Add summary view for residuals/posteriors in `RepTate/gui/views/summary_view.py`

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

**Cross-cutting note**: Tasks in this phase satisfy constitution and quality gates across all user stories, so they are intentionally not tied to a single user story.

- [X] T028 [P] Update contributor-facing docs in `README.rst` (supports constitution: Modern Python Contracts)
- [X] T029 [P] Validate quickstart steps in `specs/001-jax-nlsq-modernization/quickstart.md` (supports SC-001/SC-004 validation readiness)
- [X] T030 Update API boundary references in `specs/001-jax-nlsq-modernization/contracts/api-spec.json` (supports FR-001 to FR-010 traceability)
- [X] T031 [P] Add public API type annotations and docstrings in `RepTate/core/compute/service_api.py`, `RepTate/core/fitting/nlsq_fit.py`, `RepTate/core/inference/nuts_runner.py` (supports constitution: Modern Python Contracts)
- [X] T032 [P] Add type-check configuration update in `pyproject.toml` or `setup.cfg` (supports constitution: Modern Python Contracts)
- [X] T033 Implement deterministic-fit timing benchmark in `tests/regression/benchmark_fit_timing.py` (validates SC-001/SC-002)
- [X] T034 Implement export timing benchmark in `tests/regression/benchmark_export_timing.py` (validates SC-004)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - Integrates with US1 outputs but testable on its own
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - Can operate once fit/inference artifacts exist

### Within Each User Story

- Models before services
- Services before controllers/views
- Core implementation before UI wiring
- Story complete before moving to next priority

### Parallel Opportunities

- Phase 1: T001 and T002 can run in parallel
- Phase 2: T005, T006, and T007 can run in parallel with T004
- Phase 3: T013 and T014 can run in parallel
- Phase 4: T018 and T019 can run in parallel
- Phase 5: T023 and T024 can run in parallel
- Phase 6: T028 and T029 can run in parallel

---

## Parallel Example: User Story 1

```bash
Task: "Implement dataset import mapping in RepTate/core/data/importer.py"
Task: "Implement NLSQ deterministic fit runner in RepTate/core/fitting/nlsq_fit.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Fit a reference dataset and verify diagnostics

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Validate independently → MVP
3. Add User Story 2 → Validate independently
4. Add User Story 3 → Validate independently

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1
   - Developer B: User Story 2
   - Developer C: User Story 3
3. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence
