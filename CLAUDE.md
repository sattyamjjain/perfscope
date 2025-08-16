# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PerfScope is a Python performance profiler and APM (Application Performance Monitoring) tool that provides real-time insights into function execution time, memory consumption, call hierarchies, and bottleneck detection. The package is designed for production use with minimal overhead and zero runtime dependencies.

## Development Commands

### Installation
```bash
# Install in development mode with all dependencies
make install-dev
# Or directly with pip
pip install -e ".[dev]"
```

### Testing
```bash
# Run all tests
make test
# Or with pytest directly
pytest

# Run tests with coverage report
make test-cov
# Or with pytest
pytest --cov=perfscope --cov-report=term-missing --cov-report=html

# Run a single test file
pytest tests/test_profiler.py -v

# Run a specific test
pytest tests/test_profiler.py::test_basic_function_profiling -v
```

### Code Quality
```bash
# Format code
make format
# Or with ruff directly
ruff format src tests

# Lint code
make lint
# Or with ruff directly
ruff check src tests

# Type checking
make typecheck
# Or with mypy directly
mypy src/perfscope

# Security scanning
bandit -r src/ -f json -o bandit-report.json
safety check

# Pre-commit hooks (installed with make install-dev)
pre-commit run --all-files
```

### Building & Distribution
```bash
# Build distribution packages
make build
# Or with python directly
python -m build

# Upload to PyPI (requires credentials)
make upload
# Or with twine
python -m twine upload dist/*
```

### Cleanup
```bash
# Clean all build artifacts and caches
make clean
```

## Code Architecture

### Package Structure
- **src/perfscope/** - Main package directory
  - `__init__.py` - Package initialization, exports main API (`profile`, `Profiler`, `ProfileConfig`)
  - `config.py` - Configuration dataclass (`ProfileConfig`) for controlling profiling behavior
  - `profiler.py` - Core profiler implementation with `Profiler` class, `profile` decorator, and performance tracking logic

### Key Components

1. **ProfileConfig** (`config.py`)
   - Dataclass that controls all profiling behavior
   - Configures tracing depth, memory tracking, logging levels, filtering
   - Validates configuration parameters on initialization

2. **Profiler** (`profiler.py`)
   - Main profiler class that handles function tracing
   - Manages call stack, performance metrics collection
   - Generates reports in HTML/JSON formats
   - Thread-safe implementation for concurrent applications

3. **profile decorator** (`profiler.py`)
   - Main user-facing API as a function decorator
   - Supports both sync and async functions
   - Configurable via kwargs that map to `ProfileConfig`
   - Attaches `profile_report` attribute to decorated functions

4. **PerformanceMetrics** (`profiler.py`)
   - Dataclass storing detailed metrics for each function call
   - Tracks wall time, CPU time, memory usage, GC collections
   - Calculates derived metrics like CPU efficiency

### Important Design Patterns

- **Context Manager Pattern**: Profiler implements `__enter__`/`__exit__` for manual profiling blocks
- **Decorator Pattern**: `profile()` decorator for automatic function profiling
- **Singleton-like Behavior**: Each decorated function gets its own profiler instance
- **Thread-Local Storage**: Used for managing nested call tracking per thread

### Dependencies

- **Core**: Zero runtime dependencies (pure Python)
- **Optional**: `psutil` for enhanced memory tracking (installed with `[full]` extra)
- **Development**: pytest, ruff, mypy, pre-commit, build, twine (installed with `[dev]` extra)
- **Web/Science**: Optional extras available for `[web]`, `[science]`, `[all]`

## Testing Approach

Tests are in `tests/test_profiler.py` and cover:
- Basic function profiling (sync and async)
- Memory tracking and leak detection
- Nested function calls and recursion
- Multi-threading and concurrency
- Exception handling
- Various decorator scenarios
- Report generation

Run tests with visible output using `pytest -v -s` to see profiling logs during test execution.

## Important Notes

- The profiler uses Python's `sys.settrace()` for call tracking, which has performance implications
- Memory tracking requires `tracemalloc` to be started, handled automatically when `trace_memory=True`
- HTML reports are generated using inline CSS/JavaScript for portability
- Logging uses a dedicated "perfscope" logger with `[PerfScope]` prefix for easy filtering
- The package is typed and includes `py.typed` marker for mypy support
- **CRITICAL**: The profiler always excludes `perfscope` and `logging` modules to prevent infinite recursion
- Resource cleanup is handled properly in `stop()` method and decorator finally blocks
- Thread-local storage is used for managing per-thread call stacks
- CI/CD pipeline runs tests on Python 3.8-3.12 across Linux, Windows, and macOS
- All code must pass ruff linting, mypy type checking, and security scans before merge
- **Professional Standards**: No emojis in code, logs, or reports - clean production-ready output only
