# Contributing to PyCallMeter

🎉 First off, thank you for considering contributing to PyCallMeter! It's people like you who make PyCallMeter such a great tool for the Python community.

## 🚀 Quick Start

### Development Environment Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/yourusername/pycallmeter.git
   cd pycallmeter
   ```

2. **Set up development environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install development dependencies
   pip install -e ".[dev]"
   ```

3. **Verify installation**
   ```bash
   # Run tests
   pytest tests/ -v
   
   # Run type checking
   mypy src/
   
   # Run linting
   ruff check src/ tests/
   
   # Check formatting
   black --check src/ tests/
   ```

## 🎯 Ways to Contribute

### 🐛 Bug Reports

Found a bug? Please help us fix it!

1. **Check existing issues** first to avoid duplicates
2. **Use our bug report template** when creating new issues
3. **Include minimal reproduction code** that demonstrates the issue
4. **Provide system information** (Python version, OS, PyCallMeter version)
5. **Include relevant logs** with the issue

### ✨ Feature Requests

Have an idea for a new feature?

1. **Check existing feature requests** to see if it's already been suggested
2. **Use our feature request template** to provide details
3. **Explain the use case** and why it would be valuable
4. **Consider implementation complexity** and breaking changes

### 📝 Documentation Improvements

Documentation is crucial for user adoption:

- **Fix typos** and improve clarity
- **Add examples** for complex use cases
- **Improve API documentation** with better docstrings
- **Create tutorials** for specific frameworks or scenarios
- **Update README** with new features or improvements

### 💻 Code Contributions

Ready to write some code? Here's how:

## 🛠️ Development Guidelines

### Code Style

We use automated tools to maintain consistent code style:

- **Black** for code formatting
- **Ruff** for linting and import sorting  
- **mypy** for static type checking

```bash
# Format code
black src/ tests/

# Sort imports
ruff check --fix src/ tests/

# Type checking
mypy src/
```

### Testing

We maintain high test coverage (>95%) and use comprehensive testing:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/pycallmeter --cov-report=html

# Run specific test categories
pytest tests/ -k "test_profiler"
pytest tests/ -k "test_memory"
pytest tests/ -k "test_async"

# Run performance benchmarks
pytest tests/ -k benchmark --benchmark-only
```

### Type Hints

All code must include comprehensive type hints:

```python
from typing import Optional, Dict, Any, Callable

def example_function(
    name: str, 
    config: Optional[Dict[str, Any]] = None
) -> Callable[..., Any]:
    """Example function with proper type hints."""
    pass
```

### Documentation

All public APIs must have comprehensive docstrings:

```python
def profile_function(
    func: Callable[..., Any],
    *,
    trace_memory: bool = True,
    max_depth: int = 100
) -> Callable[..., Any]:
    """Profile a function with comprehensive performance monitoring.
    
    Args:
        func: The function to profile
        trace_memory: Whether to track memory usage
        max_depth: Maximum call stack depth to trace
        
    Returns:
        The profiled function with performance tracking
        
    Example:
        >>> @profile_function(trace_memory=True)
        >>> def my_function():
        ...     return "Hello, World!"
    """
    pass
```

## 🏗️ Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/amazing-feature
# or
git checkout -b fix/bug-description
```

### 2. Make Your Changes

- Write your code following our style guidelines
- Add comprehensive tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 3. Test Thoroughly

```bash
# Run the full test suite
pytest tests/ -v

# Test across Python versions (if available)
tox

# Run type checking
mypy src/

# Check code style
black --check src/ tests/
ruff check src/ tests/
```

### 4. Update Documentation

- Add docstrings to new functions/classes
- Update README if needed
- Add examples for new features
- Update CHANGELOG.md

### 5. Commit Your Changes

Use conventional commit messages:

```bash
git add .
git commit -m "feat: add memory leak detection capabilities"
# or
git commit -m "fix: resolve duplicate logging in async functions"
# or  
git commit -m "docs: improve FastAPI integration examples"
```

Commit types:
- `feat`: New features
- `fix`: Bug fixes
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test improvements
- `chore`: Build/CI changes

### 6. Push and Create Pull Request

```bash
git push origin feature/amazing-feature
```

Then create a pull request using our template.

## 📋 Pull Request Guidelines

### Before Submitting

- [ ] All tests pass (`pytest tests/ -v`)
- [ ] Code is properly formatted (`black src/ tests/`)
- [ ] No linting errors (`ruff check src/ tests/`)
- [ ] Type checking passes (`mypy src/`)
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated (for significant changes)

### PR Description

Use our pull request template and include:

- Clear description of changes
- Motivation and context
- List of changes made
- Testing performed
- Screenshots (if applicable)
- Breaking changes (if any)

### Review Process

1. **Automated checks** must pass (CI/CD pipeline)
2. **Code review** by maintainers
3. **Feedback incorporation** if requested
4. **Final approval** and merge

## 🎯 Contribution Areas

### High Priority

- **Performance optimizations** - Reduce profiling overhead
- **Framework integrations** - Django, Flask, FastAPI middleware
- **Visualization improvements** - Better HTML reports and charts
- **Memory analysis** - Enhanced memory leak detection
- **Documentation** - Examples, tutorials, best practices

### Medium Priority

- **Export formats** - Prometheus, Grafana, DataDog integration
- **CLI improvements** - Command-line interface enhancements
- **Configuration** - More flexible configuration options
- **Testing** - Additional test cases and edge cases

### Welcome Contributions

- **Bug fixes** - Any size, all are welcome
- **Documentation improvements** - Typos, clarity, examples
- **Test improvements** - Better coverage, edge cases
- **Examples** - Real-world usage examples
- **Performance benchmarks** - Help us stay fast

## 🔍 Code Review Process

### What We Look For

1. **Correctness** - Does the code work as intended?
2. **Performance** - Does it maintain low overhead?
3. **Security** - Are there any security implications?
4. **Maintainability** - Is the code clear and well-structured?
5. **Testing** - Is there adequate test coverage?
6. **Documentation** - Are changes properly documented?

### Review Timeline

- **Initial response**: Within 2-3 days
- **Full review**: Within 1 week
- **Follow-up**: Within 2-3 days of updates

## 🚀 Release Process

### Version Numbers

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR** version for incompatible API changes
- **MINOR** version for backwards-compatible functionality
- **PATCH** version for backwards-compatible bug fixes

### Release Schedule

- **Major releases**: Every 6-12 months
- **Minor releases**: Every 2-3 months  
- **Patch releases**: As needed for critical fixes

## 💬 Community

### Getting Help

- **GitHub Discussions**: For questions and general discussion
- **GitHub Issues**: For bug reports and feature requests
- **Stack Overflow**: Tag questions with `pycallmeter`

### Code of Conduct

This project adheres to a Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the maintainers.

### Recognition

Contributors are recognized in:
- CHANGELOG.md for significant contributions
- GitHub contributors list
- Special thanks in release notes

## 📚 Resources

### Useful Links

- **Project Repository**: https://github.com/sattyamjain/pycallmeter
- **Issue Tracker**: https://github.com/sattyamjain/pycallmeter/issues
- **PyPI Package**: https://pypi.org/project/pycallmeter/
- **Documentation**: https://github.com/sattyamjain/pycallmeter#readme

### Learning Resources

- **Python Profiling**: https://docs.python.org/3/library/profile.html
- **sys.settrace**: https://docs.python.org/3/library/sys.html#sys.settrace
- **Memory Profiling**: https://docs.python.org/3/library/tracemalloc.html
- **Async Programming**: https://docs.python.org/3/library/asyncio.html

---

## 🙏 Thank You!

Your contributions help make PyCallMeter better for everyone. Whether you're fixing a typo, adding a feature, or reporting a bug, every contribution is valuable and appreciated!

**Happy coding!** 🎉