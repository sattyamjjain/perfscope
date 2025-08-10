# Changelog

All notable changes to PyCallMeter will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Production-ready logging with [PyCallMeter] identifiers
- Smart threshold-based function logging (>0.1ms by default)
- Enhanced HTML report generation with interactive call trees
- Memory leak detection and analysis capabilities
- Web framework integration examples (Django, Flask, FastAPI)
- Comprehensive configuration options for production use
- Performance budgets and alerting capabilities

### Changed
- Improved log format for better readability and parsing
- Optimized performance with reduced overhead (2-5% in production)
- Enhanced memory tracking accuracy and garbage collection monitoring
- Better error handling and exception reporting

### Fixed
- Duplicate logging issues resolved with logger.propagate = False
- Memory leak in profiling metadata collection
- Thread safety improvements for concurrent applications
- Async/await compatibility issues in Python 3.11+

## [1.0.0] - 2024-01-15

### Added
- Initial release of PyCallMeter
- Core profiling functionality with function call tracing
- Memory usage monitoring and tracking
- Async/await support for modern Python applications
- HTML and JSON report generation
- Zero-dependency core with optional psutil integration
- Type hints and mypy compatibility
- Comprehensive test suite with 95%+ coverage

### Features
- **Performance Monitoring**
  - Function call tracing with configurable depth
  - Execution time analysis (wall-clock and CPU time)
  - Memory usage tracking and peak detection
  - Call frequency analysis and hot path identification
  - Bottleneck detection with configurable thresholds

- **Developer Experience**
  - Single decorator setup (@profile())
  - Zero configuration for basic use cases
  - IDE integration with full type hints
  - Production-ready logging format
  - Framework compatibility (Django, Flask, FastAPI)

- **Advanced Analytics** 
  - Interactive call tree visualization
  - Memory leak detection capabilities
  - Async/concurrent profiling support
  - Statistical analysis (min/max/avg times)
  - Export options (HTML, JSON, CSV)

- **Enterprise Features**
  - Configurable logging levels
  - Module filtering (include/exclude)
  - Depth control for performance optimization
  - Thread safety for multi-threaded applications
  - Production deployment ready

### Technical Details
- **Python Support**: 3.8, 3.9, 3.10, 3.11, 3.12
- **Platforms**: Linux, macOS, Windows
- **Dependencies**: Zero runtime dependencies (optional psutil for enhanced memory tracking)
- **Performance**: <5% overhead in production configurations
- **Type Safety**: Full type hints with mypy validation
- **Testing**: Comprehensive test suite with pytest
- **Documentation**: Complete API documentation and examples

### Architecture
- Thread-local storage for call stacks
- Efficient memory tracking with tracemalloc integration
- Configurable trace function with sys.settrace
- Atomic report generation with comprehensive statistics
- Plugin architecture for framework integrations

---

## Release Notes

### What's New in v1.0.0

PyCallMeter v1.0.0 introduces a revolutionary approach to Python performance profiling:

🎯 **Zero Configuration Profiling**: Just add `@profile()` to any function and get instant performance insights.

🏭 **Production Ready**: Designed for production use with minimal overhead and enterprise-grade logging.

🔍 **Deep Insights**: Get comprehensive performance analytics including:
- Function execution times (wall-clock and CPU)
- Memory usage and leak detection  
- Call tree visualization
- Bottleneck identification
- Statistical analysis

🌐 **Modern Python Support**: Full compatibility with:
- Async/await applications
- Web frameworks (Django, Flask, FastAPI)
- Data science pipelines
- Machine learning workflows
- Microservices architectures

📊 **Rich Reporting**: Export detailed reports in multiple formats:
- Interactive HTML with call trees
- JSON for programmatic analysis  
- CSV for spreadsheet analysis
- Console output for development

### Migration Guide

This is the initial release, so no migration is needed. Simply install and start profiling:

```bash
pip install pycallmeter
```

```python
from pycallmeter import profile

@profile()
def your_function():
    # Your code here
    pass
```

### Breaking Changes

None - this is the initial release.

### Deprecation Warnings

None - this is the initial release.

### Known Issues

- Line-level tracing has high overhead (50-200%) - use only for debugging
- Some third-party libraries may need to be excluded from tracing for optimal performance
- Memory tracking requires Python 3.8+ and may have platform-specific behaviors

### Contributors

Special thanks to all contributors who made this release possible:

- [@sattyamjain](https://github.com/sattyamjain) - Project creator and maintainer

### Community

- **GitHub**: https://github.com/sattyamjain/pycallmeter
- **Issues**: https://github.com/sattyamjain/pycallmeter/issues
- **Discussions**: https://github.com/sattyamjain/pycallmeter/discussions
- **PyPI**: https://pypi.org/project/pycallmeter/

---

*For more information about PyCallMeter, visit our [GitHub repository](https://github.com/sattyamjain/pycallmeter) or check out our [comprehensive documentation](https://github.com/sattyamjain/pycallmeter#readme).*