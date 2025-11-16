# Tests

This directory contains unit tests and integration tests for the Romanian LLM fine-tuning project.

## Test Structure

```
tests/
├── __init__.py
├── test_data_processing.py    # Tests for data processing functions
├── test_config_validation.py  # Tests for configuration validation
└── README.md                   # This file
```

## Running Tests

### Run all tests

```bash
pytest
```

### Run specific test file

```bash
pytest tests/test_data_processing.py
```

### Run specific test class

```bash
pytest tests/test_data_processing.py::TestTextCleaning
```

### Run specific test

```bash
pytest tests/test_data_processing.py::TestTextCleaning::test_clean_text_removes_extra_whitespace
```

### Run with coverage report

```bash
pytest --cov=scripts --cov-report=html
```

This will generate an HTML coverage report in `htmlcov/index.html`.

### Run only fast tests (skip slow tests)

```bash
pytest -m "not slow"
```

## Test Coverage

The test suite covers:

### Data Processing (`test_data_processing.py`)
- ✅ Text cleaning and normalization
- ✅ Romanian language detection
- ✅ Text validation (length, diacritics, punctuation)
- ✅ Q&A pair extraction from Wikipedia
- ✅ Instruction example creation
- ✅ JSONL file validation
- ✅ Train/validation split

### Configuration Validation (`test_config_validation.py`)
- ✅ Model configuration validation
- ✅ LoRA configuration validation
- ✅ Training configuration validation
- ✅ Optimizer configuration validation
- ✅ Romanian-specific configuration validation
- ✅ Complete hyperparameters validation
- ✅ Error handling and edge cases

## Writing New Tests

When adding new functionality, please add corresponding tests:

1. **Create a test file** following the naming convention `test_*.py`
2. **Import the module** to test from the `scripts/` directory
3. **Write test classes** for each logical component
4. **Write test methods** starting with `test_`
5. **Use fixtures** in `setup_method()` for common setup
6. **Add assertions** to verify expected behavior

### Example Test

```python
class TestMyFunction:
    """Test MyFunction functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.data = {"key": "value"}

    def test_basic_functionality(self):
        """Test basic functionality."""
        result = my_function(self.data)
        assert result == expected_value
```

## Test Best Practices

1. **Test one thing at a time** - Each test should verify a single behavior
2. **Use descriptive names** - Test names should clearly describe what they test
3. **Arrange-Act-Assert** - Structure tests with setup, execution, and verification
4. **Test edge cases** - Include tests for boundary conditions and error cases
5. **Keep tests independent** - Tests should not depend on each other
6. **Use appropriate assertions** - Choose the right assertion for clarity

## Continuous Integration

These tests are designed to run in CI/CD pipelines. The pytest configuration in `pytest.ini` includes:

- Coverage reporting
- Strict marker enforcement
- Detailed failure output
- Multiple output formats (terminal, HTML, XML)

## Troubleshooting

### Import errors

If you get import errors, make sure the `scripts/` directory is in your Python path:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/scripts
pytest
```

Or use the `-s` flag to see print statements:

```bash
pytest -s
```

### Missing dependencies

Install test dependencies:

```bash
pip install pytest pytest-cov
```

### Tests failing

1. Check that you're in the project root directory
2. Ensure all dependencies are installed: `pip install -r requirements.txt`
3. Run with verbose output: `pytest -vv`
4. Check test logs for specific error messages
