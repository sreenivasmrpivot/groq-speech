# Testing Guide - Groq Speech SDK

This document provides comprehensive information about testing the Groq Speech SDK.

## 🧪 Test Suite Overview

The test suite is organized into several categories:

### **Test Categories**

1. **Unit Tests** - Test individual components in isolation
2. **Integration Tests** - Test component interactions
3. **End-to-End Tests** - Test complete workflows
4. **CLI Tests** - Test command-line interface
5. **API Tests** - Test REST API and WebSocket endpoints

### **Test Files Structure**

```
tests/
├── test_config.py              # Shared test configuration and utilities
├── test_e2e_cli.py            # End-to-end CLI tests
├── test_e2e_api.py            # End-to-end API tests
├── test_basic_functionality.py # Basic SDK functionality tests
├── test_speech_demo.py        # CLI argument parsing tests
├── test_transcription_accuracy.py # Transcription accuracy tests
├── test_audio_processor.py    # Audio processing tests
└── unit/
    └── test_speech_config.py  # SpeechConfig unit tests
```

## 🚀 Quick Start

### **Run All Tests**
```bash
# Using the test runner
python run_tests.py

# Using Make
make test

# Using pytest directly
pytest tests/ -v
```

### **Run Specific Test Categories**
```bash
# Unit tests only
python run_tests.py --unit
make test-unit

# End-to-end tests only
python run_tests.py --e2e
make test-e2e

# CLI tests only
python run_tests.py --cli
make test-cli

# API tests only
make test-api
```

### **Run with Verbose Output**
```bash
python run_tests.py --verbose
make test
```

## 📋 Test Coverage

### **CLI Command Coverage**

The test suite covers all CLI command combinations:

| Command | Test File | Status |
|---------|-----------|--------|
| `python speech_demo.py --file test1.wav` | `test_e2e_cli.py` | ✅ |
| `python speech_demo.py --file test1.wav --diarize` | `test_e2e_cli.py` | ✅ |
| `python speech_demo.py --microphone-mode single` | `test_e2e_cli.py` | ✅ |
| `python speech_demo.py --microphone-mode single --diarize` | `test_e2e_cli.py` | ✅ |
| `python speech_demo.py --microphone-mode single --operation translation` | `test_e2e_cli.py` | ✅ |
| `python speech_demo.py --microphone-mode single --operation translation --diarize` | `test_e2e_cli.py` | ✅ |
| `python speech_demo.py --microphone-mode continuous` | `test_e2e_cli.py` | ✅ |
| `python speech_demo.py --microphone-mode continuous --diarize` | `test_e2e_cli.py` | ✅ |
| `python speech_demo.py --microphone-mode continuous --operation translation` | `test_e2e_cli.py` | ✅ |
| `python speech_demo.py --microphone-mode continuous --operation translation --diarize` | `test_e2e_cli.py` | ✅ |

### **API Endpoint Coverage**

| Endpoint | Test File | Status |
|----------|-----------|--------|
| `GET /health` | `test_e2e_api.py` | ✅ |
| `POST /api/v1/recognize` | `test_e2e_api.py` | ✅ |
| `POST /api/v1/translate` | `test_e2e_api.py` | ✅ |
| `WebSocket /ws/recognize` | `test_e2e_api.py` | ✅ |

## 🔧 Test Configuration

### **Environment Variables**

Tests use mock environment variables by default:

```python
TEST_ENV_VARS = {
    'GROQ_API_KEY': 'test_groq_api_key_here',
    'HF_TOKEN': 'test_hf_token_here',
    'GROQ_MODEL_ID': 'whisper-large-v3',
    # ... other test variables
}
```

### **Test Audio Files**

The test suite automatically creates test audio files if they don't exist:

- `examples/test1.wav` - 3-second sine wave (440Hz)
- `examples/test2.wav` - 3-second sine wave (440Hz)
- `examples/test3.mp3` - 3-second sine wave (440Hz)
- `examples/test4.mp3` - 3-second sine wave (440Hz)

### **Mock Objects**

Tests use mock objects to avoid actual API calls:

```python
MOCK_RECOGNITION_RESULT = MagicMock(
    text="This is a test transcription result",
    confidence=0.95,
    language="en-US"
)

MOCK_DIARIZATION_RESULT = MagicMock(
    text="This is a test transcription with diarization",
    num_speakers=2,
    segments=[...]
)
```

## 🧪 Running Tests

### **Prerequisites**

1. **Python Environment**: Ensure you're in the correct virtual environment
2. **Dependencies**: Install test dependencies
   ```bash
   pip install -r requirements-dev.txt
   ```
3. **API Server** (for API tests): Start the API server
   ```bash
   python -m api.server
   ```

### **Test Commands**

#### **Basic Test Commands**
```bash
# Run all tests
python run_tests.py

# Run with verbose output
python run_tests.py --verbose

# Run and stop on first failure
python run_tests.py --failfast
```

#### **Specific Test Categories**
```bash
# Unit tests only
python run_tests.py --unit

# End-to-end tests only
python run_tests.py --e2e

# CLI tests only
python run_tests.py --cli
```

#### **Using Make**
```bash
# All tests
make test

# Unit tests
make test-unit

# End-to-end tests
make test-e2e

# CLI tests
make test-cli

# API tests
make test-api

# Quick tests (unit only)
make test-quick
```

#### **Using pytest directly**
```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_e2e_cli.py -v

# Specific test method
pytest tests/test_e2e_cli.py::TestE2ECLI::test_file_transcription_basic -v

# With coverage
pytest tests/ --cov=groq_speech --cov=api --cov-report=html
```

## 🔍 Test Details

### **CLI Tests (`test_e2e_cli.py`)**

Tests all CLI command combinations by running them as subprocess calls:

- **File Processing**: Tests file transcription and translation with/without diarization
- **Microphone Processing**: Tests single and continuous modes with/without diarization
- **Error Handling**: Tests missing API keys, invalid files, invalid arguments
- **Environment Validation**: Tests environment variable requirements

### **API Tests (`test_e2e_api.py`)**

Tests REST API and WebSocket endpoints:

- **Health Check**: Tests `/health` endpoint
- **Recognition**: Tests `/api/v1/recognize` endpoint
- **Translation**: Tests `/api/v1/translate` endpoint
- **WebSocket**: Tests `/ws/recognize` WebSocket connection
- **Error Handling**: Tests invalid requests and error responses
- **CORS**: Tests CORS headers

### **Unit Tests**

- **`test_speech_config.py`**: Tests SpeechConfig class
- **`test_basic_functionality.py`**: Tests core SDK functionality
- **`test_audio_processor.py`**: Tests audio processing components

## 🐛 Debugging Tests

### **Verbose Output**
```bash
python run_tests.py --verbose
```

### **Run Single Test**
```bash
pytest tests/test_e2e_cli.py::TestE2ECLI::test_file_transcription_basic -v -s
```

### **Debug Mode**
```bash
pytest tests/test_e2e_cli.py -v -s --pdb
```

### **Test Logging**
```bash
pytest tests/ -v -s --log-cli-level=DEBUG
```

## 📊 Test Reports

### **Coverage Report**
```bash
# Generate HTML coverage report
pytest tests/ --cov=groq_speech --cov=api --cov-report=html

# View coverage report
open htmlcov/index.html
```

### **JUnit XML Report**
```bash
pytest tests/ --junitxml=test-results.xml
```

## 🚨 Common Issues

### **API Server Not Running**
```
Error: API server not available
Solution: Start the API server with `python -m api.server`
```

### **Missing Dependencies**
```
Error: ModuleNotFoundError
Solution: Install dependencies with `pip install -r requirements-dev.txt`
```

### **Environment Variables**
```
Error: API key not configured
Solution: Set GROQ_API_KEY and HF_TOKEN environment variables
```

### **Test Timeout**
```
Error: Command timed out
Solution: Increase TIMEOUT_SECONDS in test configuration
```

## 🔄 Continuous Integration

### **GitHub Actions**
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: |
          pip install -r requirements-dev.txt
      - name: Run tests
        run: |
          python run_tests.py
```

### **Local CI**
```bash
# Run all checks
make ci-test

# Run tests with coverage
make test-coverage
```

## 📈 Test Metrics

### **Coverage Targets**
- **Overall Coverage**: > 80%
- **Core Components**: > 90%
- **API Endpoints**: > 95%
- **CLI Commands**: > 95%

### **Performance Targets**
- **Unit Tests**: < 30 seconds
- **Integration Tests**: < 2 minutes
- **End-to-End Tests**: < 5 minutes
- **Full Test Suite**: < 10 minutes

## 🎯 Best Practices

### **Writing Tests**
1. **Test One Thing**: Each test should verify one specific behavior
2. **Use Descriptive Names**: Test names should clearly describe what they test
3. **Mock External Dependencies**: Don't make actual API calls in tests
4. **Clean Up**: Always clean up resources after tests
5. **Use Fixtures**: Reuse common test setup with fixtures

### **Test Organization**
1. **Group Related Tests**: Use test classes to group related tests
2. **Use Setup/Teardown**: Use setUp and tearDown methods for common setup
3. **Test Edge Cases**: Include tests for error conditions and edge cases
4. **Document Complex Tests**: Add comments for complex test logic

### **Maintenance**
1. **Keep Tests Updated**: Update tests when code changes
2. **Remove Obsolete Tests**: Delete tests for removed functionality
3. **Monitor Test Performance**: Keep test execution time reasonable
4. **Review Test Coverage**: Regularly review and improve test coverage

## 📚 Additional Resources

- [pytest Documentation](https://docs.pytest.org/)
- [unittest Documentation](https://docs.python.org/3/library/unittest.html)
- [Mock Documentation](https://docs.python.org/3/library/unittest.mock.html)
- [Coverage Documentation](https://coverage.readthedocs.io/)

---

**Note**: This testing guide is designed to help you run and maintain the test suite effectively. For questions or issues, please refer to the project documentation or create an issue.
