# Contributing to Groq Speech SDK

Thank you for your interest in contributing to the Groq Speech SDK! This document provides guidelines and information for contributors.

## 🤝 **How to Contribute**

### **Types of Contributions**

We welcome contributions in the following areas:

- **🐛 Bug Reports** - Help us identify and fix issues
- **✨ Feature Requests** - Suggest new functionality
- **📚 Documentation** - Improve guides and examples
- **🧪 Tests** - Add test coverage and improve reliability
- **🔧 Code Improvements** - Enhance performance and maintainability
- **🌐 Translations** - Help with internationalization
- **📖 Examples** - Create useful examples and tutorials

### **Before You Start**

1. **Check Existing Issues** - Search for similar issues before creating new ones
2. **Read Documentation** - Familiarize yourself with the project structure
3. **Join Discussions** - Participate in GitHub discussions
4. **Follow Guidelines** - Read this guide thoroughly

## 🚀 **Quick Start**

### **1. Fork and Clone**

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/groq-speech-sdk.git
cd groq-speech-sdk

# Add upstream remote
git remote add upstream https://github.com/groq/groq-speech-sdk.git
```

### **2. Set Up Development Environment**

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install development dependencies
make dev-setup

# Or manually:
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
pre-commit install
```

### **3. Create a Branch**

```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/your-bug-description
```

### **4. Make Your Changes**

- Write your code following our [coding standards](#coding-standards)
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### **5. Test Your Changes**

```bash
# Run all tests
make test

# Run specific test categories
make test-unit
make test-integration
make test-e2e

# Run code quality checks
make quality

# Run with coverage
make test-coverage
```

### **6. Submit Your Contribution**

```bash
# Commit your changes
git add .
git commit -m "feat: add new feature description"

# Push to your fork
git push origin feature/your-feature-name

# Create a Pull Request on GitHub
```

## 📋 **Issue Guidelines**

### **Bug Reports**

When reporting bugs, please include:

- **Clear Description** - What happened vs. what you expected
- **Reproduction Steps** - Step-by-step instructions
- **Environment Details** - OS, Python version, dependencies
- **Error Messages** - Full error traceback
- **Code Example** - Minimal code to reproduce the issue

**Template:**
```markdown
## Bug Report

### Description
[Clear description of the bug]

### Steps to Reproduce
1. [Step 1]
2. [Step 2]
3. [Step 3]

### Expected Behavior
[What you expected to happen]

### Actual Behavior
[What actually happened]

### Environment
- OS: [e.g., macOS 12.0]
- Python: [e.g., 3.11.0]
- SDK Version: [e.g., 1.0.0]

### Error Messages
```
[Full error traceback]
```

### Code Example
```python
# Minimal code to reproduce
```

### Additional Context
[Any other relevant information]
```

### **Feature Requests**

When requesting features, please include:

- **Use Case** - Why this feature is needed
- **Proposed Solution** - How you think it should work
- **Alternatives Considered** - Other approaches you've thought about
- **Impact** - Who would benefit from this feature

## 💻 **Coding Standards**

### **Python Code Style**

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

- **Line Length**: 100 characters maximum
- **Import Order**: Standard library, third-party, local
- **Type Hints**: Required for all public functions
- **Docstrings**: Google style for all public functions

### **Code Quality Tools**

We use several tools to maintain code quality:

```bash
# Format code
make format

# Check formatting
make format-check

# Run linting
make lint

# Type checking
make type-check

# Run all quality checks
make quality
```

### **Pre-commit Hooks**

We use pre-commit hooks to ensure code quality:

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

### **Code Review Checklist**

Before submitting a PR, ensure:

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] New functionality has tests
- [ ] Documentation is updated
- [ ] Type hints are included
- [ ] No security issues introduced
- [ ] Performance impact considered

## 🧪 **Testing Guidelines**

### **Test Structure**

```
tests/
├── unit/              # Unit tests
├── integration/       # Integration tests
├── e2e/              # End-to-end tests
├── performance/       # Performance tests
└── fixtures/         # Test data
```

### **Writing Tests**

- **Unit Tests**: Test individual functions/methods
- **Integration Tests**: Test component interactions
- **E2E Tests**: Test complete workflows
- **Performance Tests**: Test performance characteristics

### **Test Naming**

```python
# Good test names
def test_speech_config_initialization():
    """Test SpeechConfig initialization with valid parameters."""

def test_audio_config_with_invalid_device():
    """Test AudioConfig with invalid device ID raises exception."""

def test_recognition_with_german_language():
    """Test speech recognition with German language setting."""
```

### **Test Coverage**

- **Minimum Coverage**: 90% for new code
- **Critical Paths**: 100% coverage required
- **Edge Cases**: Include boundary conditions
- **Error Scenarios**: Test error handling

## 📚 **Documentation Standards**

### **Docstring Format**

```python
def recognize_speech(audio_data: bytes, language: str = "en-US") -> RecognitionResult:
    """Recognize speech from audio data.
    
    Args:
        audio_data: Raw audio data in bytes
        language: Language code for recognition (default: "en-US")
        
    Returns:
        RecognitionResult: Recognition result with text and confidence
        
    Raises:
        AudioError: If audio data is invalid
        RecognitionError: If recognition fails
        NetworkError: If API request fails
        
    Example:
        >>> result = recognize_speech(audio_bytes, "de-DE")
        >>> print(result.text)
        "Hallo Welt"
    """
```

### **Documentation Updates**

When adding new features, update:

- [ ] API documentation
- [ ] README.md if relevant
- [ ] Example scripts
- [ ] Configuration documentation
- [ ] Deployment guides if needed

## 🔄 **Pull Request Process**

### **1. Create Pull Request**

- Use descriptive titles
- Reference related issues
- Include summary of changes
- Add screenshots for UI changes

### **2. PR Description Template**

```markdown
## Description
[Brief description of changes]

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] All tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
- [ ] Performance impact considered

## Related Issues
Closes #[issue_number]
```

### **3. Review Process**

1. **Automated Checks** - CI/CD pipeline runs tests
2. **Code Review** - Maintainers review the code
3. **Discussion** - Address feedback and questions
4. **Approval** - At least one maintainer approval required
5. **Merge** - Changes merged to main branch

## 🏷️ **Commit Message Guidelines**

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### **Types**

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test changes
- `chore`: Build/tooling changes

### **Examples**

```bash
feat: add support for German language recognition
fix: resolve audio device detection issue
docs: update API reference with new endpoints
test: add unit tests for SpeechConfig class
refactor: improve error handling in AudioConfig
```

## 🚨 **Security Guidelines**

### **Security Best Practices**

- **Never commit secrets** - API keys, passwords, etc.
- **Validate inputs** - Sanitize user inputs
- **Use secure defaults** - Implement secure by default
- **Follow OWASP guidelines** - Web security best practices
- **Report vulnerabilities** - Use security advisories

### **Reporting Security Issues**

For security issues, please:

1. **Don't create public issues** for security vulnerabilities
2. **Email security@groq.com** with details
3. **Include reproduction steps** and impact assessment
4. **Allow time for response** before public disclosure

## 🌍 **Internationalization**

### **Translation Guidelines**

- **Use English** for code comments and documentation
- **Support Unicode** in all text processing
- **Provide translations** for user-facing messages
- **Test with different languages** and character sets

## 📊 **Performance Guidelines**

### **Performance Considerations**

- **Profile code** before optimizing
- **Measure impact** of changes
- **Consider memory usage** for large audio files
- **Optimize network requests** and caching
- **Test with realistic data** sizes

## 🎯 **Release Process**

### **Versioning**

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### **Release Checklist**

- [ ] All tests pass
- [ ] Documentation updated
- [ ] Changelog updated
- [ ] Version bumped
- [ ] Release notes prepared
- [ ] Security review completed

## 🤝 **Community Guidelines**

### **Code of Conduct**

We are committed to providing a welcoming and inclusive environment. Please:

- **Be respectful** and inclusive
- **Listen to others** and consider their perspectives
- **Give constructive feedback**
- **Help newcomers** learn and contribute
- **Report inappropriate behavior** to maintainers

### **Communication Channels**

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Discord**: Real-time chat and community
- **Email**: support@groq.com for private matters

## 🏆 **Recognition**

### **Contributor Recognition**

We recognize contributors through:

- **Contributor Hall of Fame** in documentation
- **Release notes** mentioning contributors
- **Special thanks** for significant contributions
- **Contributor badges** on GitHub profiles

### **Getting Help**

If you need help contributing:

1. **Check documentation** first
2. **Search existing issues** for similar questions
3. **Ask in discussions** for general questions
4. **Contact maintainers** for specific guidance

## 📝 **License**

By contributing to this project, you agree that your contributions will be licensed under the same license as the project (MIT License).

---

**Thank you for contributing to the Groq Speech SDK!** 🎉

Your contributions help make speech recognition more accessible and powerful for developers worldwide. 