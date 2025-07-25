# VS Code Setup Guide for Groq Speech SDK

This guide will help you set up VS Code for optimal development experience with the Groq Speech SDK.

## 🚀 Quick Start

### 1. Prerequisites

- **VS Code**: Download from [code.visualstudio.com](https://code.visualstudio.com/)
- **Python 3.8+**: Install from [python.org](https://python.org/)
- **Git**: Install from [git-scm.com](https://git-scm.com/)

### 2. Install VS Code Extensions

Open VS Code and install these recommended extensions:

```bash
# Install Python extension
code --install-extension ms-python.python

# Install Pylance (Python language server)
code --install-extension ms-python.vscode-pylance

# Install Black formatter
code --install-extension ms-python.black-formatter

# Install Flake8 linter
code --install-extension ms-python.flake8

# Install MyPy type checker
code --install-extension ms-python.mypy-type-checker

# Install Pytest adapter
code --install-extension ms-python.pytest-adapter
```

Or install all recommended extensions at once:
1. Open the project in VS Code
2. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
3. Type "Extensions: Show Recommended Extensions"
4. Click "Install All"

## 🔧 Development Environment Setup

### 1. Open the Project

```bash
# Clone the repository (if not already done)
git clone <repository-url>
cd groq-speech

# Open in VS Code
code .
```

### 2. Setup Virtual Environment

**Option A: Using VS Code Tasks (Recommended)**

1. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
2. Type "Tasks: Run Task"
3. Select "Setup Complete Development Environment"
4. This will automatically:
   - Create a virtual environment (`.venv`)
   - Install dependencies
   - Install development dependencies
   - Install the package in development mode

**Option B: Manual Setup**

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -e .[dev]
```

### 3. Select Python Interpreter

1. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
2. Type "Python: Select Interpreter"
3. Choose the interpreter from `.venv/bin/python` (or `.venv\Scripts\python.exe` on Windows)

## 🐛 Debugging Setup

### 1. Available Debug Configurations

The project includes several debug configurations:

- **Debug Demo Script**: Run the main demo
- **Debug Basic Recognition Example**: Test basic recognition
- **Debug Continuous Recognition Example**: Test continuous recognition
- **Debug Tests**: Run all tests with pytest
- **Debug Single Test File**: Run a specific test file
- **Debug SDK Module**: Test SDK functionality
- **Debug with Custom API Key**: Run with environment variables
- **Debug SDK Installation**: Test installation

### 2. Start Debugging

1. Set breakpoints in your code by clicking in the gutter
2. Press `F5` or go to Run → Start Debugging
3. Select the appropriate debug configuration
4. The debugger will stop at your breakpoints

### 3. Debug Console

- Use the Debug Console to evaluate expressions
- Access variables in the current scope
- Test code snippets interactively

## 📝 Code Quality Tools

### 1. Formatting (Black)

- **Auto-format on save**: Enabled in settings
- **Manual formatting**: `Ctrl+Shift+P` → "Format Document"
- **Command line**: `python -m black . --line-length=100`

### 2. Linting (Flake8)

- **Real-time linting**: Enabled in settings
- **Problems panel**: View all linting issues
- **Command line**: `python -m flake8 . --max-line-length=100`

### 3. Type Checking (MyPy)

- **Real-time type checking**: Enabled in settings
- **Problems panel**: View type errors
- **Command line**: `python -m mypy groq_speech/`

### 4. Testing (Pytest)

- **Test Explorer**: View and run tests
- **Debug tests**: Set breakpoints in test files
- **Command line**: `python -m pytest tests/ -v`

## 🛠️ Available Tasks

### Build Tasks

- **Setup Complete Development Environment**: Full setup automation
- **Setup Virtual Environment**: Create `.venv`
- **Install Dependencies**: Install from `requirements.txt`
- **Upgrade Dependencies**: Update all packages
- **Install Development Dependencies**: Install dev tools
- **Install Package in Development Mode**: Install SDK for development

### Test Tasks

- **Run Tests**: Run all tests with pytest
- **Run Tests with Coverage**: Generate coverage report
- **Run All Quality Checks**: Format, lint, type check, and test

### Development Tasks

- **Format Code**: Run Black formatter
- **Lint Code**: Run Flake8 linter
- **Type Check**: Run MyPy type checker
- **Clean Build Files**: Remove temporary files
- **Build Package**: Create distribution packages

### Example Tasks

- **Run Demo**: Execute the main demo script
- **Run Basic Recognition Example**: Test basic functionality
- **Run Continuous Recognition Example**: Test continuous recognition

## 🔍 Troubleshooting

### Common Issues

#### 1. Python Interpreter Not Found

**Symptoms**: VS Code can't find Python interpreter

**Solution**:
1. Press `Ctrl+Shift+P`
2. Type "Python: Select Interpreter"
3. Choose the correct interpreter from `.venv/bin/python`

#### 2. Import Errors

**Symptoms**: Module import errors in the Problems panel

**Solution**:
1. Make sure you're using the correct Python interpreter
2. Check that the virtual environment is activated
3. Run "Install Dependencies" task

#### 3. Audio Device Issues

**Symptoms**: Audio-related errors

**Solution**:
1. Check audio device connections
2. Run the "Audio Devices" test in `test_installation.py`
3. Ensure microphone permissions are granted

#### 4. API Key Issues

**Symptoms**: Authentication errors

**Solution**:
1. Set your GROQ_API_KEY environment variable
2. Use the "Debug with Custom API Key" configuration
3. Check the API key in the terminal: `echo $GROQ_API_KEY`

### Debugging Tips

1. **Use the Debug Console**: Evaluate expressions and test code
2. **Set Conditional Breakpoints**: Right-click on breakpoint → Edit
3. **Use Watch Expressions**: Add variables to watch in debug panel
4. **Step Through Code**: Use F10 (step over) and F11 (step into)
5. **Inspect Variables**: Use the Variables panel in debug view

## 📋 Keyboard Shortcuts

### Debugging
- `F5`: Start debugging
- `F9`: Toggle breakpoint
- `F10`: Step over
- `F11`: Step into
- `Shift+F11`: Step out
- `Ctrl+Shift+F5`: Restart debugging
- `Shift+F5`: Stop debugging

### Code Navigation
- `Ctrl+Click`: Go to definition
- `F12`: Go to definition
- `Alt+F12`: Peek definition
- `Shift+F12`: Find all references
- `Ctrl+T`: Go to symbol in workspace

### Editing
- `Ctrl+Space`: Trigger suggestions
- `Ctrl+Shift+Space`: Trigger parameter hints
- `Ctrl+K Ctrl+C`: Add line comment
- `Ctrl+K Ctrl+U`: Remove line comment
- `Alt+Shift+F`: Format document

### Tasks
- `Ctrl+Shift+P`: Command palette
- `Ctrl+Shift+P` → "Tasks: Run Task": Run a task
- `Ctrl+Shift+P` → "Tasks: Run Build Task": Run build task

## 🎯 Best Practices

### 1. Code Organization
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions small and focused
- Use type hints for better IDE support

### 2. Testing
- Write tests for new functionality
- Run tests before committing
- Use descriptive test names
- Test both success and error cases

### 3. Debugging
- Set breakpoints at the start of functions
- Use print statements for quick debugging
- Use the debug console for exploration
- Check the Problems panel regularly

### 4. Git Integration
- Use the Source Control panel
- Write meaningful commit messages
- Review changes before committing
- Use branches for new features

## 🚀 Next Steps

1. **Set up your API key**:
   ```bash
   export GROQ_API_KEY="your-groq-api-key"
   ```

2. **Run the installation test**:
   ```bash
   python test_installation.py
   ```

3. **Try the demo**:
   ```bash
   python demo.py
   ```

4. **Explore the examples**:
   ```bash
   python examples/basic_recognition.py
   python examples/continuous_recognition.py
   ```

5. **Run the tests**:
   ```bash
   python -m pytest tests/ -v
   ```

## 📚 Additional Resources

- [VS Code Python Documentation](https://code.visualstudio.com/docs/languages/python)
- [Python Debugging Guide](https://code.visualstudio.com/docs/python/debugging)
- [Pytest Documentation](https://docs.pytest.org/)
- [Black Formatter](https://black.readthedocs.io/)
- [Flake8 Documentation](https://flake8.pycqa.org/)
- [MyPy Documentation](https://mypy.readthedocs.io/)

Happy coding! 🎉 