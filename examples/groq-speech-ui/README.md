# Groq Speech Recognition Demo

A comprehensive Next.js application that demonstrates real-time speech transcription and translation using the Groq Speech API. This application provides both single-shot and continuous recognition modes with detailed performance metrics and visualizations.

## Features

### 🎤 Speech Recognition
- **Single Shot Mode**: Record audio and get immediate transcription/translation
- **Continuous Mode**: Real-time streaming recognition with WebSocket support
- **Multiple Languages**: Support for 10+ languages including English, German, French, Spanish, etc.
- **Translation**: Convert speech to English text from any supported language

### 📊 Performance Metrics
- **Real-time Timing**: Track microphone capture, API call, and response processing times
- **Visual Charts**: Interactive charts showing performance breakdown and trends
- **Success Rates**: Monitor recognition success and failure rates
- **Historical Data**: View performance metrics over time

### 🎨 User Interface
- **Responsive Design**: Works seamlessly on desktop and mobile devices
- **Modern UI**: Clean, intuitive interface with Tailwind CSS styling
- **Real-time Feedback**: Visual indicators for recording, processing, and results
- **Export Functionality**: Download results as JSON files

### 🔧 Configuration
- **API Key Management**: Secure storage and management of Groq API keys
- **Mock Mode**: Demo mode for testing without real API calls
- **Environment Variables**: Configurable API endpoints and settings

## Technology Stack

- **Frontend**: Next.js 14, React 18, TypeScript
- **Styling**: Tailwind CSS
- **Charts**: Recharts
- **Icons**: Lucide React
- **Audio Processing**: Web Audio API, MediaRecorder API
- **Backend Integration**: Groq Speech API

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn
- Groq API key (optional for mock mode)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd groq-speech-ui
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start the development server**
   ```bash
   npm run dev
   ```

4. **Open your browser**
   Navigate to [http://localhost:3000](http://localhost:3000)

### Configuration

#### Using Real Groq API

1. Get your API key from the [Groq Console](https://console.groq.com/)
2. Click the "Settings" button in the app
3. Enter your API key and save
4. Disable "Use Mock API" option

#### Using Mock API (Demo Mode)

1. Enable "Use Mock API" in the settings
2. No API key required
3. Simulates real API responses for demonstration

## Usage

### Single Shot Recognition

1. Select "Single Shot" mode
2. Choose operation type (Transcription or Translation)
3. Select your language
4. Click "Start Recording"
5. Speak into your microphone
6. Click "Stop Recording" when done
7. View results and performance metrics

### Continuous Recognition

1. Select "Continuous" mode
2. Choose operation type and language
3. Click "Start Recording"
4. Speak continuously - results will appear in real-time
5. Click "Stop Recording" to end

### Performance Metrics

1. Click "Show Metrics" to view detailed performance data
2. View timing breakdown charts
3. Monitor success rates and response times
4. Export data for analysis

## API Integration

The application integrates with the Groq Speech API through:

- **REST API**: For single-shot recognition
- **WebSocket**: For continuous streaming recognition
- **Audio Processing**: Converts audio to WAV format for API compatibility

### API Endpoints

- `POST /api/v1/recognize` - Single-shot transcription
- `POST /api/v1/translate` - Single-shot translation
- `WS /ws/recognize` - WebSocket for continuous recognition

## Performance Features

### Timing Metrics
- **Microphone Capture**: Time to capture audio from microphone
- **API Call**: Time for Groq API to process and respond
- **Response Processing**: Time to parse and display results
- **Total Time**: End-to-end processing time

### Visualizations
- **Pie Charts**: Timing breakdown percentages
- **Bar Charts**: Success/failure distribution
- **Line Charts**: Performance trends over time
- **Real-time Updates**: Live metric updates during recognition

## File Structure

```
src/
├── app/                    # Next.js app directory
│   ├── layout.tsx         # Root layout
│   ├── page.tsx           # Main page component
│   └── globals.css        # Global styles
├── components/             # React components
│   ├── SpeechRecognition.tsx  # Main recognition component
│   └── PerformanceMetrics.tsx # Performance charts
├── lib/                   # Utility libraries
│   ├── audio-recorder.ts  # Audio recording utilities
│   └── groq-api.ts       # API client
└── types/                 # TypeScript type definitions
    └── index.ts           # Application types
```

## Development

### Running in Development Mode

```bash
npm run dev
```

### Building for Production

```bash
npm run build
npm start
```

### Code Quality

```bash
npm run lint
npm run type-check
```

## Environment Variables

Create a `.env.local` file for custom configuration:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_GROQ_API_KEY=your_api_key_here
```

## Troubleshooting

### Microphone Access Issues

1. Ensure your browser has microphone permissions
2. Check that your microphone is working in other applications
3. Try refreshing the page and granting permissions again

### API Connection Issues

1. Verify your Groq API key is correct
2. Check your internet connection
3. Ensure the API endpoint is accessible
4. Try using mock mode for testing

### Performance Issues

1. Check browser console for errors
2. Ensure you have sufficient system resources
3. Try closing other applications using microphone
4. Check network latency to Groq API

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions:
- Check the troubleshooting section
- Review the Groq API documentation
- Open an issue on GitHub

## Acknowledgments

- Groq for providing the Speech API
- Next.js team for the excellent framework
- Recharts for the charting library
- Lucide for the beautiful icons
