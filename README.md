# JurisFinanceAI

**دستیار هوش مصنوعی حقوقی و مالی حرفه‌ای**

A professional AI-powered legal and financial desktop application for Windows, featuring document analysis, contract review, quantitative finance, and risk assessment.

## Features

### Core
- AI Chat with legal/financial knowledge (OpenAI-powered)
- Document management and analysis (PDF, DOCX, TXT, RTF)
- Contract analysis with risk scoring
- Financial analytics dashboard
- Risk assessment engine
- Local SQLite database with encryption
- Professional dark/light theme
- Persian (Farsi) RTL interface
- Plugin system for extensibility
- Multi-language support (Persian, English, Arabic)

### Quantitative Finance (15 Modules)
| Module | Description |
|--------|-------------|
| Portfolio Optimization | Markowitz, Black-Litterman, Fuzzy, Risk Parity |
| Derivatives Pricing | Black-Scholes, Binomial Tree, Monte Carlo |
| Risk Analysis | VaR (3 methods), CVaR, Stress Testing, Altman Z |
| Time Series | ADF Test, ARIMA, Rolling Statistics |
| Fuzzy Systems | Credit Scoring, AHP, ANFIS |
| Network Analysis | Correlation Networks, Contagion, Systemic Risk |
| Behavioral Finance | Disposition Effect, Herd Behavior, Prospect Theory |
| Monte Carlo | GBM, Portfolio Simulation, Credit Risk |
| Interest Rates | Vasicek, CIR, Hull-White, Duration/Convexity |
| Topological Analysis | Hurst, Lyapunov, Betti Numbers, Fractal Dimension |
| Generative Models | Diffusion Scenarios, Stress Testing |
| Explainability | SHAP, LIME, Feature Importance, Model Drift |
| Quantum Finance | QAOA, QMC, VQE, Entanglement Measures |
| NLP | Sentiment, NER, Summarization, RAG Retrieval |
| GPU Acceleration | Vectorized MC, GARCH, Correlation, Benchmarking |

### Data & Integration
- Live market data (Yahoo Finance, Binance)
- CSV/Excel data import
- Report export (PDF, Excel, CSV)

## Installation

### From Release (Recommended)
1. Download the latest release from [GitHub Releases](https://github.com/Ali-Marandi/JurisFinanceAI/releases)
2. Extract the ZIP file
3. Run `JurisFinanceAI.exe`
4. (Optional) Enter your OpenAI API key in Settings

### From Source
```bash
git clone https://github.com/Ali-Marandi/JurisFinanceAI.git
cd JurisFinanceAI
pip install -r requirements.txt
python main.py
```

### Build from Source (Windows)
```bash
pip install pyinstaller
pyinstaller build.spec --noconfirm
```
The executable will be in `dist/JurisFinanceAI/`.

## Requirements

- Python 3.12+
- Windows 10/11
- (Optional) OpenAI API key for AI features

## Dependencies

```
PyQt6
httpx
cryptography
openai
scipy
matplotlib
numpy
openpyxl
reportlab
```

## Project Structure

```
JurisFinanceAI/
├ main.py                 # Entry point
├ src/
│   ├ __init__.py           # Version info
│   ├ app.py               # Application controller
│   ├ core/                # Core systems
│   │   ├ config.py        # Configuration (encrypted API keys)
│   │   ├ database.py      # SQLite with WAL mode
│   │   ├ ai_engine.py     # OpenAI integration
│   │   ├ document_parser.py
│   │   ├ plugin_system.py # Plugin microkernel
│   │   └ i18n.py          # Internationalization
│   ├ quant/               # Quantitative finance engine (20+ classes)
│   │   ├ portfolio.py     # Portfolio optimization
│   │   ├ derivatives.py   # Options pricing
│   │   ├ risk_models.py   # VaR, CVaR, GARCH
│   │   ├ time_series.py  # ARIMA, ADF
│   │   ├ fuzzy_systems.py # Fuzzy logic
│   │   ├ network.py      # Network analysis
│   │   ├ behavioral.py   # Behavioral finance
│   │   ├ monte_carlo.py  # Monte Carlo simulation
│   │   ├ interest_rates.py
│   │   ├ topological.py  # TDA, fractals
│   │   ├ generative.py   # Diffusion models
│   │   ├ explainability.py # XAI
│   │   ├ quantum.py      # Quantum finance
│   │   ├ nlp_engine.py   # Financial NLP
│   │   ├ gpu_compute.py  # GPU acceleration
│   │   ├ data_import.py
│   │   ├ report_export.py
│   │   └ live_data.py
│   └ ui/                  # PyQt6 interface
│       ├ main_window.py   # Main window + sidebar
│       ├ quant_dashboard.py # 15-tab quant dashboard
│       ├ dashboard.py     # Analytics dashboard
│       ├ ai_chat.py      # AI chat interface
│       ├ documents.py    # Document management
│       ├ contracts.py    # Contract analysis
│       ├ finance.py      # Financial analysis
│       ├ risk.py         # Risk assessment
│       ├ settings.py     # Settings page
│       └ themes.py       # QSS themes
├ tests/                   # Test suite (pytest)
├ build.spec               # PyInstaller config
├ requirements.txt
└ .github/workflows/        # CI/CD
```

## Testing

```bash
pip install pytest
python -m pytest tests/ -v
```

## License

MIT License - see [LICENSE](LICENSE) file.

## Author

Ali Marandi
