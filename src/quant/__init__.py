"""JurisFinanceAI - Quantitative Finance Engine

Modules:
    portfolio       - Markowitz, Black-Litterman, Fuzzy portfolio optimization
    derivatives     - Black-Scholes, Binomial Tree, Monte Carlo options pricing
    risk_models     - VaR, CVaR, GARCH, stress testing, Altman Z-Score
    time_series     - ARIMA, statistical analysis, forecasting
    fuzzy_systems   - Fuzzy credit scoring, ANFIS, Fuzzy AHP
    network         - Graph theory, contagion, systemic risk
    behavioral      - Behavioral biases, prospect theory, sentiment
    monte_carlo     - General-purpose Monte Carlo simulation engine
    interest_rates  - Vasicek, CIR, Hull-White interest rate models
    topological     - TDA, persistent homology, chaos indicators, complexity
    generative      - Diffusion models, GAN-like synthetic data, VAE, stress scenarios
    explainability  - SHAP, LIME, counterfactual analysis, drift detection
    quantum         - QAOA, quantum MC pricing, quantum kernels, VQE
    nlp_engine      - Sentiment, NER, summarization, RAG retrieval
    gpu_compute     - Accelerated MC, GARCH, correlation, batch pricing
    data_import    - CSV/Excel data import, validation, demo generation
    report_export  - PDF/Excel/CSV report generation
    live_data      - Yahoo Finance & Binance live market data

"""

from .portfolio import PortfolioOptimizer, FuzzyNumber
from .derivatives import DerivativesPricer
from .risk_models import RiskEngine, GARCHModel
from .time_series import TimeSeriesAnalyzer, ARIMAModel
from .fuzzy_systems import FuzzyCreditScorer, FuzzyAHP, ANFISModel
from .network import NetworkAnalyzer
from .behavioral import BehavioralAnalyzer, ProspectTheory
from .monte_carlo import MonteCarloEngine
from .interest_rates import InterestRateModel
from .topological import TopologicalAnalyzer
from .generative import GenerativeModel
from .explainability import ExplainabilityEngine
from .quantum import QuantumFinanceEngine
from .nlp_engine import FinancialNLPEngine
from .gpu_compute import GPUAccelerator
from .data_import import DataImporter
from .report_export import ReportExporter
from .live_data import LiveDataEngine


__all__ = [
    'PortfolioOptimizer', 'FuzzyNumber',
    'DerivativesPricer',
    'RiskEngine', 'GARCHModel',
    'TimeSeriesAnalyzer', 'ARIMAModel',
    'FuzzyCreditScorer', 'FuzzyAHP', 'ANFISModel',
    'NetworkAnalyzer',
    'BehavioralAnalyzer', 'ProspectTheory',
    'MonteCarloEngine',
    'InterestRateModel',
    'TopologicalAnalyzer',
    'GenerativeModel',
    'ExplainabilityEngine',
    'QuantumFinanceEngine',
    'FinancialNLPEngine',
    'GPUAccelerator',
    'DataImporter',
    'ReportExporter',
    'LiveDataEngine',
]
