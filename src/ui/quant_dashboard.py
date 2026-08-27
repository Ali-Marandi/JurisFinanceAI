import numpy as np
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf')
plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel,
                             QPushButton, QTabWidget, QTableWidget, QTableWidgetItem,
                             QHeaderView, QComboBox, QDoubleSpinBox, QSpinBox,
                             QTextEdit, QGroupBox, QFormLayout, QAbstractItemView,
                             QSplitter, QScrollArea, QGridLayout)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QColor
from ..quant import (PortfolioOptimizer, FuzzyNumber, DerivativesPricer,
                     RiskEngine, TimeSeriesAnalyzer, FuzzyCreditScorer,
                     NetworkAnalyzer, BehavioralAnalyzer, MonteCarloEngine,
                     InterestRateModel)


# ──────────────────────────────────────────────────────────────
# Dark theme stylesheet
# ──────────────────────────────────────────────────────────────
DARK_STYLE = """
QWidget#QuantDashboard { background-color: #1e2a45; color: #f1f5f9; }
QTabWidget::pane { border: 1px solid #334155; background-color: #1e2a45; }
QTabBar::tab {
    background-color: #1a2438; color: #94a3b8; padding: 10px 18px;
    border: 1px solid #334155; border-bottom: none; border-top-left-radius: 6px;
    border-top-right-radius: 6px; margin-right: 2px; font-size: 12px;
}
QTabBar::tab:selected { background-color: #1e2a45; color: #3b82f6; border-bottom: 2px solid #3b82f6; }
QTabBar::tab:hover { color: #e2e8f0; background-color: #243352; }
QGroupBox {
    color: #f1f5f9; border: 1px solid #334155; border-radius: 8px;
    margin-top: 14px; padding-top: 18px; font-weight: bold; font-size: 13px;
}
QGroupBox::title { subcontrol-origin: margin; left: 14px; padding: 0 6px; color: #3b82f6; }
QDoubleSpinBox, QSpinBox, QComboBox, QTextEdit {
    background-color: #0f172a; color: #f1f5f9; border: 1px solid #334155;
    border-radius: 4px; padding: 4px 8px; min-height: 24px; font-size: 12px;
}
QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus, QTextEdit:focus {
    border: 1px solid #3b82f6;
}
QComboBox::drop-down { border: none; width: 24px; }
QComboBox QAbstractItemView { background-color: #0f172a; color: #f1f5f9; selection-background-color: #3b82f6; }
QPushButton {
    background-color: #3b82f6; color: #ffffff; border: none; border-radius: 6px;
    padding: 8px 20px; font-weight: bold; font-size: 12px; min-height: 28px;
}
QPushButton:hover { background-color: #2563eb; }
QPushButton:pressed { background-color: #1d4ed8; }
QPushButton#demoBtn { background-color: #8b5cf6; }
QPushButton#demoBtn:hover { background-color: #7c3aed; }
QTableWidget {
    background-color: #0f172a; color: #f1f5f9; border: 1px solid #334155;
    border-radius: 6px; gridline-color: #1e293b; font-size: 12px;
}
QTableWidget::item { padding: 4px; }
QTableWidget::item:selected { background-color: #1e3a5f; }
QHeaderView::section {
    background-color: #1a2438; color: #94a3b8; border: 1px solid #334155;
    padding: 6px 8px; font-weight: bold; font-size: 11px;
}
QLabel { color: #f1f5f9; font-size: 12px; }
QScrollBar:vertical {
    background-color: #1a2438; width: 10px; border-radius: 5px;
}
QScrollBar::handle:vertical { background-color: #475569; border-radius: 5px; min-height: 20px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QSplitter::handle { background-color: #334155; }
QScrollArea { border: none; background-color: #1e2a45; }
"""

SUCCESS_COLOR = "#22c55e"
ERROR_COLOR = "#ef4444"
ACCENT_COLOR = "#3b82f6"
PURPLE_COLOR = "#8b5cf6"
BG_COLOR = "#1e2a45"
CARD_BG = "#0f172a"
TEXT_COLOR = "#f1f5f9"
MUTED_TEXT = "#94a3b8"
GRID_COLOR = "#334155"


# ──────────────────────────────────────────────────────────────
# Worker thread for running heavy computations off the UI thread
# ──────────────────────────────────────────────────────────────
class QuantWorker(QThread):
    """Background worker for quantitative computations."""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, task_name, **kwargs):
        super().__init__()
        self.task_name = task_name
        self.kwargs = kwargs

    def run(self):
        try:
            result = self._execute()
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

    def _execute(self):
        name = self.task_name

        if name == 'markowitz':
            engine = PortfolioOptimizer()
            return engine.markowitz_optimize(
                expected_returns=self.kwargs['expected_returns'],
                cov_matrix=self.kwargs['cov_matrix'],
                target_return=self.kwargs.get('target_return'),
                risk_free_rate=self.kwargs.get('risk_free_rate', 0.02),
                short_selling=self.kwargs.get('short_selling', False),
            )

        elif name == 'max_sharpe':
            engine = PortfolioOptimizer()
            return engine.maximize_sharpe(
                expected_returns=self.kwargs['expected_returns'],
                cov_matrix=self.kwargs['cov_matrix'],
                risk_free_rate=self.kwargs.get('risk_free_rate', 0.02),
            )

        elif name == 'efficient_frontier':
            engine = PortfolioOptimizer()
            return engine._compute_efficient_frontier(
                mu=np.array(self.kwargs['expected_returns']),
                Sigma=np.array(self.kwargs['cov_matrix']),
                risk_free_rate=self.kwargs.get('risk_free_rate', 0.02),
            )

        elif name == 'black_scholes':
            engine = DerivativesPricer()
            return engine.black_scholes(
                S=self.kwargs['S'], K=self.kwargs['K'],
                T=self.kwargs['T'], r=self.kwargs['r'],
                sigma=self.kwargs['sigma'],
                option_type=self.kwargs.get('option_type', 'call'),
            )

        elif name == 'binomial_tree':
            engine = DerivativesPricer()
            return engine.binomial_tree(
                S=self.kwargs['S'], K=self.kwargs['K'],
                T=self.kwargs['T'], r=self.kwargs['r'],
                sigma=self.kwargs['sigma'],
                option_type=self.kwargs.get('option_type', 'call'),
                n_steps=self.kwargs.get('n_steps', 200),
                american=self.kwargs.get('american', False),
            )

        elif name == 'mc_pricing':
            engine = DerivativesPricer()
            return engine.monte_carlo_pricing(
                S=self.kwargs['S'], K=self.kwargs['K'],
                T=self.kwargs['T'], r=self.kwargs['r'],
                sigma=self.kwargs['sigma'],
                option_type=self.kwargs.get('option_type', 'call'),
                n_simulations=self.kwargs.get('n_simulations', 10000),
            )

        elif name == 'implied_vol':
            engine = DerivativesPricer()
            return engine.implied_volatility(
                S=self.kwargs['S'], K=self.kwargs['K'],
                T=self.kwargs['T'], r=self.kwargs['r'],
                market_price=self.kwargs['market_price'],
                option_type=self.kwargs.get('option_type', 'call'),
            )

        elif name == 'var':
            engine = RiskEngine()
            return engine.value_at_risk(
                returns=self.kwargs['returns'],
                confidence=self.kwargs.get('confidence', 0.95),
                method=self.kwargs.get('method', 'historical'),
                portfolio_value=self.kwargs.get('portfolio_value', 1e6),
            )

        elif name == 'cvar':
            engine = RiskEngine()
            return engine.conditional_var(
                returns=self.kwargs['returns'],
                confidence=self.kwargs.get('confidence', 0.95),
                portfolio_value=self.kwargs.get('portfolio_value', 1e6),
            )

        elif name == 'garch':
            engine = RiskEngine()
            return engine.garch_analysis(
                returns=self.kwargs['returns'],
                forecast_horizon=self.kwargs.get('horizon', 10),
            )

        elif name == 'stress_test':
            engine = RiskEngine()
            return engine.stress_test(
                returns=self.kwargs['returns'],
                portfolio_value=self.kwargs.get('portfolio_value', 1e6),
            )

        elif name == 'ts_full':
            engine = TimeSeriesAnalyzer()
            return engine.full_analysis(
                y=self.kwargs['y'],
                forecast_steps=self.kwargs.get('forecast_steps', 20),
            )

        elif name == 'fuzzy_credit':
            engine = FuzzyCreditScorer()
            return engine.evaluate(
                income=self.kwargs['income'],
                debt_ratio=self.kwargs['debt_ratio'],
                credit_history_years=self.kwargs['credit_history_years'],
                employment_years=self.kwargs['employment_years'],
                num_accounts=self.kwargs.get('num_accounts', 5),
            )

        elif name == 'network_correlation':
            engine = NetworkAnalyzer()
            return engine.build_correlation_network(
                returns_matrix=self.kwargs['returns_matrix'],
                threshold=self.kwargs.get('threshold', 0.3),
                asset_names=self.kwargs.get('asset_names'),
            )

        elif name == 'contagion':
            engine = NetworkAnalyzer()
            return engine.simulate_contagion(
                initial_shock_node=self.kwargs['initial_shock_node'],
                adjacency=self.kwargs['adjacency'],
                shock_magnitude=self.kwargs.get('shock_magnitude', 0.1),
                recovery_rate=self.kwargs.get('recovery_rate', 0.05),
                max_rounds=self.kwargs.get('max_rounds', 50),
            )

        elif name == 'behavioral_disposition':
            engine = BehavioralAnalyzer()
            return engine.detect_disposition_effect(
                trades=self.kwargs['trades'],
                prices=self.kwargs.get('prices'),
            )

        elif name == 'behavioral_overconfidence':
            engine = BehavioralAnalyzer()
            return engine.overconfidence_analysis(
                returns=self.kwargs['returns'],
                predicted_returns=self.kwargs.get('predicted_returns'),
            )

        elif name == 'behavioral_sentiment':
            engine = BehavioralAnalyzer()
            return engine.sentiment_indicators(
                returns=self.kwargs['returns'],
                volume=self.kwargs.get('volume'),
            )

        elif name == 'behavioral_herd':
            engine = BehavioralAnalyzer()
            return engine.herd_behavior(
                returns_matrix=self.kwargs['returns_matrix'],
                market_index=self.kwargs.get('market_index'),
            )

        elif name == 'mc_gbm':
            engine = MonteCarloEngine()
            return engine.geometric_brownian_motion(
                S0=self.kwargs['S0'], mu=self.kwargs['mu'],
                sigma=self.kwargs['sigma'], T=self.kwargs['T'],
                n_steps=self.kwargs.get('n_steps', 252),
                n_simulations=self.kwargs.get('n_simulations', 1000),
            )

        elif name == 'mc_portfolio':
            engine = MonteCarloEngine()
            return engine.portfolio_simulation(
                initial_value=self.kwargs['initial_value'],
                expected_returns=self.kwargs['expected_returns'],
                cov_matrix=self.kwargs['cov_matrix'],
                n_steps=self.kwargs.get('n_steps', 252),
                n_simulations=self.kwargs.get('n_simulations', 1000),
            )

        elif name == 'vasicek':
            engine = InterestRateModel()
            return engine.vasicek(
                r0=self.kwargs['r0'], a=self.kwargs['a'],
                b=self.kwargs['b'], sigma=self.kwargs['sigma'],
                T=self.kwargs.get('T', 1.0),
                n_steps=self.kwargs.get('n_steps', 252),
                n_simulations=self.kwargs.get('n_simulations', 1000),
            )

        elif name == 'cir':
            engine = InterestRateModel()
            return engine.cir_model(
                r0=self.kwargs['r0'], a=self.kwargs['a'],
                b=self.kwargs['b'], sigma=self.kwargs['sigma'],
                T=self.kwargs.get('T', 1.0),
                n_steps=self.kwargs.get('n_steps', 252),
                n_simulations=self.kwargs.get('n_simulations', 1000),
            )

        elif name == 'hull_white':
            engine = InterestRateModel()
            return engine.hull_white(
                r0=self.kwargs['r0'], a=self.kwargs['a'],
                sigma=self.kwargs['sigma'],
                T=self.kwargs.get('T', 1.0),
                n_steps=self.kwargs.get('n_steps', 252),
                n_simulations=self.kwargs.get('n_simulations', 1000),
            )

        elif name == 'duration_convexity':
            engine = InterestRateModel()
            return engine.duration_convexity(
                cashflows=self.kwargs['cashflows'],
                rates=self.kwargs['rates'],
                ytm=self.kwargs['ytm'],
            )

        else:
            return {"error": f"Unknown task: {name}"}


# ──────────────────────────────────────────────────────────────
# Helper: create a styled spinbox
# ──────────────────────────────────────────────────────────────
def _dspin(val=0.0, lo=-1e9, hi=1e9, step=0.01, decimals=4, suffix=""):
    sb = QDoubleSpinBox()
    sb.setRange(lo, hi)
    sb.setSingleStep(step)
    sb.setDecimals(decimals)
    sb.setValue(val)
    if suffix:
        sb.setSuffix(suffix)
    return sb


def _ispin(val=0, lo=0, hi=99999, step=1):
    sb = QSpinBox()
    sb.setRange(lo, hi)
    sb.setSingleStep(step)
    sb.setValue(val)
    return sb


def _combo(items, current=0):
    cb = QComboBox()
    cb.addItems(items)
    cb.setCurrentIndex(current)
    return cb


def _label(text, bold=False, color=None, size=12):
    lbl = QLabel(text)
    f = lbl.font()
    f.setPointSize(size)
    if bold:
        f.setBold(True)
    lbl.setFont(f)
    if color:
        lbl.setStyleSheet(f"color: {color};")
    return lbl


def _button(text, obj_name=None):
    btn = QPushButton(text)
    if obj_name:
        btn.setObjectName(obj_name)
    return btn


def make_table(headers, rows=0, cols=0):
    """Create a styled QTableWidget."""
    ncols = len(headers) if headers else cols
    nrows = rows if rows > 0 else 0
    table = QTableWidget(nrows, ncols)
    table.setHorizontalHeaderLabels(headers)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.verticalHeader().setVisible(False)
    table.setMaximumHeight(250)
    return table


def populate_table(table, data, headers=None):
    """Populate table from list-of-dicts or list-of-lists."""
    if not data:
        table.setRowCount(1)
        table.setColumnCount(1)
        table.setHorizontalHeaderLabels([""])
        item = QTableWidgetItem("\u062fاده‌ای برای نمایش وجود ندارد")
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        table.setItem(0, 0, item)
        return
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        headers = headers or list(data[0].keys())
        table.setColumnCount(len(headers))
        table.setRowCount(len(data))
        table.setHorizontalHeaderLabels([str(h) for h in headers])
        for i, row in enumerate(data):
            for j, h in enumerate(headers):
                val = row.get(h, "")
                if isinstance(val, float):
                    txt = f"{val:.6f}"
                elif isinstance(val, (list, np.ndarray)):
                    arr = np.asarray(val)
                    txt = f"[{arr.min():.4f}, ..., {arr.max():.4f}] (n={len(arr)})"
                else:
                    txt = str(val)
                item = QTableWidgetItem(txt)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                table.setItem(i, j, item)
    elif isinstance(data, dict):
        table.setRowCount(len(data))
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["\u067eارامتر", "\u0645قدار"])
        for i, (k, v) in enumerate(data.items()):
            kitem = QTableWidgetItem(str(k))
            kitem.setTextAlignment(Qt.AlignmentFlag.AlignRight)
            if isinstance(v, float):
                vtxt = f"{v:.6f}"
            elif isinstance(v, np.ndarray):
                vtxt = f"array(shape={v.shape})"
            elif isinstance(v, (list, tuple)) and len(v) > 5:
                vtxt = f"[{v[0]:.4f}, ...] (n={len(v)})"
            else:
                vtxt = str(v)
            vitem = QTableWidgetItem(vtxt)
            vitem.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            table.setItem(i, 0, kitem)
            table.setItem(i, 1, vitem)
    table.resizeColumnsToContents()


def make_figure(figsize=(6, 4)):
    """Create a matplotlib Figure embedded in a FigureCanvas."""
    fig = Figure(figsize=figsize, facecolor=BG_COLOR)
    canvas = FigureCanvas(fig)
    canvas.setMinimumHeight(300)
    return fig, canvas


def style_axes(ax):
    """Apply dark theme to matplotlib axes."""
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=MUTED_TEXT, labelsize=9)
    ax.xaxis.label.set_color(MUTED_TEXT)
    ax.yaxis.label.set_color(MUTED_TEXT)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(True, alpha=0.15, color=GRID_COLOR)


# ──────────────────────────────────────────────────────────────
# Base class for each tab
# ──────────────────────────────────────────────────────────────
class BaseTab(QWidget):
    """Base widget providing common layout for all 9 tabs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(16, 16, 16, 16)
        self.main_layout.setSpacing(12)

        # Header label
        self.header = _label("", bold=True, color=ACCENT_COLOR, size=16)
        self.main_layout.addWidget(self.header)

        # Top section: inputs + buttons in a splitter
        self.input_group = QGroupBox(self._group_title())
        self.input_form = QFormLayout()
        self.input_form.setSpacing(8)
        self.input_group.setLayout(self.input_form)
        self.main_layout.addWidget(self.input_group)

        # Buttons row
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)
        self.demo_btn = _button("\U0001f3a8 Demo", "demoBtn")
        self.exec_btn = _button("\u25b6 Execute")
        btn_layout.addWidget(self.demo_btn)
        btn_layout.addWidget(self.exec_btn)
        btn_layout.addStretch()
        self.status_label = QLabel("")
        self.status_label.setStyleSheet(f"color: {MUTED_TEXT}; font-style: italic;")
        btn_layout.addWidget(self.status_label)
        self.main_layout.addLayout(btn_layout)

        # Results splitter: table on top, chart on bottom
        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.results_table = make_table([])
        self.results_table.setMinimumHeight(120)
        self.splitter.addWidget(self.results_table)
        self.fig, self.canvas = make_figure()
        self.splitter.addWidget(self.canvas)
        self.splitter.setSizes([200, 350])
        self.main_layout.addWidget(self.splitter)

        # Connect buttons
        self.demo_btn.clicked.connect(self.run_demo)
        self.exec_btn.clicked.connect(self.run_execute)

        self._worker = None
        self._build_inputs()

    def _group_title(self):
        return "\u062aنظیمات \u0648رود\u06cc"

    def _build_inputs(self):
        """Override in subclasses to add input controls to self.input_form."""
        pass

    def _get_params(self):
        """Override in subclasses to collect user inputs as a dict."""
        return {}

    def _demo_params(self):
        """Override in subclasses to return demo parameter dict."""
        return {}

    def _run_task(self, task_name, params):
        """Launch a background worker."""
        self.status_label.setText("\u23f3 \u062fر \u062dا\u0644 \u0627\u062c\u0631\u0627...")
        self.status_label.setStyleSheet(f"color: {ACCENT_COLOR}; font-style: italic;")
        self.demo_btn.setEnabled(False)
        self.exec_btn.setEnabled(False)
        self._worker = QuantWorker(task_name, **params)
        self._worker.finished.connect(self._on_result)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_result(self, result):
        self.status_label.setText("\u2705 \u062a\u06a9\u0645\u06cc\u0644 \u0634\u062f")
        self.status_label.setStyleSheet(f"color: {SUCCESS_COLOR}; font-weight: bold;")
        self.demo_btn.setEnabled(True)
        self.exec_btn.setEnabled(True)
        self._display_result(result)

    def _on_error(self, msg):
        self.status_label.setText(f"\u274c \u062e\u0637\u0627: {msg}")
        self.status_label.setStyleSheet(f"color: {ERROR_COLOR}; font-weight: bold;")
        self.demo_btn.setEnabled(True)
        self.exec_btn.setEnabled(True)

    def run_demo(self):
        params = self._demo_params()
        task = self._task_name(params)
        if task and params:
            self._run_task(task, params)

    def run_execute(self):
        params = self._get_params()
        task = self._task_name(params)
        if task and params:
            self._run_task(task, params)

    def _task_name(self, params):
        """Override to return the QuantWorker task name."""
        return ""

    def _display_result(self, result):
        """Override to display results in table and chart."""
        if "error" in result:
            self._on_error(result["error"])
            return
        populate_table(self.results_table, result)



# ═══════════════════════════════════════════════════════════════
# Tab 1: Portfolio Optimization  —  بهینه‌سازی پرتفوی
# ═══════════════════════════════════════════════════════════════
class PortfolioTab(BaseTab):
    def __init__(self, parent=None):
        self.sp_n_assets = _ispin(5, 2, 20)
        self.sp_risk_free = _dspin(0.02, 0, 1, 0.005, 4)
        self.sp_target_ret = _dspin(0.10, -0.5, 2, 0.01, 4)
        self.cb_method = _combo(["Markowitz MVO", "Max Sharpe", "Efficient Frontier"], 0)
        self.cb_short = _combo(["بدون Short Selling", "با Short Selling"], 0)
        super().__init__(parent)
        self.header.setText("✨ بهینه‌سازی پرتفوی")
        self._last_frontier = None

    def _group_title(self):
        return "تنظیمات بهینه‌سازی پرتفوی"

    def _build_inputs(self):
        self.input_form.addRow(_label("تعداد دارایی‌ها:"), self.sp_n_assets)
        self.input_form.addRow(_label("نرخ بدون ریسک:"), self.sp_risk_free)
        self.input_form.addRow(_label("بازده هدف:"), self.sp_target_ret)
        self.input_form.addRow(_label("روش بهینه‌سازی:"), self.cb_method)
        self.input_form.addRow(_label("محدودیت فروش استقراضی:"), self.cb_short)

    def _get_params(self):
        n = self.sp_n_assets.value()
        np.random.seed(42)
        mu = np.random.uniform(0.05, 0.25, n)
        A = np.random.randn(n, n) * 0.1
        cov = A @ A.T + np.eye(n) * 0.01
        short = self.cb_short.currentIndex() == 1
        return {
            "expected_returns": mu.tolist(),
            "cov_matrix": cov.tolist(),
            "risk_free_rate": self.sp_risk_free.value(),
            "target_return": self.sp_target_ret.value(),
            "short_selling": short,
            "_n_assets": n,
        }

    def _demo_params(self):
        np.random.seed(123)
        n = 5
        mu = np.array([0.12, 0.10, 0.15, 0.08, 0.20])
        A = np.array([
            [0.04, 0.006, 0.002, -0.001, 0.003],
            [0.006, 0.09, 0.005, 0.002, -0.001],
            [0.002, 0.005, 0.0625, 0.004, 0.007],
            [-0.001, 0.002, 0.004, 0.036, 0.001],
            [0.003, -0.001, 0.007, 0.001, 0.16],
        ])
        cov = A @ A.T + np.eye(n) * 0.001
        return {
            "expected_returns": mu.tolist(),
            "cov_matrix": cov.tolist(),
            "risk_free_rate": 0.02,
            "target_return": 0.12,
            "short_selling": False,
            "_n_assets": n,
        }

    def _task_name(self, params):
        idx = self.cb_method.currentIndex()
        return ["markowitz", "max_sharpe", "efficient_frontier"][idx]

    def _display_result(self, result):
        if "error" in result:
            self._on_error(result["error"])
            return
        self.fig.clear()
        idx = self.cb_method.currentIndex()
        if idx == 0:
            # Markowitz
            w = result.get("weights", [])
            assets = [f"دارایی {i+1}" for i in range(len(w))]
            table_data = [{"دارایی": assets[i], "وزن": w[i]} for i in range(len(w))]
            table_data.append({"دارایی": "بازده پرتفوی", "وزن": result.get("portfolio_return", 0)})
            table_data.append({"دارایی": "ریسک پرتفوی", "وزن": result.get("portfolio_risk", 0)})
            table_data.append({"دارایی": "نسبت شارپ", "وزن": result.get("sharpe_ratio", 0)})
            populate_table(self.results_table, table_data)
            # Pie chart
            ax = self.fig.add_subplot(111)
            colors = plt.cm.Set2(np.linspace(0, 1, len(w)))
            ax.pie(w, labels=assets, autopct='%1.1f%%', colors=colors,
                   textprops={"color": TEXT_COLOR, "fontsize": 10})
            ax.set_title("تخصیص وزن پرتفوی", fontsize=13, color=TEXT_COLOR)
        elif idx == 1:
            # Max Sharpe
            w = result.get("weights", [])
            assets = [f"دارایی {i+1}" for i in range(len(w))]
            table_data = [{"دارایی": assets[i], "وزن": w[i]} for i in range(len(w))]
            table_data.append({"دارایی": "بازده", "وزن": result.get("portfolio_return", 0)})
            table_data.append({"دارایی": "ریسک", "وزن": result.get("portfolio_risk", 0)})
            table_data.append({"دارایی": "شارپ", "وزن": result.get("sharpe_ratio", 0)})
            populate_table(self.results_table, table_data)
            ax = self.fig.add_subplot(111)
            colors = plt.cm.Set3(np.linspace(0, 1, len(w)))
            bars = ax.bar(assets, w, color=colors, edgecolor=GRID_COLOR)
            ax.set_ylabel("وزن")
            ax.set_title("تخصیص بهینه (حداکثر شارپ)", fontsize=13, color=TEXT_COLOR)
            style_axes(ax)
            for bar, val in zip(bars, w):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", color=TEXT_COLOR, fontsize=9)
        elif idx == 2:
            # Efficient Frontier
            frontiers = result.get("frontiers", [])
            if frontiers:
                risks = [f["risk"] for f in frontiers if "risk" in f]
                rets = [f["return"] for f in frontiers if "return" in f]
                populate_table(self.results_table, frontiers[:20])
                ax = self.fig.add_subplot(111)
                ax.plot(risks, rets, '-', color=ACCENT_COLOR, linewidth=2, label="مرز کارا")
                ax.scatter(risks, rets, c=PURPLE_COLOR, s=20, alpha=0.7)
                ax.set_xlabel("ریسک (انحراف معیار)")
                ax.set_ylabel("بازده مورد انتظار")
                ax.set_title("مرز کارا (Efficient Frontier)", fontsize=13, color=TEXT_COLOR)
                ax.legend(facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
                style_axes(ax)
            else:
                populate_table(self.results_table, result)
        self.fig.tight_layout()
        self.canvas.draw()


# ═══════════════════════════════════════════════════════════════
# Tab 2: Derivatives Pricing  —  قیمت‌گذاری مشتقات
# ═══════════════════════════════════════════════════════════════
class DerivativesTab(BaseTab):
    def __init__(self, parent=None):
        self.sp_S = _dspin(100, 0.01, 1e6, 1, 2)
        self.sp_K = _dspin(100, 0.01, 1e6, 1, 2)
        self.sp_T = _dspin(1.0, 0.01, 30, 0.1, 4, " سال")
        self.sp_r = _dspin(0.05, 0, 1, 0.005, 4)
        self.sp_sigma = _dspin(0.20, 0.001, 5, 0.01, 4)
        self.cb_type = _combo(["Call", "Put"], 0)
        self.cb_method = _combo(["Black-Scholes", "Binomial Tree", "Monte Carlo", "Implied Volatility"], 0)
        self.sp_n_steps = _ispin(200, 10, 10000, 50)
        self.sp_n_sim = _ispin(10000, 100, 1000000, 1000)
        self.sp_market_price = _dspin(10.0, 0.01, 1e6, 0.5, 2)
        super().__init__(parent)
        self.header.setText("⚙️ قیمت‌گذاری مشتقات")

    def _group_title(self):
        return "تنظیمات قیمت‌گذاری اختیار معامله"

    def _build_inputs(self):
        self.input_form.addRow(_label("قیمت پایه (S):"), self.sp_S)
        self.input_form.addRow(_label("قیمت اعمال (K):"), self.sp_K)
        self.input_form.addRow(_label("زمان تا سررسید (T):"), self.sp_T)
        self.input_form.addRow(_label("نرخ بهره بدون ریسک (r):"), self.sp_r)
        self.input_form.addRow(_label("نوسان‌پذیری (σ):"), self.sp_sigma)
        self.input_form.addRow(_label("نوع اختیار:"), self.cb_type)
        self.input_form.addRow(_label("روش قیمت‌گذاری:"), self.cb_method)
        self.input_form.addRow(_label("تعداد مراحل (درخت دوجمله‌ای):"), self.sp_n_steps)
        self.input_form.addRow(_label("تعداد شبیه‌سازی (MC):"), self.sp_n_sim)
        self.input_form.addRow(_label("قیمت بازار (برای IV):"), self.sp_market_price)

    def _demo_params(self):
        return {
            "S": 100, "K": 105, "T": 0.5, "r": 0.05,
            "sigma": 0.25, "option_type": "call",
            "n_steps": 200, "n_simulations": 10000,
            "market_price": 8.5,
        }

    def _get_params(self):
        return {
            "S": self.sp_S.value(), "K": self.sp_K.value(),
            "T": self.sp_T.value(), "r": self.sp_r.value(),
            "sigma": self.sp_sigma.value(),
            "option_type": self.cb_type.currentText().lower(),
            "n_steps": self.sp_n_steps.value(),
            "n_simulations": self.sp_n_sim.value(),
            "market_price": self.sp_market_price.value(),
        }

    def _task_name(self, params):
        idx = self.cb_method.currentIndex()
        return ["black_scholes", "binomial_tree", "mc_pricing", "implied_vol"][idx]

    def _display_result(self, result):
        if "error" in result:
            self._on_error(result["error"])
            return
        self.fig.clear()
        # Always show key metrics in table
        display = {}
        skip_keys = {"method", "option_type", "spot", "strike", "time_to_maturity",
                      "risk_free_rate", "volatility"}
        for k, v in result.items():
            if k not in skip_keys and not isinstance(v, (list, np.ndarray)):
                display[k] = v
        populate_table(self.results_table, display)

        idx = self.cb_method.currentIndex()
        if idx in (0, 1, 2):  # BS, Binomial, MC
            ax1 = self.fig.add_subplot(121)
            ax2 = self.fig.add_subplot(122)
            # Payoff diagram
            S_range = np.linspace(50, 150, 200)
            K = result.get("strike", self.sp_K.value())
            opt_type = result.get("option_type", "call")
            price = result.get("price", 0)
            if opt_type == "call":
                payoff = np.maximum(S_range - K, 0) - price
            else:
                payoff = np.maximum(K - S_range, 0) - price
            ax1.plot(S_range, payoff, color=SUCCESS_COLOR if payoff[-1] > 0 else ERROR_COLOR, linewidth=2)
            ax1.axhline(y=0, color=MUTED_TEXT, linestyle="--", alpha=0.5)
            ax1.axvline(x=K, color=PURPLE_COLOR, linestyle="--", alpha=0.5, label=f"K={K}")
            ax1.set_xlabel("قیمت پایه در سررسید")
            ax1.set_ylabel("سود/زیان")
            ax1.set_title(f"نمودار سود/زیان ({opt_type})")
            ax1.legend(facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
            style_axes(ax1)
            # Greeks bar chart
            greeks = {"Delta": result.get("delta", 0), "Gamma": result.get("gamma", 0),
                      "Theta": result.get("theta", 0), "Vega": result.get("vega", 0),
                      "Rho": result.get("rho", 0)}
            names = list(greeks.keys())
            vals = list(greeks.values())
            colors_g = [ACCENT_COLOR, PURPLE_COLOR, ERROR_COLOR, SUCCESS_COLOR, MUTED_TEXT]
            ax2.bar(names, vals, color=colors_g, edgecolor=GRID_COLOR)
            ax2.set_title("حساسیت‌ها (Greeks)")
            ax2.axhline(y=0, color=MUTED_TEXT, linestyle="--", alpha=0.3)
            style_axes(ax2)
            for bar, val in zip(ax2.patches, vals):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001 * np.sign(bar.get_height()),
                        f"{val:.4f}", ha="center", va="bottom" if val >= 0 else "top",
                        color=TEXT_COLOR, fontsize=9)
        elif idx == 3:  # Implied Vol
            ax = self.fig.add_subplot(111)
            iv = result.get("implied_volatility", result.get("iv", 0))
            ax.bar(["IV"], [iv], color=PURPLE_COLOR, edgecolor=GRID_COLOR, width=0.3)
            ax.set_ylabel("نوسان‌پذیری ضمنی")
            ax.set_title(f"نوسان‌پذیری ضمنی: {iv:.4f}", fontsize=13, color=TEXT_COLOR)
            style_axes(ax)
        self.fig.tight_layout()
        self.canvas.draw()


# ═══════════════════════════════════════════════════════════════
# Tab 3: Risk Analysis  —  تحلیل ریسک
# ═══════════════════════════════════════════════════════════════
class RiskTab(BaseTab):
    def __init__(self, parent=None):
        self.sp_n_obs = _ispin(500, 50, 10000, 100)
        self.sp_confidence = _dspin(0.95, 0.50, 0.999, 0.01, 3)
        self.sp_portfolio_val = _dspin(1e6, 1000, 1e12, 1e5, 0)
        self.cb_method = _combo(["VaR تاریخی", "VaR پارامتریک", "CVaR", "GARCH", "تست استرس"], 0)
        self.sp_horizon = _ispin(10, 1, 100, 5)
        self._returns_cache = None
        super().__init__(parent)
        self.header.setText("⚠️ تحلیل ریسک")

    def _group_title(self):
        return "تنظیمات تحلیل ریسک"

    def _build_inputs(self):
        self.input_form.addRow(_label("تعداد مشاهدات:"), self.sp_n_obs)
        self.input_form.addRow(_label("سطح اطمینان:"), self.sp_confidence)
        self.input_form.addRow(_label("ارزش پرتفوی:"), self.sp_portfolio_val)
        self.input_form.addRow(_label("روش تحلیل:"), self.cb_method)
        self.input_form.addRow(_label("افق پیش‌بینی (GARCH):"), self.sp_horizon)

    def _generate_returns(self, n, seed=42):
        np.random.seed(seed)
        r = np.random.normal(0.0008, 0.015, n)
        r += 0.003 * np.sin(np.arange(n) / 50)
        r -= 0.01 * (np.random.random(n) < 0.02) * np.abs(np.random.randn(n))
        return r.tolist()

    def _demo_params(self):
        returns = self._generate_returns(500, seed=42)
        self._returns_cache = returns
        idx = self.cb_method.currentIndex()
        base = {
            "returns": returns,
            "confidence": 0.95,
            "portfolio_value": 1e6,
        }
        if idx == 3:
            base["horizon"] = 10
        return base

    def _get_params(self):
        n = self.sp_n_obs.value()
        returns = self._generate_returns(n, seed=None)
        self._returns_cache = returns
        idx = self.cb_method.currentIndex()
        base = {
            "returns": returns,
            "confidence": self.sp_confidence.value(),
            "portfolio_value": self.sp_portfolio_val.value(),
        }
        if idx == 3:
            base["horizon"] = self.sp_horizon.value()
        if idx == 0:
            base["method"] = "historical"
        elif idx == 1:
            base["method"] = "parametric"
        return base

    def _task_name(self, params):
        idx = self.cb_method.currentIndex()
        return ["var", "var", "cvar", "garch", "stress_test"][idx]

    def _display_result(self, result):
        if "error" in result:
            self._on_error(result["error"])
            return
        self.fig.clear()
        display = {k: v for k, v in result.items()
                   if not isinstance(v, (list, np.ndarray)) and k != "fitted_volatility"}
        populate_table(self.results_table, display)

        idx = self.cb_method.currentIndex()
        returns = np.array(self._returns_cache or [])
        if idx in (0, 1, 2):  # VaR, CVaR
            ax = self.fig.add_subplot(111)
            ax.hist(returns, bins=60, density=True, alpha=0.7, color=ACCENT_COLOR, edgecolor=GRID_COLOR)
            var_val = result.get("var", result.get("value_at_risk", 0))
            if isinstance(var_val, (int, float)):
                ax.axvline(x=var_val, color=ERROR_COLOR, linewidth=2, linestyle="--",
                          label=f"VaR = {var_val:.4f}")
            cvar = result.get("cvar", result.get("conditional_var", None))
            if isinstance(cvar, (int, float)):
                ax.axvline(x=cvar, color=PURPLE_COLOR, linewidth=2, linestyle=":",
                          label=f"CVaR = {cvar:.4f}")
            ax.set_xlabel("بازده")
            ax.set_ylabel("چگالی")
            ax.set_title("توزیع بازده و سطوح ریسک", fontsize=13, color=TEXT_COLOR)
            ax.legend(facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
            style_axes(ax)
        elif idx == 3:  # GARCH
            fitted_vol = result.get("fitted_volatility")
            if fitted_vol is not None and len(fitted_vol) > 0:
                ax = self.fig.add_subplot(111)
                v = np.array(fitted_vol)
                t = np.arange(len(v))
                ax.plot(t, v, color=ACCENT_COLOR, linewidth=1)
                ax.fill_between(t, 0, v, alpha=0.2, color=ACCENT_COLOR)
                ax.set_xlabel("زمان")
                ax.set_ylabel("نوسان‌پذیری شرطی")
                ax.set_title("نوسان‌پذیری شرطی GARCH(1,1)", fontsize=13, color=TEXT_COLOR)
                style_axes(ax)
            elif "conditional_volatility" in result:
                ax = self.fig.add_subplot(111)
                v = np.array(result["conditional_volatility"])
                ax.plot(v, color=ACCENT_COLOR, linewidth=1)
                ax.fill_between(range(len(v)), 0, v, alpha=0.2, color=ACCENT_COLOR)
                ax.set_title("نوسان‌پذیری شرطی GARCH(1,1)", fontsize=13, color=TEXT_COLOR)
                style_axes(ax)
        elif idx == 4:  # Stress test
            scenarios = result.get("scenarios", [])
            if scenarios:
                names_s = [s.get("name", f"سناریو {i}") for i, s in enumerate(scenarios)]
                losses = [s.get("loss", s.get("portfolio_loss", 0)) for s in scenarios]
                colors_s = [ERROR_COLOR if l > 0 else SUCCESS_COLOR for l in losses]
                ax = self.fig.add_subplot(111)
                bars = ax.barh(names_s, losses, color=colors_s, edgecolor=GRID_COLOR)
                ax.set_xlabel("زیان")
                ax.set_title("نتایج تست استرس", fontsize=13, color=TEXT_COLOR)
                style_axes(ax)
                for bar, val in zip(bars, losses):
                    ax.text(bar.get_width() + max(losses)*0.01, bar.get_y() + bar.get_height()/2,
                            f"{val:,.0f}", va="center", color=TEXT_COLOR, fontsize=9)
        self.fig.tight_layout()
        self.canvas.draw()


# ═══════════════════════════════════════════════════════════════
# Tab 4: Time Series Analysis  —  تحلیل سری زمانی
# ═══════════════════════════════════════════════════════════════
class TimeSeriesTab(BaseTab):
    def __init__(self, parent=None):
        self.sp_n_points = _ispin(500, 50, 5000, 100)
        self.sp_forecast = _ispin(20, 5, 200, 5)
        self.cb_analysis = _combo(["تحلیل کامل", "ADF Test", "آمار متحرک", "تحلیل بازده"], 0)
        self._ts_cache = None
        super().__init__(parent)
        self.header.setText("📈 تحلیل سری زمانی")

    def _group_title(self):
        return "تنظیمات تحلیل سری زمانی"

    def _build_inputs(self):
        self.input_form.addRow(_label("تعداد نقاط:"), self.sp_n_points)
        self.input_form.addRow(_label("افق پیش‌بینی:"), self.sp_forecast)
        self.input_form.addRow(_label("نوع تحلیل:"), self.cb_analysis)

    def _generate_ts(self, n, seed=42):
        np.random.seed(seed)
        t = np.arange(n)
        y = 100 + 0.3 * t + 5 * np.sin(2 * np.pi * t / 50) + np.random.randn(n) * 2
        # add some AR structure
        for i in range(2, n):
            y[i] += 0.4 * y[i-1] - 0.2 * y[i-2]
        return y.tolist()

    def _demo_params(self):
        y = self._generate_ts(500, seed=42)
        self._ts_cache = y
        return {"y": y, "forecast_steps": 20}

    def _get_params(self):
        n = self.sp_n_points.value()
        y = self._generate_ts(n, seed=None)
        self._ts_cache = y
        return {"y": y, "forecast_steps": self.sp_forecast.value()}

    def _task_name(self, params):
        idx = self.cb_analysis.currentIndex()
        if idx == 0:
            return "ts_full"
        # For other methods, we still use full analysis as the worker
        return "ts_full"

    def _display_result(self, result):
        if "error" in result:
            self._on_error(result["error"])
            return
        self.fig.clear()
        display = {k: v for k, v in result.items()
                   if not isinstance(v, (list, np.ndarray)) or (isinstance(v, list) and len(v) < 8)}
        populate_table(self.results_table, display)

        y = np.array(self._ts_cache or [])
        ax1 = self.fig.add_subplot(211)
        ax1.plot(y, color=ACCENT_COLOR, linewidth=0.8, label="داده اصلی")
        ax1.set_title("سری زمانی", fontsize=12, color=TEXT_COLOR)
        ax1.legend(facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
        style_axes(ax1)

        # Show forecast if available
        forecast = result.get("forecast")
        if forecast is not None and len(forecast) > 0:
            fc = np.array(forecast)
            ax2 = self.fig.add_subplot(212)
            last_n = min(100, len(y))
            ax2.plot(range(len(y) - last_n, len(y)), y[-last_n:], color=ACCENT_COLOR, linewidth=1, label="تاریخچه")
            fc_range = range(len(y), len(y) + len(fc))
            ax2.plot(fc_range, fc, color=PURPLE_COLOR, linewidth=2, linestyle="--", label="پیش‌بینی")
            ax2.fill_between(fc_range, fc * 0.97, fc * 1.03, alpha=0.15, color=PURPLE_COLOR, label="باند اطمینان")
            ax2.set_title("پیش‌بینی ARIMA", fontsize=12, color=TEXT_COLOR)
            ax2.legend(facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
            style_axes(ax2)
        else:
            ax2 = self.fig.add_subplot(212)
            if len(y) > 20:
                window = min(20, len(y) // 5)
                rolling_mean = np.convolve(y, np.ones(window)/window, mode='valid')
                ax2.plot(rolling_mean, color=SUCCESS_COLOR, linewidth=1.5, label=f"میانگین متحرک ({window})")
                ax2.plot(y, color=ACCENT_COLOR, alpha=0.3, linewidth=0.5, label="داده")
                ax2.set_title("میانگین متحرک", fontsize=12, color=TEXT_COLOR)
                ax2.legend(facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
                style_axes(ax2)
        self.fig.tight_layout()
        self.canvas.draw()


# ═══════════════════════════════════════════════════════════════
# Tab 5: Fuzzy Systems  —  سیستم‌های فازی
# ═══════════════════════════════════════════════════════════════
class FuzzyTab(BaseTab):
    def __init__(self, parent=None):
        self.sp_income = _dspin(8000, 0, 1e6, 500, 0, " $")
        self.sp_debt_ratio = _dspin(0.35, 0, 2, 0.05, 3)
        self.sp_credit_hist = _dspin(10, 0, 30, 1, 1, " سال")
        self.sp_employ_years = _dspin(5, 0, 40, 1, 1, " سال")
        self.sp_n_accounts = _ispin(5, 0, 50, 1)
        self._sensitivity_data = None
        super().__init__(parent)
        self.header.setText("🌀 سیستم‌های فازی")

    def _group_title(self):
        return "تنظیمات امتیازدهی اعتباری فازی"

    def _build_inputs(self):
        self.input_form.addRow(_label("درآمد ماهانه:"), self.sp_income)
        self.input_form.addRow(_label("نسبت بدهی:"), self.sp_debt_ratio)
        self.input_form.addRow(_label("سوابق اعتباری:"), self.sp_credit_hist)
        self.input_form.addRow(_label("سابقه اشتغال:"), self.sp_employ_years)
        self.input_form.addRow(_label("تعداد حساب‌ها:"), self.sp_n_accounts)

    def _demo_params(self):
        return {
            "income": 12000,
            "debt_ratio": 0.30,
            "credit_history_years": 15,
            "employment_years": 8,
            "num_accounts": 6,
        }

    def _get_params(self):
        return {
            "income": self.sp_income.value(),
            "debt_ratio": self.sp_debt_ratio.value(),
            "credit_history_years": self.sp_credit_hist.value(),
            "employment_years": self.sp_employ_years.value(),
            "num_accounts": self.sp_n_accounts.value(),
        }

    def _task_name(self, params):
        return "fuzzy_credit"

    def _display_result(self, result):
        if "error" in result:
            self._on_error(result["error"])
            return
        self.fig.clear()
        display = {k: v for k, v in result.items()
                   if not isinstance(v, (list, np.ndarray, dict))}
        populate_table(self.results_table, display)

        score = result.get("score", result.get("credit_score", 0))
        risk_level = result.get("risk_level", result.get("classification", "نامشخص"))

        ax1 = self.fig.add_subplot(121)
        # Gauge-like visualization
        categories = ["خیلی ضعیف", "ضعیف", "متوسط", "خوب", "عالی"]
        positions = np.linspace(0, 100, 5)
        colors_gauge = [ERROR_COLOR, ERROR_COLOR, MUTED_TEXT, SUCCESS_COLOR, SUCCESS_COLOR]
        ax1.barh(categories, [100]*5, color=["#2a2a3a"]*5, edgecolor=GRID_COLOR)
        ax1.barh(categories, [20]*5, left=positions - 10, color=colors_gauge, alpha=0.3)
        ax1.axvline(x=score, color=ACCENT_COLOR, linewidth=3, label=f"امتیاز: {score:.1f}")
        ax1.set_xlabel("امتیاز")
        ax1.set_title(f"امتیاز اعتباری فازی: {risk_level}", fontsize=12, color=TEXT_COLOR)
        ax1.legend(facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
        style_axes(ax1)

        # Membership functions visualization
        ax2 = self.fig.add_subplot(122)
        x = np.linspace(0, 1, 200)
        # Plot some example membership functions
        ax2.plot(x, np.maximum(0, np.minimum(x / 0.3, (1 - x) / 0.4)), color=ACCENT_COLOR, label="نسبت بدهی")
        ax2.plot(x, np.maximum(0, np.minimum(x / 0.2, (0.8 - x) / 0.5)), color=SUCCESS_COLOR, label="درآمد")
        ax2.plot(x, np.maximum(0, np.minimum((x - 0.3) / 0.3, (0.9 - x) / 0.3)), color=PURPLE_COLOR, label="سوابق")
        ax2.set_xlabel("درجه عضویت")
        ax2.set_ylabel("نرمال‌شده")
        ax2.set_title("توابع عضویت فازی", fontsize=12, color=TEXT_COLOR)
        ax2.legend(facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=9)
        style_axes(ax2)
        self.fig.tight_layout()
        self.canvas.draw()


# ═══════════════════════════════════════════════════════════════
# Tab 6: Network Analysis  —  تحلیل شبکه
# ═══════════════════════════════════════════════════════════════
class NetworkTab(BaseTab):
    def __init__(self, parent=None):
        self.sp_n_assets = _ispin(10, 3, 50, 1)
        self.sp_threshold = _dspin(0.3, 0, 1, 0.05, 2)
        self.cb_analysis = _combo(["شبکه همبستگی", "شبیه‌سازی سرایت"], 0)
        self.sp_shock_mag = _dspin(0.1, 0.01, 1, 0.01, 3)
        self.sp_recovery = _dspin(0.05, 0, 1, 0.01, 3)
        self.sp_max_rounds = _ispin(50, 5, 200, 5)
        self._net_cache = None
        super().__init__(parent)
        self.header.setText("🕸️ تحلیل شبکه")

    def _group_title(self):
        return "تنظیمات تحلیل شبکه"

    def _build_inputs(self):
        self.input_form.addRow(_label("تعداد دارایی‌ها:"), self.sp_n_assets)
        self.input_form.addRow(_label("آستانه همبستگی:"), self.sp_threshold)
        self.input_form.addRow(_label("نوع تحلیل:"), self.cb_analysis)
        self.input_form.addRow(_label("شدت شوک (سرایت):"), self.sp_shock_mag)
        self.input_form.addRow(_label("نرخ بازیابی:"), self.sp_recovery)
        self.input_form.addRow(_label("حداکثر دورها:"), self.sp_max_rounds)

    def _generate_correlation_matrix(self, n, seed=42):
        np.random.seed(seed)
        A = np.random.randn(n, n) * 0.5
        cov = A @ A.T
        d = np.sqrt(np.diag(cov))
        corr = cov / np.outer(d, d + 1e-10)
        corr = (corr + corr.T) / 2
        np.fill_diagonal(corr, 1.0)
        return corr

    def _demo_params(self):
        n = 10
        corr = self._generate_correlation_matrix(n, seed=42)
        returns = np.random.multivariate_normal(np.zeros(n), corr * 0.01**2, 252)
        self._net_cache = {"corr": corr, "returns": returns}
        idx = self.cb_analysis.currentIndex()
        if idx == 0:
            return {
                "returns_matrix": returns.tolist(),
                "threshold": 0.3,
                "asset_names": [f"A{i+1}" for i in range(n)],
            }
        else:
            adj = (np.abs(corr) > 0.3).astype(float)
            np.fill_diagonal(adj, 0)
            return {
                "initial_shock_node": 0,
                "adjacency": adj.tolist(),
                "shock_magnitude": 0.1,
                "recovery_rate": 0.05,
                "max_rounds": 50,
            }

    def _get_params(self):
        n = self.sp_n_assets.value()
        corr = self._generate_correlation_matrix(n, seed=None)
        returns = np.random.multivariate_normal(np.zeros(n), corr * 0.01**2, 252)
        self._net_cache = {"corr": corr, "returns": returns}
        idx = self.cb_analysis.currentIndex()
        if idx == 0:
            return {
                "returns_matrix": returns.tolist(),
                "threshold": self.sp_threshold.value(),
                "asset_names": [f"A{i+1}" for i in range(n)],
            }
        else:
            adj = (np.abs(corr) > self.sp_threshold.value()).astype(float)
            np.fill_diagonal(adj, 0)
            return {
                "initial_shock_node": 0,
                "adjacency": adj.tolist(),
                "shock_magnitude": self.sp_shock_mag.value(),
                "recovery_rate": self.sp_recovery.value(),
                "max_rounds": self.sp_max_rounds.value(),
            }

    def _task_name(self, params):
        idx = self.cb_analysis.currentIndex()
        return ["network_correlation", "contagion"][idx]

    def _display_result(self, result):
        if "error" in result:
            self._on_error(result["error"])
            return
        self.fig.clear()
        display = {k: v for k, v in result.items()
                   if not isinstance(v, (list, np.ndarray))}
        populate_table(self.results_table, display)

        idx = self.cb_analysis.currentIndex()
        if idx == 0:
            # Correlation network heatmap
            cache = self._net_cache or {}
            corr = cache.get("corr")
            if corr is not None:
                ax = self.fig.add_subplot(111)
                im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
                self.fig.colorbar(im, ax=ax, label="همبستگی", shrink=0.8)
                ax.set_title("ماتریس همبستگی", fontsize=13, color=TEXT_COLOR)
                n = corr.shape[0]
                ax.set_xticks(range(n))
                ax.set_yticks(range(n))
                labels = [f"A{i+1}" for i in range(n)]
                ax.set_xticklabels(labels, fontsize=8, color=MUTED_TEXT)
                ax.set_yticklabels(labels, fontsize=8, color=MUTED_TEXT)
                ax.tick_params(colors=MUTED_TEXT)
        else:
            # Contagion plot
            contagion = result.get("contagion_effects", [])
            if contagion:
                rounds_c = [c.get("round", i) for i, c in enumerate(contagion)]
                affected = [c.get("n_affected", c.get("affected_count", 0)) for c in contagion]
                avg_shock = [c.get("avg_shock", 0) for c in contagion]
                ax = self.fig.add_subplot(111)
                ax2_twin = ax.twinx()
                ax.bar(rounds_c, affected, color=ACCENT_COLOR, alpha=0.6, label="تعداد متاثر")
                ax2_twin.plot(rounds_c, avg_shock, color=ERROR_COLOR, linewidth=2, marker="o", markersize=4, label="میانگین شوک")
                ax.set_xlabel("دور سرایت")
                ax.set_ylabel("تعداد متاثر", color=ACCENT_COLOR)
                ax2_twin.set_ylabel("میانگین شوک", color=ERROR_COLOR)
                ax.set_title("شبیه‌سازی سرایت سیستمیک", fontsize=13, color=TEXT_COLOR)
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2_twin.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2,
                         facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
                ax.tick_params(colors=MUTED_TEXT)
                ax2_twin.tick_params(colors=MUTED_TEXT)
                for spine in ax.spines.values():
                    spine.set_color(GRID_COLOR)
                for spine in ax2_twin.spines.values():
                    spine.set_color(GRID_COLOR)
        self.fig.tight_layout()
        self.canvas.draw()


# ═══════════════════════════════════════════════════════════════
# Tab 7: Behavioral Finance  —  مالی رفتاری
# ═══════════════════════════════════════════════════════════════
class BehavioralTab(BaseTab):
    def __init__(self, parent=None):
        self.sp_n_trades = _ispin(200, 20, 5000, 50)
        self.sp_n_returns = _ispin(500, 50, 5000, 100)
        self.cb_method = _combo([
            "اثر ترجیح عمل", "اثر اعتماد به نفس", "رفتار گله‌ای", "شاخص‌های احساسات"
        ], 0)
        self._data_cache = None
        super().__init__(parent)
        self.header.setText("🧠 مالی رفتاری")

    def _group_title(self):
        return "تنظیمات تحلیل رفتاری"

    def _build_inputs(self):
        self.input_form.addRow(_label("تعداد معاملات:"), self.sp_n_trades)
        self.input_form.addRow(_label("تعداد بازده‌ها:"), self.sp_n_returns)
        self.input_form.addRow(_label("نوع تحلیل:"), self.cb_method)

    def _generate_trades(self, n, seed=42):
        np.random.seed(seed)
        trades = []
        for _ in range(n):
            entry = np.random.uniform(80, 120)
            pnl = np.random.normal(100, 500)
            holding = max(1, int(np.random.exponential(5)))
            exit_price = entry + pnl / entry * entry
            trades.append({"entry_price": entry, "exit_price": exit_price, "pnl": pnl, "holding_days": holding})
        return trades

    def _generate_returns(self, n, seed=42):
        np.random.seed(seed)
        return np.random.normal(0.0005, 0.015, n).tolist()

    def _demo_params(self):
        idx = self.cb_method.currentIndex()
        if idx == 0:
            trades = self._generate_trades(200, seed=42)
            self._data_cache = trades
            return {"trades": trades}
        elif idx == 1:
            rets = self._generate_returns(500, seed=42)
            pred = [r + np.random.normal(0, 0.005) for r in rets]
            self._data_cache = rets
            return {"returns": rets, "predicted_returns": pred}
        elif idx == 2:
            np.random.seed(42)
            n_a, n_t = 10, 500
            R = np.random.normal(0.001, 0.02, (n_t, n_a))
            R[:, 0] *= 0.3  # Add common factor
            R += np.random.normal(0, 0.005, (n_t, 1))
            self._data_cache = R
            return {"returns_matrix": R.tolist()}
        else:
            rets = self._generate_returns(500, seed=42)
            volume = np.random.lognormal(15, 0.5, 500).tolist()
            self._data_cache = rets
            return {"returns": rets, "volume": volume}

    def _get_params(self):
        idx = self.cb_method.currentIndex()
        if idx == 0:
            trades = self._generate_trades(self.sp_n_trades.value(), seed=None)
            self._data_cache = trades
            return {"trades": trades}
        elif idx == 1:
            n = self.sp_n_returns.value()
            rets = self._generate_returns(n, seed=None)
            pred = [r + np.random.normal(0, 0.005) for r in rets]
            self._data_cache = rets
            return {"returns": rets, "predicted_returns": pred}
        elif idx == 2:
            np.random.seed()
            n_a, n_t = 10, self.sp_n_returns.value()
            R = np.random.normal(0.001, 0.02, (n_t, n_a))
            R += np.random.normal(0, 0.005, (n_t, 1))
            self._data_cache = R
            return {"returns_matrix": R.tolist()}
        else:
            n = self.sp_n_returns.value()
            rets = self._generate_returns(n, seed=None)
            volume = np.random.lognormal(15, 0.5, n).tolist()
            self._data_cache = rets
            return {"returns": rets, "volume": volume}

    def _task_name(self, params):
        idx = self.cb_method.currentIndex()
        return ["behavioral_disposition", "behavioral_overconfidence", "behavioral_herd", "behavioral_sentiment"][idx]

    def _display_result(self, result):
        if "error" in result:
            self._on_error(result["error"])
            return
        self.fig.clear()
        display = {k: v for k, v in result.items()
                   if not isinstance(v, (list, np.ndarray))}
        populate_table(self.results_table, display)

        idx = self.cb_method.currentIndex()
        if idx == 0:  # Disposition
            ax = self.fig.add_subplot(111)
            trades = self._data_cache or []
            if trades:
                pnls = [t["pnl"] for t in trades]
                holds = [t["holding_days"] for t in trades]
                winners = [i for i, p in enumerate(pnls) if p > 0]
                losers = [i for i, p in enumerate(pnls) if p < 0]
                ax.scatter([holds[i] for i in winners], [pnls[i] for i in winners],
                          c=SUCCESS_COLOR, alpha=0.6, s=30, label="سود")
                ax.scatter([holds[i] for i in losers], [pnls[i] for i in losers],
                          c=ERROR_COLOR, alpha=0.6, s=30, label="زیان")
                ax.axhline(y=0, color=MUTED_TEXT, linestyle="--", alpha=0.5)
                ax.set_xlabel("دروز نگهداری")
                ax.set_ylabel("سود/زیان")
                ax.set_title("اثر ترجیح عمل (Disposition Effect)", fontsize=13, color=TEXT_COLOR)
                ax.legend(facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
                style_axes(ax)
        elif idx == 1:  # Overconfidence
            ax1 = self.fig.add_subplot(121)
            ax2 = self.fig.add_subplot(122)
            oc_score = result.get("overconfidence_score", 0)
            categories_oc = ["خوب", "متوسط", "بالا"]
            vals_oc = [max(0, 100-oc_score), max(0, 60-oc_score), oc_score]
            colors_oc = [SUCCESS_COLOR, MUTED_TEXT, ERROR_COLOR]
            ax1.bar(categories_oc, vals_oc, color=colors_oc, edgecolor=GRID_COLOR)
            ax1.set_title("امتیاز اعتماد به نفس", fontsize=12, color=TEXT_COLOR)
            style_axes(ax1)
            # Win rate pie
            win_rate = result.get("win_rate", 0.5)
            ax2.pie([win_rate, 1-win_rate], labels=["سود", "زیان"],
                   colors=[SUCCESS_COLOR, ERROR_COLOR], autopct='%1.1f%%',
                   textprops={"color": TEXT_COLOR})
            ax2.set_title("نرخ برد", fontsize=12, color=TEXT_COLOR)
        elif idx == 2:  # Herding
            ax = self.fig.add_subplot(111)
            data = self._data_cache
            if data is not None and isinstance(data, np.ndarray):
                R = data
                market = R.mean(axis=1)
                csad = np.mean(np.abs(R - market.reshape(-1, 1)), axis=1)
                ax.plot(csad, color=ACCENT_COLOR, linewidth=0.8)
                ax.fill_between(range(len(csad)), 0, csad, alpha=0.2, color=ACCENT_COLOR)
                ax.set_xlabel("زمان")
                ax.set_ylabel("CSAD")
                ax.set_title("پراکندگی مقطعی عرضی (رفتار گله‌ای)", fontsize=13, color=TEXT_COLOR)
                style_axes(ax)
        elif idx == 3:  # Sentiment
            ax = self.fig.add_subplot(111)
            rets = np.array(self._data_cache or [])
            if len(rets) > 0:
                fg = result.get("fear_greed_index", 50)
                sentiment = result.get("sentiment", "neutral")
                cum_ret = np.cumsum(rets)
                color_line = SUCCESS_COLOR if fg > 50 else ERROR_COLOR if fg < 50 else MUTED_TEXT
                ax.plot(cum_ret, color=color_line, linewidth=1)
                ax.fill_between(range(len(cum_ret)), 0, cum_ret,
                              where=cum_ret >= 0, alpha=0.2, color=SUCCESS_COLOR)
                ax.fill_between(range(len(cum_ret)), 0, cum_ret,
                              where=cum_ret < 0, alpha=0.2, color=ERROR_COLOR)
                ax.axhline(y=0, color=MUTED_TEXT, linestyle="--", alpha=0.5)
                ax.set_title(f"شاخص ترس/طمع: {fg:.1f} ({sentiment})", fontsize=13, color=TEXT_COLOR)
                style_axes(ax)
        self.fig.tight_layout()
        self.canvas.draw()


# ═══════════════════════════════════════════════════════════════
# Tab 8: Monte Carlo Simulation  —  مونت‌کارلو
# ═══════════════════════════════════════════════════════════════
class MonteCarloTab(BaseTab):
    def __init__(self, parent=None):
        self.sp_S0 = _dspin(100, 0.01, 1e6, 1, 2)
        self.sp_mu = _dspin(0.08, -1, 1, 0.01, 4)
        self.sp_sigma = _dspin(0.20, 0.001, 5, 0.01, 4)
        self.sp_T = _dspin(1.0, 0.01, 10, 0.1, 2, " سال")
        self.sp_n_steps = _ispin(252, 10, 5000, 50)
        self.sp_n_sim = _ispin(1000, 50, 50000, 500)
        self.cb_method = _combo(["حرکت براونی هندسی", "شبیه‌سازی پرتفوی"], 0)
        self.sp_initial_val = _dspin(1e6, 1000, 1e12, 1e5, 0)
        self._mc_cache = None
        super().__init__(parent)
        self.header.setText("🎲 مونت‌کارلو")

    def _group_title(self):
        return "تنظیمات شبیه‌سازی مونت‌کارلو"

    def _build_inputs(self):
        self.input_form.addRow(_label("قیمت اولیه (S0):"), self.sp_S0)
        self.input_form.addRow(_label("بازده مورد انتظار (μ):"), self.sp_mu)
        self.input_form.addRow(_label("نوسان‌پذیری (σ):"), self.sp_sigma)
        self.input_form.addRow(_label("زمان (T):"), self.sp_T)
        self.input_form.addRow(_label("تعداد مراحل:"), self.sp_n_steps)
        self.input_form.addRow(_label("تعداد شبیه‌سازی:"), self.sp_n_sim)
        self.input_form.addRow(_label("نوع شبیه‌سازی:"), self.cb_method)
        self.input_form.addRow(_label("ارزش اولیه پرتفوی:"), self.sp_initial_val)

    def _demo_params(self):
        idx = self.cb_method.currentIndex()
        if idx == 0:
            return {
                "S0": 100, "mu": 0.08, "sigma": 0.20, "T": 1.0,
                "n_steps": 252, "n_simulations": 1000,
            }
        else:
            np.random.seed(42)
            n_assets = 5
            mu = np.array([0.08, 0.10, 0.12, 0.06, 0.15])
            A = np.random.randn(n_assets, n_assets) * 0.1
            cov = A @ A.T + np.eye(n_assets) * 0.01
            return {
                "initial_value": 1e6,
                "expected_returns": mu.tolist(),
                "cov_matrix": cov.tolist(),
                "n_steps": 252, "n_simulations": 500,
            }

    def _get_params(self):
        idx = self.cb_method.currentIndex()
        if idx == 0:
            return {
                "S0": self.sp_S0.value(), "mu": self.sp_mu.value(),
                "sigma": self.sp_sigma.value(), "T": self.sp_T.value(),
                "n_steps": self.sp_n_steps.value(),
                "n_simulations": self.sp_n_sim.value(),
            }
        else:
            np.random.seed(42)
            n_assets = 5
            mu = np.random.uniform(0.05, 0.20, n_assets)
            A = np.random.randn(n_assets, n_assets) * 0.1
            cov = A @ A.T + np.eye(n_assets) * 0.01
            return {
                "initial_value": self.sp_initial_val.value(),
                "expected_returns": mu.tolist(),
                "cov_matrix": cov.tolist(),
                "n_steps": self.sp_n_steps.value(),
                "n_simulations": self.sp_n_sim.value(),
            }

    def _task_name(self, params):
        idx = self.cb_method.currentIndex()
        return ["mc_gbm", "mc_portfolio"][idx]

    def _display_result(self, result):
        if "error" in result:
            self._on_error(result["error"])
            return
        self.fig.clear()
        display = {k: v for k, v in result.items()
                   if not isinstance(v, (list, np.ndarray)) and k != "paths"}
        populate_table(self.results_table, display)

        idx = self.cb_method.currentIndex()
        paths = result.get("paths")
        if paths is not None and len(paths) > 0:
            P = np.array(paths)
            ax = self.fig.add_subplot(111)
            n_show = min(200, P.shape[0])
            for i in range(n_show):
                alpha = 0.15 if n_show > 50 else 0.4
                ax.plot(P[i], color=ACCENT_COLOR, alpha=alpha, linewidth=0.5)
            # Mean and percentiles
            mean_path = P.mean(axis=0)
            p5 = np.percentile(P, 5, axis=0)
            p95 = np.percentile(P, 95, axis=0)
            ax.plot(mean_path, color=SUCCESS_COLOR, linewidth=2.5, label="میانگین")
            ax.fill_between(range(len(p5)), p5, p95, alpha=0.15, color=PURPLE_COLOR, label="90% CI")
            ax.set_xlabel("زمان")
            ax.set_ylabel("قیمت / ارزش")
            title = "شبیه‌سازی حرکت براونی هندسی" if idx == 0 else "شبیه‌سازی پرتفوی"
            ax.set_title(title, fontsize=13, color=TEXT_COLOR)
            ax.legend(facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
            style_axes(ax)
        else:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "داده‌ای برای رسم نمودار وجود ندارد",
                   transform=ax.transAxes, ha="center", va="center",
                   color=MUTED_TEXT, fontsize=14)
            ax.set_facecolor(CARD_BG)
        self.fig.tight_layout()
        self.canvas.draw()


# ═══════════════════════════════════════════════════════════════
# Tab 9: Interest Rate Models  —  نرخ بهره
# ═══════════════════════════════════════════════════════════════
class InterestRateTab(BaseTab):
    def __init__(self, parent=None):
        self.sp_r0 = _dspin(0.05, 0, 1, 0.005, 4)
        self.sp_a = _dspin(0.1, 0, 10, 0.01, 4)
        self.sp_b = _dspin(0.05, 0, 1, 0.005, 4)
        self.sp_sigma_ir = _dspin(0.01, 0.0001, 1, 0.001, 4)
        self.sp_T_ir = _dspin(1.0, 0.1, 30, 0.5, 2, " سال")
        self.sp_n_steps_ir = _ispin(252, 10, 5000, 50)
        self.sp_n_sim_ir = _ispin(500, 50, 10000, 100)
        self.cb_model = _combo(["Vasicek", "CIR", "Hull-White", "مدت و تحدب"], 0)
        self._ir_cache = None
        super().__init__(parent)
        self.header.setText("💰 نرخ بهره")

    def _group_title(self):
        return "تنظیمات مدل نرخ بهره"

    def _build_inputs(self):
        self.input_form.addRow(_label("نرخ اولیه (r0):"), self.sp_r0)
        self.input_form.addRow(_label("سرعت بازگشت (a):"), self.sp_a)
        self.input_form.addRow(_label("نرخ بلندمدت (b):"), self.sp_b)
        self.input_form.addRow(_label("نوسان‌پذیری (σ):"), self.sp_sigma_ir)
        self.input_form.addRow(_label("زمان (T):"), self.sp_T_ir)
        self.input_form.addRow(_label("تعداد مراحل:"), self.sp_n_steps_ir)
        self.input_form.addRow(_label("تعداد شبیه‌سازی:"), self.sp_n_sim_ir)
        self.input_form.addRow(_label("مدل:"), self.cb_model)

    def _demo_params(self):
        idx = self.cb_model.currentIndex()
        if idx < 3:
            base = {
                "r0": 0.05, "a": 0.15, "b": 0.05,
                "sigma": 0.01, "T": 2.0,
                "n_steps": 252, "n_simulations": 500,
            }
            return base
        else:
            # Duration/Convexity demo
            return {
                "cashflows": [50, 50, 50, 1050],
                "rates": [0.03, 0.035, 0.04, 0.045],
                "ytm": 0.04,
            }

    def _get_params(self):
        idx = self.cb_model.currentIndex()
        if idx < 3:
            return {
                "r0": self.sp_r0.value(), "a": self.sp_a.value(),
                "b": self.sp_b.value(), "sigma": self.sp_sigma_ir.value(),
                "T": self.sp_T_ir.value(),
                "n_steps": self.sp_n_steps_ir.value(),
                "n_simulations": self.sp_n_sim_ir.value(),
            }
        else:
            return {
                "cashflows": [50, 50, 50, 1050],
                "rates": [0.03, 0.035, 0.04, 0.045],
                "ytm": self.sp_r0.value(),
            }

    def _task_name(self, params):
        idx = self.cb_model.currentIndex()
        return ["vasicek", "cir", "hull_white", "duration_convexity"][idx]

    def _display_result(self, result):
        if "error" in result:
            self._on_error(result["error"])
            return
        self.fig.clear()
        display = {k: v for k, v in result.items()
                   if not isinstance(v, (list, np.ndarray)) and k != "rate_paths"}
        populate_table(self.results_table, display)

        idx = self.cb_model.currentIndex()
        if idx < 3:
            paths = result.get("rate_paths")
            if paths is not None and len(paths) > 0:
                P = np.array(paths)
                ax = self.fig.add_subplot(111)
                n_show = min(100, P.shape[0])
                for i in range(n_show):
                    ax.plot(P[i], color=ACCENT_COLOR, alpha=0.1, linewidth=0.4)
                mean_path = P.mean(axis=0)
                ax.plot(mean_path, color=SUCCESS_COLOR, linewidth=2.5, label="میانگین")
                ax.axhline(y=result.get("long_term_rate", self.sp_b.value()),
                          color=PURPLE_COLOR, linestyle="--", alpha=0.7, label="نرخ بلندمدت")
                model_names = ["Vasicek", "CIR", "Hull-White"]
                ax.set_title(f"مدل {model_names[idx]}", fontsize=13, color=TEXT_COLOR)
                ax.set_xlabel("زمان")
                ax.set_ylabel("نرخ بهره")
                ax.legend(facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
                style_axes(ax)
        else:
            # Duration/Convexity visualization
            ax = self.fig.add_subplot(111)
            dur = result.get("duration", 0)
            conv = result.get("convexity", 0)
            price = result.get("price", result.get("bond_price", 0))
            labels_ir = ["قیمت", "مدت (Duration)", "تحدب (Convexity)"]
            vals_ir = [price, dur, conv]
            colors_ir = [ACCENT_COLOR, SUCCESS_COLOR, PURPLE_COLOR]
            bars = ax.bar(labels_ir, vals_ir, color=colors_ir, edgecolor=GRID_COLOR, width=0.4)
            ax.set_title("تحلیل مدت و تحدب اوراق", fontsize=13, color=TEXT_COLOR)
            style_axes(ax)
            for bar, val in zip(bars, vals_ir):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * abs(bar.get_height()),
                       f"{val:.4f}", ha="center", va="bottom", color=TEXT_COLOR, fontsize=10)
        self.fig.tight_layout()
        self.canvas.draw()


# ═══════════════════════════════════════════════════════════════
# Main Dashboard Widget — assembling all 9 tabs
# ═══════════════════════════════════════════════════════════════
class QuantDashboard(QWidget):
    """Main quantitative finance dashboard with 9 integrated tabs.

    Integrates all quantitative finance modules:
    1. Portfolio Optimization (Markowitz, Sharpe, Efficient Frontier)
    2. Derivatives Pricing (Black-Scholes, Binomial, MC, IV)
    3. Risk Analysis (VaR, CVaR, GARCH, Stress Test)
    4. Time Series Analysis (ARIMA, ADF, Rolling Stats)
    5. Fuzzy Systems (Credit Scoring, ANFIS, Fuzzy AHP)
    6. Network Analysis (Correlation Network, Contagion)
    7. Behavioral Finance (Disposition, Overconfidence, Herding)
    8. Monte Carlo Simulation (GBM, Portfolio MC)
    9. Interest Rate Models (Vasicek, CIR, Hull-White)
    """

    TAB_TITLES = [
        "\u0628\u0647\u06cc\u0646\u0647\u200c\u0633\u0627\u0632\u06cc \u067e\u0631\u062a\u0641\u0648\u06cc",
        "\u0642\u06cc\u0645\u062a\u200c\u06af\u0630\u0627\u0631\u06cc \u0645\u0634\u062a\u0642\u0627\u062a",
        "\u062a\u062d\u0644\u06cc\u0644 \u0631\u06cc\u0633\u06a9",
        "\u062a\u062d\u0644\u06cc\u0644 \u0633\u0631\u06cc \u0632\u0645\u0627\u0646\u06cc",
        "\u0633\u06cc\u0633\u062a\u0645\u200c\u0647\u0627\u06cc \u0641\u0627\u0632\u06cc",
        "\u062a\u062d\u0644\u06cc\u0644 \u0634\u0628\u06a9\u0647",
        "\u0645\u0627\u0644\u06cc \u0631\u0641\u062a\u0627\u0631\u06cc",
        "\u0645\u0648\u0646\u062a\u200c\u06a9\u0627\u0631\u0644\u0648",
        "\u0646\u0631\u062e \u0628\u0647\u0631\u0647",
    ]

    TAB_ICONS = [
        "\u2728", "\u2699\ufe0f", "\u26a0\ufe0f", "\ud83d\udcc8",
        "\ud83c\udf00", "\ud83d\udd78\ufe0f", "\ud83e\udde0", "\ud83c\udfb2", "\ud83d\udcb0",
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("QuantDashboard")
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Title bar
        title_bar = QFrame()
        title_bar.setStyleSheet(f"background-color: #0f172a; border-bottom: 1px solid {GRID_COLOR};")
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(20, 12, 20, 12)
        title_lbl = _label("\ud83d\udcca JurisFinanceAI \u2014 \u062f\u0627\u0634\u0628\u0648\u0631\u062f \u06a9\u0645\u06cc", bold=True, color=TEXT_COLOR, size=16)
        title_layout.addWidget(title_lbl)
        title_layout.addStretch()
        info_lbl = _label("\u0628\u0647\u06cc\u0646\u0647\u200c\u0633\u0627\u0632\u06cc | \u0645\u0634\u062a\u0642\u0627\u062a | \u0631\u06cc\u0633\u06a9 | \u0633\u0631\u06cc \u0632\u0645\u0627\u0646\u06cc | \u0641\u0627\u0632\u06cc | \u0634\u0628\u06a9\u0647 | \u0631\u0641\u062a\u0627\u0631\u06cc | MC | \u0646\u0631\u062e \u0628\u0647\u0631\u0647",
                         color=MUTED_TEXT, size=11)
        title_layout.addWidget(info_lbl)
        main_layout.addWidget(title_bar)

        # Tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet(DARK_STYLE)
        self.tab_widget.setDocumentMode(True)
        self.tab_widget.setMovable(False)
        self.tab_widget.setUsesScrollButtons(True)

        # Create all 9 tabs
        tabs = [
            PortfolioTab(),
            DerivativesTab(),
            RiskTab(),
            TimeSeriesTab(),
            FuzzyTab(),
            NetworkTab(),
            BehavioralTab(),
            MonteCarloTab(),
            InterestRateTab(),
        ]

        for i, (tab, title, icon) in enumerate(zip(tabs, self.TAB_TITLES, self.TAB_ICONS)):
            self.tab_widget.addTab(tab, f"{icon}  {title}")

        main_layout.addWidget(self.tab_widget)

    def get_current_tab(self):
        """Return the currently visible tab widget."""
        return self.tab_widget.currentWidget()

    def set_tab(self, index):
        """Switch to a specific tab by index (0-8)."""
        if 0 <= index < self.tab_widget.count():
            self.tab_widget.setCurrentIndex(index)

    def run_demo_on_current(self):
        """Trigger the Demo button on the current tab."""
        tab = self.get_current_tab()
        if tab and hasattr(tab, 'demo_btn'):
            tab.demo_btn.click()

    def run_execute_on_current(self):
        """Trigger the Execute button on the current tab."""
        tab = self.get_current_tab()
        if tab and hasattr(tab, 'exec_btn'):
            tab.exec_btn.click()
