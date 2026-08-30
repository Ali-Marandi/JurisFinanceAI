import numpy as np
import re
from collections import Counter


class FinancialNLPEngine:
    """Natural Language Processing engine for financial text analysis.

    Implements sentiment analysis, named entity recognition for
    financial instruments, extractive summarization, automated
    risk report generation, and RAG-style retrieval from local knowledge.
    All methods are rule-based + statistical (no external NLP libraries).
    """

    FINANCIAL_POSITIVE = [
        'profit', 'growth', 'gain', 'increase', 'surge', 'rally', 'bullish',
        'outperform', 'upgrade', 'beat', 'exceed', 'strong', 'robust',
        'recovery', 'expansion', 'dividend', 'yield', 'return', 'upside',
        'opportunity', 'innovation', 'breakthrough', 'acquisition', 'merger',
        'saudi', 'profitable', 'efficient', 'optimistic', 'confidence',
        'recommend', 'overweight', 'buy', 'long', 'momentum'
    ]

    FINANCIAL_NEGATIVE = [
        'loss', 'decline', 'drop', 'fall', 'crash', 'bearish', 'risk',
        'downgrade', 'miss', 'debt', 'default', 'recession', 'inflation',
        'volatility', 'uncertainty', 'crisis', 'sanction', 'deficit',
        'negative', 'weak', 'poor', 'disappoint', 'cut', 'reduce',
        'layoff', 'bankruptcy', 'fraud', 'investigation', 'lawsuit',
        'sell', 'short', 'panic', 'fear', 'concern', 'warning'
    ]

    FINANCIAL_ENTITIES = {
        'CURRENCY': [
            'USD', 'EUR', 'GBP', 'JPY', 'CNY', 'IRR', 'SAR', 'AED',
            'dollar', 'euro', 'pound', 'yen', 'yuan', 'rial', 'riyal',
            'BTC', 'ETH', 'bitcoin', 'ethereum'
        ],
        'INDEX': [
            'S&P', 'Dow Jones', 'NASDAQ', 'FTSE', 'DAX', 'Nikkei',
            'TEDPIX', 'Shanghai', 'Bovespa', 'Hang Seng', 'KOSPI',
            'index', 'benchmark'
        ],
        'COMMODITY': [
            'oil', 'gold', 'silver', 'copper', 'iron', 'natural gas',
            'wheat', 'corn', 'soybean', 'Brent', 'WTI', 'OPEC'
        ],
        'RATE': [
            'interest rate', 'federal reserve', 'Fed', 'ECB', 'CBI',
            'monetary policy', 'basis point', 'yield curve', 'libor',
            'inflation rate', 'unemployment rate'
        ],
        'SECTOR': [
            'banking', 'technology', 'healthcare', 'energy', 'real estate',
            'automotive', 'pharmaceutical', 'retail', 'telecom', 'mining',
            'petrochemical', 'insurance', 'fintech', 'crypto'
        ]
    }

    def sentiment_analysis(self, text):
        """Analyze sentiment of financial text.

        Returns overall sentiment score (-1 to +1), magnitude,
        and breakdown by category.
        """
        if not text or not isinstance(text, str):
            return {'score': 0, 'magnitude': 0, 'breakdown': {}}

        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)

        # Positive/negative scoring
        pos_count = sum(1 for w in words if w in self.FINANCIAL_POSITIVE)
        neg_count = sum(1 for w in words if w in self.FINANCIAL_NEGATIVE)

        # Intensity modifiers
        intensifiers = {'very', 'extremely', 'highly', 'significantly', 'substantially'}
        negators = {'not', 'no', 'never', 'neither', 'nor', 'without'}

        pos_words_found = []
        neg_words_found = []

        for i, w in enumerate(words):
            if w in self.FINANCIAL_POSITIVE:
                multiplier = 1.5 if (i > 0 and words[i-1] in intensifiers) else 1.0
                if i > 0 and words[i-1] in negators:
                    neg_words_found.append(w)
                    neg_count += multiplier
                else:
                    pos_words_found.append(w)
                    pos_count += multiplier
            elif w in self.FINANCIAL_NEGATIVE:
                multiplier = 1.5 if (i > 0 and words[i-1] in intensifiers) else 1.0
                if i > 0 and words[i-1] in negators:
                    pos_words_found.append(w)
                    pos_count += multiplier
                else:
                    neg_words_found.append(w)
                    neg_count += multiplier

        total = pos_count + neg_count
        if total == 0:
            score = 0
            magnitude = 0
        else:
            score = (pos_count - neg_count) / total
            magnitude = total / (len(words) + 1)

        # Category breakdown
        categories = {
            'market_direction': 0,
            'volatility': 0,
            'credit_risk': 0,
            'macro': 0
        }

        market_words = ['bullish', 'bearish', 'rally', 'crash', 'trend', 'momentum']
        vol_words = ['volatility', 'uncertainty', 'risk', 'stable', 'turbulent']
        credit_words = ['default', 'debt', 'credit', 'downgrade', 'bankruptcy']
        macro_words = ['inflation', 'gdp', 'unemployment', 'fed', 'ecb', 'rate']

        for w in words:
            if w in market_words:
                categories['market_direction'] += 1 if w in pos_words_found else -1
            if w in vol_words:
                categories['volatility'] += 1
            if w in credit_words:
                categories['credit_risk'] -= 1
            if w in macro_words:
                categories['macro'] += 0.5

        # Sentiment label
        if score > 0.3:
            label = 'Very Bullish'
        elif score > 0.1:
            label = 'Bullish'
        elif score > -0.1:
            label = 'Neutral'
        elif score > -0.3:
            label = 'Bearish'
        else:
            label = 'Very Bearish'

        return {
            'score': float(np.clip(score, -1, 1)),
            'magnitude': float(magnitude),
            'label': label,
            'positive_count': int(pos_count),
            'negative_count': int(neg_count),
            'positive_words': pos_words_found[:10],
            'negative_words': neg_words_found[:10],
            'category_breakdown': categories
        }

    def named_entity_recognition(self, text):
        """Extract financial named entities from text.

        Identifies currencies, indices, commodities, rates, sectors,
        and numerical financial values.
        """
        if not text or not isinstance(text, str):
            return {'entities': [], 'numerical_values': []}

        text_lower = text.lower()
        entities = []

        # Entity detection
        for entity_type, keywords in self.FINANCIAL_ENTITIES.items():
            for kw in keywords:
                pattern = re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE)
                matches = pattern.findall(text)
                for m in matches:
                    entities.append({
                        'text': m,
                        'type': entity_type,
                        'count': len(matches)
                    })

        # Deduplicate
        seen = set()
        unique_entities = []
        for e in entities:
            key = (e['text'].lower(), e['type'])
            if key not in seen:
                seen.add(key)
                unique_entities.append(e)

        # Extract numerical values
        numerical_values = []
        num_patterns = [
            (r'\$([\d,]+\.?\d*)\s*(billion|million|trillion)?', 'USD'),
            (r'(\d+\.?\d*)\s*%', 'percentage'),
            (r'(\d+)\s*basis\s*points?', 'basis_points'),
            (r'(\d+\.?\d*)\s*(bbl|barrel)', 'barrels'),
        ]

        for pattern, unit in num_patterns:
            matches = re.findall(pattern, text_lower)
            for m in matches:
                value = m[0] if isinstance(m, str) else m[0]
                try:
                    numerical_values.append({
                        'value': float(value.replace(',', '')),
                        'unit': unit,
                        'raw': str(m)
                    })
                except ValueError:
                    continue

        return {
            'entities': unique_entities,
            'numerical_values': numerical_values,
            'entity_type_counts': dict(Counter(e['type'] for e in unique_entities))
        }

    def extractive_summarization(self, text, n_sentences=3):
        """Extractive summarization using sentence scoring.

        Scores sentences by keyword density, position, and length.
        Returns the top-N most important sentences.
        """
        if not text or not isinstance(text, str):
            return {'summary': '', 'sentence_scores': []}

        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if len(sentences) <= n_sentences:
            return {
                'summary': ' '.join(sentences),
                'sentence_scores': [(s, 1.0) for s in sentences]
            }

        # Score each sentence
        all_words = re.findall(r'\b\w+\b', text.lower())
        word_freq = Counter(all_words)
        important_words = set(
            self.FINANCIAL_POSITIVE + self.FINANCIAL_NEGATIVE +
            [k for keys in self.FINANCIAL_ENTITIES.values() for k in keys]
        )

        scores = []
        for i, sent in enumerate(sentences):
            words = re.findall(r'\b\w+\b', sent.lower())

            # Keyword score
            keyword_score = sum(1 for w in words if w in important_words)

            # TF score
            tf_score = sum(word_freq.get(w, 0) for w in words) / (len(words) + 1)

            # Position score (first and last sentences are important)
            position_score = 1.0 if i == 0 else (0.8 if i == len(sentences) - 1 else 0.5)

            # Length score (prefer medium-length sentences)
            length_score = 1.0 - abs(len(words) - 20) / 40
            length_score = max(0, length_score)

            # Numerical score (sentences with numbers are informative)
            num_count = len(re.findall(r'\d+', sent))
            num_score = min(num_count * 0.3, 1.0)

            total_score = (
                0.3 * keyword_score +
                0.2 * tf_score +
                0.2 * position_score +
                0.15 * length_score +
                0.15 * num_score
            )

            scores.append((sent, total_score))

        # Select top sentences
        scores.sort(key=lambda x: -x[1])
        top_sentences = scores[:n_sentences]
        top_sentences.sort(key=lambda x: sentences.index(x[0]))  # Original order

        summary = ' '.join(s[0] for s in top_sentences)

        return {
            'summary': summary,
            'sentence_scores': [(s, float(sc)) for s, sc in scores],
            'selected_indices': [sentences.index(s) for s, _ in top_sentences]
        }

    def risk_report_generator(self, portfolio_data, market_data=None):
        """Generate an automated risk assessment report in text form.

        Takes portfolio positions and optionally market data,
        produces a structured risk report.
        """
        if isinstance(portfolio_data, dict):
            names = portfolio_data.get('names', [])
            weights = np.array(portfolio_data.get('weights', []), dtype=float)
            returns = portfolio_data.get('returns', None)
        else:
            return {'report': 'Invalid portfolio data format.'}

        report_sections = []

        # Section 1: Portfolio Overview
        total_weight = np.sum(weights)
        report_sections.append(
            f"Portfolio Overview: {len(names)} assets with total weight "
            f"{total_weight:.2%}. "
            f"Largest position: {names[np.argmax(weights)]} at {np.max(weights):.2%}. "
            f"Smallest position: {names[np.argmin(weights)]} at {np.min(weights):.2%}."
        )

        # Section 2: Concentration Risk
        hhi = np.sum(weights ** 2)  # Herfindahl-Hirschman Index
        max_allocation = np.max(weights)
        if hhi > 0.25:
            concentration = f"HIGH concentration risk (HHI={hhi:.3f}). "
        elif hhi > 0.15:
            concentration = f"MODERATE concentration risk (HHI={hhi:.3f}). "
        else:
            concentration = f"LOW concentration risk (HHI={hhi:.3f}). "
        concentration += f"Maximum allocation is {max_allocation:.2%}."
        report_sections.append(concentration)

        # Section 3: Return Analysis
        if returns is not None and len(returns) > 1:
            returns_arr = np.asarray(returns, dtype=float)
            mean_ret = np.mean(returns_arr)
            std_ret = np.std(returns_arr, ddof=1)
            sharpe = mean_ret / (std_ret + 1e-10)

            report_sections.append(
                f"Return Analysis: Mean return {mean_ret:.4%}, "
                f"volatility {std_ret:.4%}, Sharpe ratio {sharpe:.2f}. "
                f"Best period: {np.max(returns_arr):.4%}, "
                f"Worst period: {np.min(returns_arr):.4%}."
            )

            # Value at Risk
            sorted_returns = np.sort(returns_arr)
            var_95 = -np.percentile(sorted_returns, 5)
            var_99 = -np.percentile(sorted_returns, 1)
            report_sections.append(
                f"Risk Metrics: VaR(95%)={var_95:.4%}, VaR(99%)={var_99:.4%}. "
                f"Maximum drawdown would require full return series analysis."
            )
        else:
            report_sections.append(
                "Return Analysis: Insufficient return data provided for analysis."
            )

        # Section 4: Recommendations
        if hhi > 0.25:
            report_sections.append(
                "Recommendation: Consider diversifying to reduce concentration risk. "
                "No single asset should exceed 25% of the portfolio."
            )
        else:
            report_sections.append(
                "Recommendation: Portfolio diversification appears adequate. "
                "Focus on individual asset risk-return optimization."
            )

        full_report = '\n'.join(report_sections)

        return {
            'report': full_report,
            'sections': report_sections,
            'hhi': float(hhi),
            'n_assets': len(names)
        }

    def rag_retrieval(self, query, knowledge_base=None, top_k=5):
        """Retrieval-Augmented Generation style retrieval from local knowledge.

        Computes TF-IDF similarity between query and knowledge base entries.
        Returns ranked results with relevance scores.
        """
        if knowledge_base is None:
            knowledge_base = self._default_knowledge_base()

        query_lower = query.lower()
        query_terms = set(re.findall(r'\b\w+\b', query_lower))

        # Compute IDF for knowledge base
        all_docs = list(knowledge_base.values())
        n_docs = len(all_docs)
        doc_freq = Counter()
        for doc in all_docs:
            terms = set(re.findall(r'\b\w+\b', doc.lower()))
            for term in terms:
                doc_freq[term] += 1

        results = []
        for key, doc in knowledge_base.items():
            doc_terms = re.findall(r'\b\w+\b', doc.lower())
            doc_term_set = set(doc_terms)

            # TF-IDF score
            score = 0
            for term in query_terms:
                if term in doc_term_set:
                    tf = doc_terms.count(term) / (len(doc_terms) + 1)
                    idf = np.log((n_docs + 1) / (doc_freq.get(term, 0) + 1)) + 1
                    score += tf * idf

            # Bonus for exact phrase match
            if query_lower in doc.lower():
                score *= 2

            results.append({'key': key, 'content': doc, 'score': float(score)})

        results.sort(key=lambda x: -x['score'])
        return results[:top_k]

    def _default_knowledge_base(self):
        return {
            'portfolio_optimization': (
                'Portfolio optimization is the process of selecting the best mix of assets '
                'to maximize return for a given level of risk. Key methods include Markowitz '
                'mean-variance optimization, Black-Litterman model, risk parity, and '
                'maximum Sharpe ratio. The efficient frontier represents all optimal portfolios.'
            ),
            'value_at_risk': (
                'Value at Risk (VaR) measures the maximum potential loss over a specified '
                'time period at a given confidence level. Common methods: Historical simulation, '
                'parametric (variance-covariance), and Monte Carlo simulation. VaR does not '
                'capture tail risk, use Conditional VaR (Expected Shortfall) for that.'
            ),
            'black_scholes': (
                'The Black-Scholes model prices European options using closed-form solution. '
                'Key assumptions: constant volatility, log-normal distribution, no dividends, '
                'continuous trading. Greeks (Delta, Gamma, Theta, Vega, Rho) measure '
                'sensitivity to underlying parameters.'
            ),
            'garch': (
                'GARCH models capture volatility clustering in financial time series. '
                'GARCH(1,1) is most common: variance depends on past variance and past '
                'squared returns. Extensions: EGARCH (asymmetric), GJR-GARCH (leverage), '
                'and integrated GARCH for long-memory volatility.'
            ),
            'fuzzy_logic': (
                'Fuzzy logic handles uncertainty in financial decision-making. Fuzzy numbers '
                'use alpha-cuts and membership functions. Applications: Fuzzy portfolio '
                'optimization with fuzzy returns, fuzzy credit scoring with linguistic variables, '
                'ANFIS for adaptive neuro-fuzzy inference, and Fuzzy AHP for multi-criteria decisions.'
            ),
            'prospect_theory': (
                'Prospect Theory (Kahneman-Tversky) describes how people actually make '
                'decisions under risk. Key features: reference dependence, loss aversion '
                '(losses hurt 2x more than equivalent gains), diminishing sensitivity, '
                'and probability weighting (overweight small probabilities).'
            ),
            'topological_analysis': (
                'Topological Data Analysis (TDA) studies the shape of financial data. '
                'Persistent homology tracks features across scales. Betti numbers count '
                'connected components (B0), loops (B1), and voids (B2). Useful for '
                'detecting market regime changes and systemic risk.'
            ),
            'quantum_finance': (
                'Quantum computing promises exponential speedup for financial algorithms. '
                'QAOA for portfolio optimization, Quantum Amplitude Estimation for option '
                'pricing (quadratic speedup over Monte Carlo), and Quantum Kernel methods '
                'for classification. Currently in simulation phase; hardware access needed.'
            ),
        }

    def full_analysis(self, text):
        """Run complete NLP pipeline on financial text."""
        sentiment = self.sentiment_analysis(text)
        entities = self.named_entity_recognition(text)
        summary = self.extractive_summarization(text)

        return {
            'sentiment': sentiment,
            'entities': entities,
            'summary': summary
        }
