import numpy as np
import json
import time
from datetime import datetime, timedelta


class LiveDataEngine:
    """Live market data engine using Yahoo Finance and Binance APIs.

    Fetches real-time and historical data via public HTTP APIs.
    No API key required for basic endpoints.
    Includes caching, rate limiting, and fallback mechanisms.
    """

    def __init__(self):
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes
        self._last_request = 0
        self._min_interval = 0.5  # Rate limit
        self._session = None

    def _get_session(self):
        """Get or create HTTP session."""
        if self._session is None:
            try:
                import httpx
                self._session = httpx.Client(timeout=15, follow_redirects=True)
            except ImportError:
                self._session = None
        return self._session

    def _rate_limit(self):
        elapsed = time.time() - self._last_request
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request = time.time()

    def _cache_get(self, key):
        if key in self._cache:
            data, timestamp = self._cache[key]
            if time.time() - timestamp < self._cache_ttl:
                return data
        return None

    def _cache_set(self, key, data):
        self._cache[key] = (data, time.time())

    def fetch_yahoo_history(self, symbol, period='1y', interval='1d'):
        """Fetch historical price data from Yahoo Finance.

        Uses the v8 chart API endpoint (no API key needed).

        Args:
            symbol: Ticker symbol (e.g., 'AAPL', 'BTC-USD')
            period: '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'
            interval: '1d', '1wk', '1mo'

        Returns:
            dict with 'dates', 'prices', 'volumes', 'returns'
        """
        cache_key = f'yahoo_{symbol}_{period}_{interval}'
        cached = self._cache_get(cache_key)
        if cached:
            return cached

        self._rate_limit()
        session = self._get_session()
        if session is None:
            return self._fallback_data(symbol, period)

        # Yahoo Finance chart API
        period_map = {
            '1mo': '1mo', '3mo': '3mo', '6mo': '6mo',
            '1y': '1y', '2y': '2y', '5y': '5y', 'max': 'max'
        }
        interval_map = {'1d': '1d', '1wk': '1wk', '1mo': '1mo'}

        url = f'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}'
        params = {
            'range': period_map.get(period, '1y'),
            'interval': interval_map.get(interval, '1d'),
            'includePrePost': 'false'
        }
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) JurisFinanceAI/4.1.0'
        }

        try:
            resp = session.get(url, params=params, headers=headers)
            if resp.status_code != 200:
                return self._fallback_data(symbol, period)

            data = resp.json()
            result = data.get('chart', {}).get('result', [])
            if not result:
                return self._fallback_data(symbol, period)

            r = result[0]
            timestamps = r.get('timestamp', [])
            quotes = r.get('indicators', {}).get('quote', [{}])[0]

            closes = np.array(quotes.get('close', []), dtype=float)
            volumes = np.array(quotes.get('volume', []), dtype=float)
            opens = np.array(quotes.get('open', []), dtype=float)
            highs = np.array(quotes.get('high', []), dtype=float)
            lows = np.array(quotes.get('low', []), dtype=float)

            # Remove NaN
            valid = ~np.isnan(closes)
            closes = closes[valid]
            volumes = volumes[valid]
            opens = opens[valid]
            highs = highs[valid]
            lows = lows[valid]
            timestamps = [timestamps[i] for i in range(len(valid)) if valid[i]]

            dates = [datetime.fromtimestamp(ts).strftime('%Y-%m-%d') for ts in timestamps]
            returns = np.diff(np.log(closes + 1e-10)) if len(closes) > 1 else np.array([])

            result_data = {
                'symbol': symbol,
                'dates': dates,
                'opens': opens,
                'highs': highs,
                'lows': lows,
                'closes': closes,
                'volumes': volumes,
                'returns': returns,
                'n_periods': len(closes),
                'last_price': float(closes[-1]) if len(closes) > 0 else 0,
                'change_pct': float((closes[-1] / closes[0] - 1) * 100) if len(closes) > 1 else 0,
                'mean_return': float(np.mean(returns)) if len(returns) > 0 else 0,
                'volatility': float(np.std(returns) * np.sqrt(252)) if len(returns) > 0 else 0,
                'source': 'Yahoo Finance',
                'fetched_at': datetime.now().isoformat(),
            }

            self._cache_set(cache_key, result_data)
            return result_data

        except Exception as e:
            return self._fallback_data(symbol, period)

    def fetch_yahoo_batch(self, symbols, period='1y'):
        """Fetch data for multiple symbols."""
        results = {}
        for symbol in symbols:
            results[symbol] = self.fetch_yahoo_history(symbol, period)
            time.sleep(0.3)  # Be gentle
        return results

    def fetch_crypto(self, symbol='BTCUSDT', days=90):
        """Fetch cryptocurrency data from Binance public API.

        No API key required for klines endpoint.

        Args:
            symbol: Binance symbol (e.g., 'BTCUSDT', 'ETHUSDT')
            days: Number of days of history

        Returns:
            Same format as fetch_yahoo_history
        """
        cache_key = f'binance_{symbol}_{days}'
        cached = self._cache_get(cache_key)
        if cached:
            return cached

        self._rate_limit()
        session = self._get_session()
        if session is None:
            return self._fallback_data(symbol, f'{days}d')

        url = 'https://api.binance.com/api/v3/klines'
        params = {
            'symbol': symbol,
            'interval': '1d',
            'limit': min(days, 1000)
        }

        try:
            resp = session.get(url, params=params)
            if resp.status_code != 200:
                return self._fallback_data(symbol, f'{days}d')

            klines = resp.json()
            closes = np.array([float(k[4]) for k in klines])
            volumes = np.array([float(k[5]) for k in klines])
            opens = np.array([float(k[1]) for k in klines])
            highs = np.array([float(k[2]) for k in klines])
            lows = np.array([float(k[3]) for k in klines])
            dates = [datetime.fromtimestamp(k[0] / 1000).strftime('%Y-%m-%d') for k in klines]

            returns = np.diff(np.log(closes + 1e-10)) if len(closes) > 1 else np.array([])

            result_data = {
                'symbol': symbol,
                'dates': dates,
                'opens': opens,
                'highs': highs,
                'lows': lows,
                'closes': closes,
                'volumes': volumes,
                'returns': returns,
                'n_periods': len(closes),
                'last_price': float(closes[-1]),
                'change_pct': float((closes[-1] / closes[0] - 1) * 100),
                'mean_return': float(np.mean(returns)) if len(returns) > 0 else 0,
                'volatility': float(np.std(returns) * np.sqrt(365)) if len(returns) > 0 else 0,
                'source': 'Binance',
                'fetched_at': datetime.now().isoformat(),
            }

            self._cache_set(cache_key, result_data)
            return result_data

        except Exception:
            return self._fallback_data(symbol, f'{days}d')

    def fetch_multiasset_data(self, symbols_data, period='1y'):
        """Fetch and align data for multiple assets.

        Args:
            symbols_data: list of dicts with 'symbol' and 'source' keys
                [{'symbol': 'AAPL', 'source': 'yahoo'}, {'symbol': 'BTCUSDT', 'source': 'binance'}]
        """
        all_closes = []
        all_returns = []
        names = []
        common_dates = None

        for item in symbols_data:
            sym = item['symbol']
            src = item.get('source', 'yahoo')

            if src == 'binance':
                data = self.fetch_crypto(sym)
            else:
                data = self.fetch_yahoo_history(sym, period)

            if data.get('closes') is not None and len(data['closes']) > 0:
                all_closes.append(data['closes'])
                all_returns.append(data['returns'])
                names.append(sym)

        if not all_closes:
            return {'error': 'No data fetched for any symbol'}

        # Align to shortest series
        min_len = min(len(r) for r in all_returns if len(r) > 0)
        aligned_returns = np.array([r[-min_len:] for r in all_returns if len(r) >= min_len])

        if aligned_returns.shape[0] < 2:
            aligned_returns = aligned_returns.reshape(1, -1)

        return {
            'names': names,
            'returns': aligned_returns,
            'expected_returns': np.mean(aligned_returns, axis=1) * 252,
            'cov_matrix': np.cov(aligned_returns) * 252 if aligned_returns.shape[0] > 1 else np.array([[np.var(aligned_returns) * 252]]),
            'correlation': np.corrcoef(aligned_returns) if aligned_returns.shape[0] > 1 else np.array([[1.0]]),
            'n_assets': len(names),
            'n_periods': min_len,
        }

    def search_symbols(self, query):
        """Search for stock/crypto symbols using Yahoo search."""
        self._rate_limit()
        session = self._get_session()
        if session is None:
            return []

        url = f'https://query1.finance.yahoo.com/v1/finance/search'
        params = {'q': query, 'quotesCount': 10, 'newsCount': 0}
        headers = {'User-Agent': 'Mozilla/5.0 JurisFinanceAI/4.1.0'}

        try:
            resp = session.get(url, params=params, headers=headers)
            if resp.status_code != 200:
                return []
            data = resp.json()
            results = []
            for item in data.get('quotes', [])[:10]:
                results.append({
                    'symbol': item.get('symbol', ''),
                    'name': item.get('shortname', item.get('longname', '')),
                    'type': item.get('quoteType', ''),
                    'exchange': item.get('exchange', ''),
                })
            return results
        except Exception:
            return []

    def _fallback_data(self, symbol, period):
        """Generate realistic fallback data when API is unavailable."""
        np.random.seed(hash(symbol) % 2**31)
        n = {'1mo': 22, '3mo': 65, '6mo': 130, '1y': 252, '2y': 504, '5y': 1260}
        n_periods = n.get(period, 252)

        mu = 0.0004
        sigma = 0.015
        returns = np.random.randn(n_periods) * sigma + mu
        prices = 100 * np.exp(np.cumsum(returns))
        dates = [(datetime.now() - timedelta(days=n_periods - i)).strftime('%Y-%m-%d')
                 for i in range(n_periods)]

        return {
            'symbol': symbol,
            'dates': dates,
            'closes': prices,
            'volumes': np.random.randint(100000, 10000000, n_periods).astype(float),
            'opens': prices * (1 + np.random.randn(n_periods) * 0.005),
            'highs': prices * (1 + np.abs(np.random.randn(n_periods) * 0.01)),
            'lows': prices * (1 - np.abs(np.random.randn(n_periods) * 0.01)),
            'returns': returns,
            'n_periods': n_periods,
            'last_price': float(prices[-1]),
            'change_pct': float((prices[-1] / prices[0] - 1) * 100),
            'mean_return': float(mu * 252),
            'volatility': float(sigma * np.sqrt(252)),
            'source': 'Simulated (API unavailable)',
            'fetched_at': datetime.now().isoformat(),
        }
