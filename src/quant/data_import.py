import numpy as np
import csv
import os
from datetime import datetime


class DataImporter:
    """Import financial data from CSV, Excel, and text formats.

    Supports OHLCV data, return series, and correlation matrices.
    Auto-detects formats and validates data integrity.
    """

    def import_csv(self, filepath, date_col=None, price_cols=None,
                    has_header=True, delimiter=',', skip_rows=0):
        """Import financial data from CSV file.

        Args:
            filepath: Path to CSV file
            date_col: Column index for dates (None = no dates)
            price_cols: List of column indices for price data (None = all numeric)
            has_header: Whether file has header row
            delimiter: Column delimiter
            skip_rows: Number of rows to skip

        Returns:
            dict with 'dates', 'names', 'data', 'returns' keys
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f'File not found: {filepath}')

        with open(filepath, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()

        lines = lines[skip_rows:]
        if not lines:
            raise ValueError('File is empty')

        # Parse rows
        rows = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            rows.append(line.split(delimiter))

        if not rows:
            raise ValueError('No data rows found')

        # Extract header
        if has_header:
            header = rows[0]
            data_rows = rows[1:]
        else:
            header = [f'Col_{i}' for i in range(len(rows[0]))]
            data_rows = rows

        # Find numeric columns
        if price_cols is None:
            price_cols = []
            for j in range(len(header)):
                numeric_count = 0
                for row in data_rows[:min(20, len(data_rows))]:
                    if j < len(row):
                        try:
                            float(row[j].strip())
                            numeric_count += 1
                        except ValueError:
                            pass
                if numeric_count > len(data_rows[:20]) * 0.5:
                    price_cols.append(j)

        # Extract dates
        dates = []
        data_start = 0
        if date_col is not None:
            for row in data_rows:
                if date_col < len(row):
                    dates.append(row[date_col].strip())
            # Remove date column from price data
            if date_col in price_cols:
                price_cols = [c for c in price_cols if c != date_col]

        # Extract price data
        names = [header[j].strip() for j in price_cols]
        data_matrix = []
        valid_rows = []

        for row in data_rows:
            values = []
            valid = True
            for j in price_cols:
                if j < len(row):
                    try:
                        values.append(float(row[j].strip()))
                    except ValueError:
                        valid = False
                        break
                else:
                    valid = False
                    break
            if valid and len(values) == len(price_cols):
                data_matrix.append(values)
                valid_rows.append(row)

        if not data_matrix:
            raise ValueError('No valid numeric data found')

        data = np.array(data_matrix)

        # Compute returns
        returns = np.diff(np.log(data + 1e-10), axis=0) if data.shape[0] > 1 else np.zeros_like(data)

        return {
            'dates': dates[:len(data_matrix)],
            'names': names,
            'data': data,
            'returns': returns,
            'n_assets': len(names),
            'n_periods': len(data_matrix),
            'format': 'CSV',
            'filepath': filepath,
            'imported_at': datetime.now().isoformat()
        }

    def import_excel(self, filepath, sheet_name=None, price_cols=None,
                      has_header=True, skip_rows=0):
        """Import financial data from Excel file.

        Uses openpyxl if available, otherwise falls back to csv-like parsing.
        """
        try:
            import openpyxl
            return self._import_excel_openpyxl(filepath, sheet_name, price_cols, has_header, skip_rows)
        except ImportError:
            # Fallback: convert to CSV-like format
            return self._import_excel_fallback(filepath, sheet_name, price_cols, has_header, skip_rows)

    def _import_excel_openpyxl(self, filepath, sheet_name, price_cols, has_header, skip_rows):
        import openpyxl
        wb = openpyxl.load_workbook(filepath, data_only=True, read_only=True)

        if sheet_name:
            ws = wb[sheet_name]
        else:
            ws = wb.active

        rows = []
        for i, row in enumerate(ws.iter_rows(values_only=True)):
            if i < skip_rows:
                continue
            rows.append([str(c) if c is not None else '' for c in row])

        wb.close()

        if not rows:
            raise ValueError('No data found in Excel file')

        header = rows[0] if has_header else [f'Col_{i}' for i in range(len(rows[0]))]
        data_rows = rows[1:] if has_header else rows

        if price_cols is None:
            price_cols = []
            for j in range(len(header)):
                numeric = sum(1 for r in data_rows[:20] if j < len(r) and self._is_numeric(r[j]))
                if numeric > len(data_rows[:20]) * 0.5:
                    price_cols.append(j)

        names = [header[j].strip() for j in price_cols]
        data_matrix = []

        for row in data_rows:
            values = []
            for j in price_cols:
                if j < len(row) and self._is_numeric(row[j]):
                    values.append(float(row[j]))
                else:
                    break
            if len(values) == len(price_cols):
                data_matrix.append(values)

        data = np.array(data_matrix)
        returns = np.diff(np.log(data + 1e-10), axis=0) if data.shape[0] > 1 else np.zeros_like(data)

        return {
            'names': names,
            'data': data,
            'returns': returns,
            'n_assets': len(names),
            'n_periods': len(data_matrix),
            'format': 'Excel',
            'filepath': filepath,
            'imported_at': datetime.now().isoformat()
        }

    def _import_excel_fallback(self, filepath, sheet_name, price_cols, has_header, skip_rows):
        raise ImportError('openpyxl not available. Install with: pip install openpyxl')

    def import_clipboard_text(self, text, delimiter='\t', has_header=True):
        """Import data from clipboard/tab-separated text."""
        import io
        temp_path = '/tmp/_jfa_clipboard_import.csv'
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(text)
        try:
            result = self.import_csv(temp_path, delimiter=delimiter, has_header=has_header)
            result['format'] = 'Clipboard'
            return result
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def auto_import(self, filepath, **kwargs):
        """Auto-detect file format and import."""
        ext = os.path.splitext(filepath)[1].lower()
        if ext in ('.csv', '.txt'):
            return self.import_csv(filepath, **kwargs)
        elif ext in ('.xlsx', '.xls'):
            return self.import_excel(filepath, **kwargs)
        else:
            # Try CSV first
            try:
                return self.import_csv(filepath, **kwargs)
            except Exception:
                return self.import_excel(filepath, **kwargs)

    def validate_data(self, data_dict):
        """Validate imported data quality."""
        data = data_dict.get('data', np.array([]))
        returns = data_dict.get('returns', np.array([]))
        issues = []

        if data.size == 0:
            issues.append('No data loaded')
            return {'valid': False, 'issues': issues, 'quality_score': 0}

        # Check for NaN/Inf
        nan_count = np.sum(np.isnan(data))
        inf_count = np.sum(np.isinf(data))
        if nan_count > 0:
            issues.append(f'{nan_count} NaN values found')
        if inf_count > 0:
            issues.append(f'{inf_count} Inf values found')

        # Check for zero/negative prices
        zero_count = np.sum(data <= 0)
        if zero_count > 0:
            issues.append(f'{zero_count} non-positive values (may be returns, not prices)')

        # Check for constant series
        for j in range(data.shape[1]):
            if np.std(data[:, j]) < 1e-10:
                issues.append(f'Column "{data_dict["names"][j]}" is constant')

        # Check data length
        if data.shape[0] < 30:
            issues.append(f'Only {data.shape[0]} periods (recommend 100+)')

        # Quality score
        score = 100
        score -= min(nan_count * 2, 30)
        score -= min(inf_count * 5, 20)
        score -= min(zero_count, 10)
        if data.shape[0] < 100:
            score -= 10
        if data.shape[0] < 30:
            score -= 20
        score = max(0, score)

        return {
            'valid': len([i for i in issues if 'NaN' in i or 'Inf' in i or 'empty' in i]) == 0,
            'issues': issues,
            'quality_score': score,
            'n_nan': int(nan_count),
            'n_inf': int(inf_count),
            'data_shape': data.shape
        }

    def generate_demo_data(self, n_assets=5, n_periods=500, seed=42):
        """Generate realistic demo market data."""
        np.random.seed(seed)
        names = [f'Asset_{i+1}' for i in range(n_assets)]

        # Generate correlated returns
        mu = np.random.uniform(0.03, 0.15, n_assets) / 252
        sigma = np.random.uniform(0.15, 0.40, n_assets) / np.sqrt(252)

        # Random correlation matrix
        A = np.random.randn(n_assets, n_assets)
        corr = np.corrcoef(A)
        L = np.linalg.cholesky(corr + 0.01 * np.eye(n_assets))

        returns = np.random.randn(n_periods, n_assets) @ L.T
        returns = returns * sigma + mu

        # Generate prices from returns
        prices = 100 * np.exp(np.cumsum(returns, axis=0))

        return {
            'names': names,
            'data': prices,
            'returns': returns,
            'n_assets': n_assets,
            'n_periods': n_periods,
            'format': 'Generated',
            'expected_returns': mu * 252,
            'cov_matrix': np.cov(returns.T),
            'correlation': corr,
        }

    def _is_numeric(self, s):
        try:
            float(s)
            return True
        except (ValueError, TypeError):
            return False
