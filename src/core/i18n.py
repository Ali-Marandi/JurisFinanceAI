import json
import os


class TranslationManager:
    """Internationalization (i18n) manager for JurisFinanceAI.

    Supports Persian (Farsi), English, and Arabic.
    Dynamic language switching without restart.
    Falls back to English for missing translations.
    """

    LANGUAGES = {
        'fa': {'name': 'فارسی', 'dir': 'rtl', 'native': 'فارسی'},
        'en': {'name': 'English', 'dir': 'ltr', 'native': 'English'},
        'ar': {'name': 'العربية', 'dir': 'rtl', 'native': 'العربية'},
    }

    DEFAULT_LANGUAGE = 'fa'

    def __init__(self, lang=None):
        self._current = lang or self.DEFAULT_LANGUAGE
        self._translations = self._load_all()

    def _load_all(self):
        """Load translations for all languages."""
        translations = {}
        translations['fa'] = self._persian()
        translations['en'] = self._english()
        translations['ar'] = self._arabic()
        return translations

    def _persian(self):
        return {
            # App
            'app_title': 'جوریس فایننس ای‌آی',
            'version': 'نسخه',
            'dashboard': 'داشبورد',
            'settings': 'تنظیمات',
            'exit': 'خروج',
            # Dashboard tabs
            'tab_portfolio': 'بهره‌برداری سبد',
            'tab_derivatives': 'ابزار مشتق',
            'tab_risk': 'مدیریت ریسک',
            'tab_timeseries': 'سری زمانی',
            'tab_fuzzy': 'فازی',
            'tab_network': 'شبکه',
            'tab_behavioral': 'رفتاری',
            'tab_montecarlo': 'مونت‌کارلو',
            'tab_interest': 'نرخ بهره',
            'tab_topological': 'توپولوژیکی',
            'tab_generative': 'ژنراتیو',
            'tab_explainability': 'تفسیرپذیری',
            'tab_quantum': 'کوانتومی',
            'tab_nlp': 'پردازش زبان',
            'tab_gpu': 'شتاب‌گردی',
            # Actions
            'demo': 'دمو',
            'execute': 'اجرا',
            'import_data': 'وارد کردن داده',
            'export_report': 'خروجی گزارش',
            'clear': 'پاک‌سازی',
            'search': 'جستجو',
            'refresh': 'بازخوانی',
            # File
            'open_file': 'باز کردن فایل',
            'save_report': 'ذخیره گزارش',
            'csv_files': 'فایل‌های CSV',
            'excel_files': 'فایل‌های اکسل',
            'pdf_files': 'فایل‌های PDF',
            'all_files': 'همه فایل‌ها',
            # Analysis
            'results': 'نتایج',
            'chart': 'نمودار',
            'table': 'جدول',
            'no_results': 'نتیجه‌ای وجود ندارد',
            'running': 'در حال اجرا...',
            'completed': 'تکمیل شد',
            'error': 'خطا',
            'success': 'موفقیت',
            # Portfolio
            'optimization_method': 'روش بهینه‌سازی',
            'risk_free_rate': 'نرخ بدون ریسک',
            'expected_return': 'بازده مورد انتظار',
            'volatility': 'نوسان‌پذیری',
            'sharpe_ratio': 'نسبت شارپ',
            'efficient_frontier': 'مرز کارا',
            # Risk
            'confidence_level': 'سطح اطمینان',
            'value_at_risk': 'ارزش در معرض ریسک',
            'stress_test': 'تست استرس',
            'max_drawdown': 'بیشینه افت',
            # Data
            'live_data': 'داده زنده',
            'symbol': 'نماد',
            'period': 'بازه زمانی',
            'fetch': 'دریافت',
            'offline_mode': 'حالت آفلاین',
            # Plugin
            'plugins': 'افزونه‌ها',
            'enabled': 'فعال',
            'disabled': 'غیرفعال',
            'install': 'نصب',
            'uninstall': 'حذف',
            # Language
            'language': 'زبان',
            'persian': 'فارسی',
            'english': 'انگلیسی',
            'arabic': 'عربی',
            # Quantum
            'n_qubits': 'تعداد کیوبیت‌ها',
            'n_layers': 'تعداد لایه‌ها',
            'qaoa': 'الگوریتم QAOA',
            'quantum_mc': 'مونت‌کارلو کوانتومی',
            # General
            'ok': 'تایید',
            'cancel': 'انصراف',
            'close': 'بستن',
            'help': 'راهنما',
            'about': 'درباره',
        }

    def _english(self):
        return {
            'app_title': 'JurisFinanceAI',
            'version': 'Version',
            'dashboard': 'Dashboard',
            'settings': 'Settings',
            'exit': 'Exit',
            'tab_portfolio': 'Portfolio',
            'tab_derivatives': 'Derivatives',
            'tab_risk': 'Risk Management',
            'tab_timeseries': 'Time Series',
            'tab_fuzzy': 'Fuzzy Systems',
            'tab_network': 'Network',
            'tab_behavioral': 'Behavioral',
            'tab_montecarlo': 'Monte Carlo',
            'tab_interest': 'Interest Rates',
            'tab_topological': 'Topological',
            'tab_generative': 'Generative',
            'tab_explainability': 'Explainability',
            'tab_quantum': 'Quantum',
            'tab_nlp': 'NLP',
            'tab_gpu': 'GPU Acceleration',
            'demo': 'Demo',
            'execute': 'Execute',
            'import_data': 'Import Data',
            'export_report': 'Export Report',
            'clear': 'Clear',
            'search': 'Search',
            'refresh': 'Refresh',
            'open_file': 'Open File',
            'save_report': 'Save Report',
            'csv_files': 'CSV Files',
            'excel_files': 'Excel Files',
            'pdf_files': 'PDF Files',
            'all_files': 'All Files',
            'results': 'Results',
            'chart': 'Chart',
            'table': 'Table',
            'no_results': 'No results',
            'running': 'Running...',
            'completed': 'Completed',
            'error': 'Error',
            'success': 'Success',
            'optimization_method': 'Optimization Method',
            'risk_free_rate': 'Risk-Free Rate',
            'expected_return': 'Expected Return',
            'volatility': 'Volatility',
            'sharpe_ratio': 'Sharpe Ratio',
            'efficient_frontier': 'Efficient Frontier',
            'confidence_level': 'Confidence Level',
            'value_at_risk': 'Value at Risk',
            'stress_test': 'Stress Test',
            'max_drawdown': 'Max Drawdown',
            'live_data': 'Live Data',
            'symbol': 'Symbol',
            'period': 'Period',
            'fetch': 'Fetch',
            'offline_mode': 'Offline Mode',
            'plugins': 'Plugins',
            'enabled': 'Enabled',
            'disabled': 'Disabled',
            'install': 'Install',
            'uninstall': 'Uninstall',
            'language': 'Language',
            'persian': 'Persian',
            'english': 'English',
            'arabic': 'Arabic',
            'n_qubits': 'Qubits',
            'n_layers': 'Layers',
            'qaoa': 'QAOA Algorithm',
            'quantum_mc': 'Quantum Monte Carlo',
            'ok': 'OK',
            'cancel': 'Cancel',
            'close': 'Close',
            'help': 'Help',
            'about': 'About',
        }

    def _arabic(self):
        return {
            'app_title': 'جوريس فاينانس',
            'version': 'الإصدار',
            'dashboard': 'لوحة القيادة',
            'settings': 'الإعدادات',
            'exit': 'خروج',
            'tab_portfolio': 'المحفظة',
            'tab_derivatives': 'المشتقات',
            'tab_risk': 'إدارة المخاطر',
            'tab_timeseries': 'السلاسل الزمنية',
            'tab_fuzzy': 'الضبابي',
            'tab_network': 'الشبكة',
            'tab_behavioral': 'السلوكي',
            'tab_montecarlo': 'مونت كارلو',
            'tab_interest': 'أسعار الفائدة',
            'tab_topological': 'الطوبولوجي',
            'tab_generative': 'التوليدي',
            'tab_explainability': 'القابلية للتفسير',
            'tab_quantum': 'الكمي',
            'tab_nlp': 'معالجة اللغة',
            'tab_gpu': 'تسريع GPU',
            'demo': 'عرض تجريبي',
            'execute': 'تنفيذ',
            'import_data': 'استيراد البيانات',
            'export_report': 'تصدير التقرير',
            'clear': 'مسح',
            'search': 'بحث',
            'refresh': 'تحديث',
            'open_file': 'فتح ملف',
            'save_report': 'حفظ التقرير',
            'csv_files': 'ملفات CSV',
            'excel_files': 'ملفات إكسل',
            'pdf_files': 'ملفات PDF',
            'all_files': 'جميع الملفات',
            'results': 'النتائج',
            'chart': 'الرسم البياني',
            'table': 'الجدول',
            'no_results': 'لا توجد نتائج',
            'running': 'جارٍ التشغيل...',
            'completed': 'مكتمل',
            'error': 'خطأ',
            'success': 'نجاح',
            'optimization_method': 'طريقة التحسين',
            'risk_free_rate': 'معدل خالي من المخاطر',
            'expected_return': 'العائد المتوقع',
            'volatility': 'التقلب',
            'sharpe_ratio': 'نسبة شارب',
            'efficient_frontier': 'الحدود الكفؤة',
            'confidence_level': 'مستوى الثقة',
            'value_at_risk': 'القيمة المعرضة للخطر',
            'stress_test': 'اختبار الإجهاد',
            'max_drawdown': 'أقصى انخفاض',
            'live_data': 'بيانات حية',
            'symbol': 'الرمز',
            'period': 'الفترة',
            'fetch': 'جلب',
            'offline_mode': 'وضع عدم الاتصال',
            'plugins': 'الإضافات',
            'enabled': 'مفعّل',
            'disabled': 'معطّل',
            'install': 'تثبيت',
            'uninstall': 'إلغاء التثبيت',
            'language': 'اللغة',
            'persian': 'الفارسية',
            'english': 'الإنجليزية',
            'arabic': 'العربية',
            'n_qubits': 'الكيوبتات',
            'n_layers': 'الطبقات',
            'qaoa': 'خوارزمية QAOA',
            'quantum_mc': 'مونت كارلو الكمي',
            'ok': 'موافق',
            'cancel': 'إلغاء',
            'close': 'إغلاق',
            'help': 'مساعدة',
            'about': 'حول',
        }

    @property
    def current_language(self):
        return self._current

    @property
    def current_direction(self):
        return self.LANGUAGES.get(self._current, {}).get('dir', 'ltr')

    def set_language(self, lang):
        if lang in self.LANGUAGES:
            self._current = lang
            return True
        return False

    def t(self, key, lang=None):
        """Translate a key to the current (or specified) language."""
        target = lang or self._current
        translations = self._translations.get(target, self._translations['en'])
        return translations.get(key, self._translations['en'].get(key, key))

    def available_languages(self):
        return list(self.LANGUAGES.keys())

    def get_language_info(self, lang):
        return self.LANGUAGES.get(lang, {})
