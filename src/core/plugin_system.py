import os
import re
import importlib
import importlib.util
import json
from datetime import datetime


class Plugin:
    """Base class for JurisFinanceAI plugins.

    All plugins must inherit from this class and implement:
    - name(): Plugin identifier
    - description(): Plugin description
    - version(): Plugin version string
    - run_analysis(data): Main analysis method
    - get_tab_widget(): Return PyQt6 widget (optional)
    """

    @staticmethod
    def name():
        raise NotImplementedError

    @staticmethod
    def description():
        raise NotImplementedError

    @staticmethod
    def version():
        return '4.1.0'

    def run_analysis(self, data):
        raise NotImplementedError

    def get_tab_widget(self):
        return None

    def get_metadata(self):
        return {
            'name': self.name(),
            'description': self.description(),
            'version': self.version(),
            'type': self.__class__.__name__,
        }


class PluginManager:
    """Microkernel plugin system for JurisFinanceAI.

    Manages plugin lifecycle: discovery, loading, enabling/disabling,
    and execution. Plugins can be Python files or packages in the
    plugins/ directory.
    """

    def __init__(self, app_dir=None):
        self.app_dir = app_dir or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.plugin_dir = os.path.join(self.app_dir, 'plugins')
        self.plugins = {}
        self.plugin_classes = {}
        self.enabled = set()
        self._config_path = os.path.join(self.app_dir, 'plugin_config.json')
        self._load_config()

    def _load_config(self):
        """Load plugin configuration from JSON file."""
        if os.path.exists(self._config_path):
            try:
                with open(self._config_path, 'r') as f:
                    config = json.load(f)
                    self.enabled = set(config.get('enabled', []))
            except Exception:
                self.enabled = set()

    def _save_config(self):
        """Save plugin configuration."""
        config = {'enabled': list(self.enabled), 'version': '1.0'}
        try:
            with open(self._config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception:
            pass

    def discover_plugins(self):
        """Scan plugin directory and discover available plugins."""
        discovered = []

        if not os.path.exists(self.plugin_dir):
            os.makedirs(self.plugin_dir, exist_ok=True)
            return discovered

        for item in os.listdir(self.plugin_dir):
            item_path = os.path.join(self.plugin_dir, item)

            if item.endswith('.py') and not item.startswith('_'):
                discovered.append({
                    'path': item_path,
                    'name': item[:-3],
                    'type': 'single_file'
                })
            elif os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, '__init__.py')):
                discovered.append({
                    'path': item_path,
                    'name': item,
                    'type': 'package'
                })

        return discovered

    def load_plugin(self, plugin_info):
        """Load a plugin from file path."""
        try:
            if plugin_info['type'] == 'single_file':
                spec = importlib.util.spec_from_file_location(
                    plugin_info['name'], plugin_info['path']
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            else:
                module = importlib.import_module(f'plugins.{plugin_info["name"]}')

            # Find Plugin subclasses
            plugin_classes = []
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and issubclass(attr, Plugin)
                        and attr is not Plugin and attr.name is not None):
                    plugin_classes.append(attr)

            for cls in plugin_classes:
                name = cls.name()
                self.plugin_classes[name] = cls
                instance = cls()
                self.plugins[name] = instance
                if name not in self.enabled:
                    self.enabled.add(name)

            self._save_config()
            return True

        except Exception as e:
            return False

    def load_all(self):
        """Discover and load all available plugins."""
        discovered = self.discover_plugins()
        loaded = 0
        for info in discovered:
            if self.load_plugin(info):
                loaded += 1
        return loaded

    def unload_plugin(self, name):
        """Unload a plugin by name."""
        if name in self.plugins:
            del self.plugins[name]
        if name in self.plugin_classes:
            del self.plugin_classes[name]
        self.enabled.discard(name)
        self._save_config()

    def enable_plugin(self, name):
        self.enabled.add(name)
        self._save_config()

    def disable_plugin(self, name):
        self.enabled.discard(name)
        self._save_config()

    def is_enabled(self, name):
        return name in self.enabled

    def list_plugins(self):
        """List all loaded plugins with metadata."""
        return [p.get_metadata() for p in self.plugins.values()]

    def run_plugin(self, name, data):
        """Execute a plugin's analysis."""
        if name not in self.plugins:
            return {'error': f'Plugin "{name}" not found'}
        if not self.is_enabled(name):
            return {'error': f'Plugin "{name}" is disabled'}
        try:
            return self.plugins[name].run_analysis(data)
        except Exception as e:
            return {'error': str(e)}

    def get_tab_widgets(self):
        """Get PyQt6 tab widgets from all plugins."""
        widgets = []
        for name, plugin in self.plugins.items():
            if self.is_enabled(name):
                widget = plugin.get_tab_widget()
                if widget is not None:
                    widgets.append((name, widget))
        return widgets

    def create_plugin_template(self, name, description=''):
        """Generate a template file for a new plugin."""
        import re
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '', name)
        if not safe_name:
            raise ValueError(f"Invalid plugin name: {name}")

        if not os.path.exists(self.plugin_dir):
            os.makedirs(self.plugin_dir, exist_ok=True)

        template = f'''"""{safe_name} plugin for JurisFinanceAI."""
import numpy as np
from src.core.plugin_system import Plugin


class {safe_name}Plugin(Plugin):
    @staticmethod
    def name():
        return "{safe_name.lower()}"

    @staticmethod
    def description():
        return "{description or safe_name + ' analysis plugin'}"

    @staticmethod
    def version():
        return "1.0.0"

    def run_analysis(self, data):
        """Run {safe_name} analysis on the provided data."""
        # data is a dict with 'returns', 'prices', etc.
        returns = data.get('returns', np.array([]))
        results = {{}}
        # Your analysis logic here
        results['summary'] = f'Analyzed {{len(returns)}} data points'
        return results

    def get_tab_widget(self):
        # Return a PyQt6 QWidget for the dashboard
        # Return None if no UI is needed
        return None
'''

        filepath = os.path.join(self.plugin_dir, f'{safe_name.lower()}.py')
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(template)

        return filepath
