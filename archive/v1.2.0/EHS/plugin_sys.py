import os
import importlib
import importlib.util
import sys
from colorama import Fore, Style
from .chklib_clrscr import clear_screen
from .logger import logger

class PluginSystem:
    def __init__(self):
        if getattr(sys, 'frozen', False):
            self.plugins_dir = os.path.dirname(sys.executable)
        else:
            self.plugins_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.loaded_plugins = {}
        self.excluded_files = {
            '__init__.py', '__pycache__', 'main.py', 'constants.py',
            'chklib_clrscr.py', 'chkupd_speedtest.py', 'logger.py',
            'aliases.py', 'archiver.py', 'art.py', 'calc.py', 'editors.py',
            'shell.py', 'theme_manager.py', 'tools.py', 'tutorial.py',
            'weather.py', 'plugin_sys.py'
        }
    
    def discover_plugins(self):
        """Discover all Python files that can be plugins"""
        plugins = {}
        
        try:
            if not os.path.exists(self.plugins_dir):
                logger.log(f"Plugins directory not found: {self.plugins_dir}", "WARN")
                return plugins
            
            for filename in os.listdir(self.plugins_dir):
                if (filename.endswith('.py') and 
                    filename not in self.excluded_files and
                    not filename.startswith('_')):
                    
                    plugin_name = filename[:-3]
                    plugins[plugin_name] = {
                        'filename': filename,
                        'full_path': os.path.join(self.plugins_dir, filename),
                        'display_name': plugin_name.replace('_', ' ').title()
                    }
                    logger.log(f"Discovered plugin: {plugin_name}", "DEBUG")
        
        except Exception as e:
            logger.log(f"Error discovering plugins: {str(e)}", "FAIL")
        
        return plugins
    
    def load_plugin(self, plugin_name):
        try:
            if plugin_name in self.loaded_plugins:
                return self.loaded_plugins[plugin_name]
            
            plugin_info = self.discover_plugins().get(plugin_name)
            if not plugin_info:
                logger.log(f"Plugin not found: {plugin_name}", "FAIL")
                return None
            
            if getattr(sys, 'frozen', False):
                spec = importlib.util.spec_from_file_location(plugin_name, plugin_info['full_path'])
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            else:
                module = importlib.import_module(f"EHS.{plugin_name}")
            
            if hasattr(module, 'main'):
                self.loaded_plugins[plugin_name] = {
                    'module': module,
                    'info': plugin_info
                }
                logger.log(f"Plugin loaded successfully: {plugin_name}", "OK")
                return self.loaded_plugins[plugin_name]
            else:
                logger.log(f"Plugin has no main() function: {plugin_name}", "FAIL")
                return None
                
        except Exception as e:
            logger.log(f"Error loading plugin {plugin_name}: {str(e)}", "FAIL")
            return None
    
    def run_plugin(self, plugin_name):
        plugin = self.load_plugin(plugin_name)
        if not plugin:
            print(f"{Fore.RED}Failed to load plugin: {plugin_name}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Make sure {plugin_name}.py is in the same directory as the EXE{Style.RESET_ALL}")
            return False
        
        try:
            print(f"{Fore.CYAN}Starting plugin: {plugin['info']['display_name']}{Style.RESET_ALL}")
            plugin['module'].main()
            logger.log(f"Plugin executed successfully: {plugin_name}", "OK")
            return True
        except Exception as e:
            logger.log(f"Error running plugin {plugin_name}: {str(e)}", "FAIL")
            print(f"{Fore.RED}Plugin execution failed: {str(e)}{Style.RESET_ALL}")
            return False

def plugin_system_menu():
    plugin_sys = PluginSystem()
    
    while True:
        clear_screen()
        print(f"{Fore.CYAN}=== Plugin System (Beta) ==={Style.RESET_ALL}")
        
        print(f"{Fore.YELLOW}Searching plugins in: {plugin_sys.plugins_dir}{Style.RESET_ALL}")
        
        plugins = plugin_sys.discover_plugins()
        
        if not plugins:
            print(f"\n{Fore.RED}No plugins found!{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}To add plugins:{Style.RESET_ALL}")
            print(f"1. Create .py files in: {plugin_sys.plugins_dir}")
            print(f"2. Each file must contain a main() function")
            print(f"3. Restart EHS to detect new plugins")
        else:
            print(f"\n{Fore.YELLOW}Available plugins:{Style.RESET_ALL}")
            for i, (plugin_name, plugin_info) in enumerate(plugins.items(), 1):
                status = f"{Fore.GREEN}[LOADED]" if plugin_name in plugin_sys.loaded_plugins else f"{Fore.YELLOW}[AVAILABLE]"
                print(f"{i}. {plugin_info['display_name']} {status}{Style.RESET_ALL}")
        
        print(f"\n{Fore.YELLOW}0. Back to main menu{Style.RESET_ALL}")
        
        try:
            choice = input(f"{Fore.GREEN}Select plugin (1-{len(plugins)}): {Style.RESET_ALL}").strip()
            
            if choice == "0":
                logger.log("Exiting plugin system", "INFO")
                break
            
            if choice.isdigit():
                choice_num = int(choice)
                if 1 <= choice_num <= len(plugins):
                    plugin_name = list(plugins.keys())[choice_num - 1]
                    
                    if plugin_sys.run_plugin(plugin_name):
                        input(f"{Fore.BLUE}Press Enter to continue...{Style.RESET_ALL}")
                    else:
                        input(f"{Fore.RED}Press Enter to continue...{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}Invalid selection!{Style.RESET_ALL}")
                    input("Press Enter to continue...")
            else:
                print(f"{Fore.RED}Please enter a number!{Style.RESET_ALL}")
                input("Press Enter to continue...")
                
        except KeyboardInterrupt:
            print(f"{Fore.YELLOW}\nReturning to main menu...{Style.RESET_ALL}")
            break
        except Exception as e:
            logger.log(f"Error in plugin menu: {str(e)}", "FAIL")
            print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
            input("Press Enter to continue...")

def main():
    plugin_system_menu()

plugin_system = PluginSystem()
