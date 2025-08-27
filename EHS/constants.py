from colorama import Fore, Style
import os
from dotenv import load_dotenv
load_dotenv()
class Config:
    SHELL_VERSION = "v1.1.1"
    GITHUB_LATEST_UPDATE = "https://raw.githubusercontent.com/QUIK1001/Event-Horizon-Shell/main/check_update"
    CHANGELOG = f"""
{Fore.RED} What's new in {SHELL_VERSION}? {Style.RESET_ALL}
{Fore.YELLOW} Themes {Style.RESET_ALL}
{Fore.YELLOW} Aliases: {Style.RESET_ALL}
  {Fore.GREEN}• show/create/remove aliases {Style.RESET_ALL}
{Fore.YELLOW} Update in Shell: {Style.RESET_ALL}
  {Fore.GREEN}• copy, rename, move {Style.RESET_ALL}
"""
    ENABLE_LOGGING = True
    API_KEY = os.getenv('OPENWEATHER_API_KEY', '')
    BASE_URL = "http://api.openweathermap.org/data/2.5/weather"
    
    THEMES = {
        "default": {
            "prompt": Fore.CYAN,
            "error": Fore.RED, 
            "success": Fore.GREEN,
            "info": Fore.BLUE,
            "warning": Fore.YELLOW,
            "menu": Style.BRIGHT
        },
        "dark": {
            "prompt": Fore.MAGENTA,
            "error": Fore.YELLOW,
            "success": Fore.CYAN,
            "info": Fore.LIGHTBLUE_EX,
            "warning": Fore.LIGHTYELLOW_EX,
            "menu": Style.BRIGHT
        },
        "light": {
            "prompt": Fore.BLUE,
            "error": Fore.RED,
            "success": Fore.GREEN,
            "info": Fore.BLACK,
            "warning": Fore.YELLOW,
            "menu": Style.NORMAL
        },
        "matrix": {
            "prompt": Fore.GREEN,
            "error": Fore.RED,
            "success": Fore.GREEN,
            "info": Fore.GREEN,
            "warning": Fore.YELLOW,
            "menu": Style.BRIGHT
        }
    }
    
    CURRENT_THEME = "default"

    ALIASES = {
        "ls": "dir",
        "cls": "clear",
        "list": "dir",
        "rm": "rmfile",
        "del": "rmfile",
        "delete": "rmfile",
        "mv": "move",
        "cp": "copy",
        "rn": "rename",
    }

config = Config()
