from colorama import Fore, Style

class Config:
    SHELL_VERSION = "v1.2.1"
    GITHUB_LATEST_UPDATE = "https://raw.githubusercontent.com/QUIK1001/Event-Horizon-Shell/main/check_update"
    CHANGELOG = f"""
{Fore.RED} What's new in {SHELL_VERSION}? {Style.RESET_ALL}
{Fore.YELLOW} Shell:
  {Fore.GREEN}• grep
  • wc{Style.RESET_ALL}
"""
    ENABLE_LOGGING = False
    API_KEY = "22665d0c08a6c4891dd1cf15717ce820"
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
