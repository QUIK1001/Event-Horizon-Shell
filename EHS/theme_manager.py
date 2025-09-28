from colorama import Fore, Style

_ORIGINAL_FORE = Fore.__dict__.copy()
_ORIGINAL_STYLE = Style.__dict__.copy()

def apply_theme(config):
    theme = config.THEMES[config.CURRENT_THEME]

    Fore.RED = theme["error"]
    Fore.GREEN = theme["success"]
    Fore.BLUE = theme["info"]
    Fore.YELLOW = theme["warning"]
    Fore.CYAN = theme["prompt"]
    Fore.MAGENTA = theme.get("special", _ORIGINAL_FORE["MAGENTA"])
    Fore.WHITE = theme.get("normal", _ORIGINAL_FORE["WHITE"])

    Style.BRIGHT = theme["menu"]

def reset_theme():
    for attr, value in _ORIGINAL_FORE.items():
        setattr(Fore, attr, value)
    for attr, value in _ORIGINAL_STYLE.items():
        setattr(Style, attr, value)
