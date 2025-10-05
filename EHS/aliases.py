from .constants import config
from .logger import logger

def resolve_alias(command):
    """Convert alias to real command"""
    parts = command.strip().split()
    if not parts:
        return command

    alias = parts[0]
    if alias in config.ALIASES:
        resolved = config.ALIASES[alias]
        if len(parts) > 1:
            resolved += " " + " ".join(parts[1:])
        logger.log(f"Alias resolved: {alias} -> {resolved}", "DEBUG")
        return resolved

    return command

def show_aliases():
    """Show all aliases"""
    from colorama import Fore, Style

    if not config.ALIASES:
        print(Fore.YELLOW + "No aliases defined" + Style.RESET_ALL)
        return

    print(Fore.CYAN + "⣦ Command aliases:" + Style.RESET_ALL)
    for alias, command in config.ALIASES.items():
        print(f"{Fore.GREEN}{alias:10}{Style.RESET_ALL} -> {command}")

def add_alias(alias, command):
    """Add new alias"""
    from colorama import Fore, Style

    if not alias or not command:
        print(Fore.RED + "Usage: alias add <alias> <command>" + Style.RESET_ALL)
        return

    config.ALIASES[alias] = command
    print(Fore.GREEN + f"Alias added: {alias} -> {command}" + Style.RESET_ALL)
    logger.log(f"Alias added: {alias} -> {command}", "INFO")

def remove_alias(alias):
    """Remove alias"""
    from colorama import Fore, Style

    if alias in config.ALIASES:
        del config.ALIASES[alias]
        print(Fore.GREEN + f"Alias removed: {alias}" + Style.RESET_ALL)
        logger.log(f"Alias removed: {alias}", "INFO")
    else:
        print(Fore.RED + f"Alias not found: {alias}" + Style.RESET_ALL)
