from colorama import Fore, Style

SHELL_VERSION = "v1.0.9"
GITHUB_LATEST_UPDATE = (
    "https://raw.githubusercontent.com/QUIK1001/Event-Horizon-Shell/main/check_update"
)
CHANGELOG = f"""
{Fore.RED} What's new in {SHELL_VERSION}? {Style.RESET_ALL}
{Fore.YELLOW} Modular structure: {Style.RESET_ALL}
  {Fore.GREEN}• menu.py - main menu
  • shell.py - built-in shell
  • tools.py - tools (timer, echo, ASCII arts, shell info, shutdown, os info)
  • calc.py - calculator
  • chkupd_speedtest.py - check updates & internet speed
  • constants.py - version, URL constants
  • art.py - loading screen
  • chklib_clrscr.py - check dependencies & screen cleaning {Style.RESET_ALL}
  {Fore.YELLOW} OS info: {Style.RESET_ALL}
  {Fore.GREEN}• Shows system information:
    • System information
    • CPU cores and frequency
    • RAM size 
    • Python version
    • System architecture {Style.RESET_ALL}
"""
