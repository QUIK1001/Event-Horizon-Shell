from colorama import Fore, Style
class Config:
    SHELL_VERSION = "v1.1.0"
    GITHUB_LATEST_UPDATE = "https://raw.githubusercontent.com/QUIK1001/Event-Horizon-Shell/main/check_update"
    CHANGELOG = f"""
{Fore.RED} What's new in {SHELL_VERSION}? {Style.RESET_ALL}
{Fore.YELLOW} Logs: {Style.RESET_ALL}
  {Fore.GREEN}• Shows detailed logs everywhere :)
    • It is possible to disable it using function 11 under the name "toggle logs"{Style.RESET_ALL}
{Fore.YELLOW} Weather: {Style.RESET_ALL}
  {Fore.GREEN}• Shows the weather {Style.RESET_ALL}
{Fore.YELLOW} Text editors: {Style.RESET_ALL}
  {Fore.GREEN}• Can create\open a text file using the default text editor {Style.RESET_ALL}
"""
    ENABLE_LOGGING = True
    API_KEY = "22665d0c08a6c4891dd1cf15717ce820"
    BASE_URL = "http://api.openweathermap.org/data/2.5/weather"

config = Config()
