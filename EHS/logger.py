import os
from datetime import datetime
from colorama import Fore, Style
from . import constants

class Logger:
    def __init__(self):
        self.log_file = self._get_log_file_path()
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
    def _get_log_file_path(self):
        home_dir = os.path.expanduser("~")
        return os.path.join(home_dir, ".ehs_logs", "ehs_log.txt")
        
    def log(self, message, status="INFO", exception=None):
        if not constants.config.ENABLE_LOGGING:
            return
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        status_colors = {
            "OK": Fore.GREEN,
            "FAIL": Fore.RED,
            "INFO": Fore.BLUE,
            "WARN": Fore.YELLOW,
            "DEBUG": Fore.MAGENTA
        }
        
        color = status_colors.get(status, Fore.WHITE)
        
        full_message = f"{message}: {exception}" if exception else message
        
        console_message = f"{Style.DIM}[{timestamp}]{Style.RESET_ALL} {full_message} {color}[{status}]{Style.RESET_ALL}"
        
        file_message = f"[{timestamp}] [{status}] {full_message}"

        print(console_message)

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(file_message + "\n")

logger = Logger()
