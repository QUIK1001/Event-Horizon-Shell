
import time

from .chklib_clrscr import clear_screen
from .constants import GITHUB_LATEST_UPDATE, SHELL_VERSION

# Check updates
def check_updates():
    """
    Check for updates on GitHub repository.
    
    Compares current version with latest version available online.
    Shows update notification if newer version exists.
    """
    import requests
    from colorama import Fore, Style
    from packaging import version
    try:
        chk_upd_response = requests.get(GITHUB_LATEST_UPDATE, timeout=3)
        chk_upd_response.raise_for_status()
        chk_upd_latest_version = chk_upd_response.text.strip()
        if version.parse(chk_upd_latest_version) > version.parse(SHELL_VERSION):
            print(
                Fore.RED
                + f"Update! {SHELL_VERSION} < {chk_upd_latest_version}"
                + Style.RESET_ALL
            )
            print(
                Fore.CYAN
                + "Download: https://github.com/QUIK1001/Event-Horizon-Shell"
                + Style.RESET_ALL
            )
            return True
        elif version.parse(chk_upd_latest_version) < version.parse(SHELL_VERSION):
            print(Fore.LIGHTBLACK_EX + "Are you from the future? :D" + Style.RESET_ALL)
            time.sleep(0.3)
            raise FutureWarning(
                Fore.RED + "Wait until this version comes out :)" + Style.RESET_ALL
            )
        print(Fore.GREEN + "Actual version" + Style.RESET_ALL)
        return False
    except requests.exceptions.RequestException as e:
        print(Fore.RED + f"Error connecting: {str(e)}" + Style.RESET_ALL)

# Check internet speed
def speed_test():
    """
    Perform internet speed test by downloading a test file.
    
    Measures download speed from a remote server and displays results
    in KB/s with download statistics.
    """
    import requests
    from colorama import Fore, Style
    print(Fore.CYAN + "⣦ Checking internet speed..." + Style.RESET_ALL)
    speed_test_test_url = "https://speedtest.selectel.ru/100MB"
    try:
        speed_test_start_time = time.time()
        speed_test_response = requests.get(speed_test_test_url, timeout=30)
        speed_test_response.raise_for_status()
        speed_test_end_time = time.time()
        speed_test_downloaded = len(speed_test_response.content)
        total_time = max(0.1, speed_test_end_time - speed_test_start_time)
        speed_kbs = speed_test_downloaded / total_time / 1024
        print(
            Fore.GREEN
            + f"\nDownloaded: {speed_test_downloaded / 1024:.1f} KB"
            + Style.RESET_ALL
        )
        print(Fore.GREEN + f"Time: {total_time:.2f} seconds" + Style.RESET_ALL)
        print(Fore.GREEN + f"Speed: {speed_kbs:.1f} KB/s" + Style.RESET_ALL)
    except requests.exceptions.Timeout:
        print(
            Fore.RED
            + "\n⣏!⣽ Connection timeout - check your internet connection"
            + Style.RESET_ALL
        )
    except requests.exceptions.ConnectionError:
        print(
            Fore.RED
            + "\n⣏!⣽ Connection error - check your internet connection"
            + Style.RESET_ALL
        )
    except Exception as e:
        print(Fore.RED + f"\n⣏!⣽ Error: {str(e)}" + Style.RESET_ALL)
    input("Press Enter to continue")
    clear_screen()
