import time

from .chklib_clrscr import clear_screen
from .constants import config
from .logger import logger

# Check updates
def check_updates():
    """
    Check for updates on GitHub repository.
    """
    import requests
    from colorama import Fore, Style
    from packaging import version

    logger.log("Checking for updates", "INFO")

    try:
        chk_upd_response = requests.get(config.GITHUB_LATEST_UPDATE, timeout=3)
        chk_upd_response.raise_for_status()
        chk_upd_latest_version = chk_upd_response.text.strip()

        logger.log(f"Current version: {config.SHELL_VERSION}, Latest version: {chk_upd_latest_version}", "DEBUG")

        if version.parse(chk_upd_latest_version) > version.parse(config.SHELL_VERSION):
            logger.log(f"Update available: {config.SHELL_VERSION} -> {chk_upd_latest_version}", "WARN")
            print(
                Fore.RED
                + f"Update! {config.SHELL_VERSION} < {chk_upd_latest_version}"
                + Style.RESET_ALL
            )
            print(
                Fore.CYAN
                + "GitHub: https://github.com/QUIK1001/Event-Horizon-Shell"
                + Style.RESET_ALL
            )
            print(
                Fore.CYAN
                + "Telegram: https://t.me/Event_Horizon_Shell"
                + Style.RESET_ALL
            )

            download_choice = input(f"{Fore.YELLOW}Download update from Telegram? (Y/N): {Style.RESET_ALL}").upper()
            if download_choice == "Y":
                print(f"{Fore.GREEN}Opening Telegram channel...{Style.RESET_ALL}")
                import webbrowser
                webbrowser.open("https://t.me/Event_Horizon_Shell")
            return True

        elif version.parse(chk_upd_latest_version) < version.parse(config.SHELL_VERSION):
            logger.log("Current version is newer than latest release", "DEBUG")
            print(Fore.LIGHTBLACK_EX + "Are you from the future? :D" + Style.RESET_ALL)
            time.sleep(0.3)
            raise FutureWarning(
                Fore.RED + "Wait until this version comes out :)" + Style.RESET_ALL
            )

        logger.log("Shell is up to date", "OK")
        print(Fore.GREEN + "Actual version" + Style.RESET_ALL)
        return False

    except requests.exceptions.RequestException as e:
        logger.log(f"Error checking for updates: {str(e)}", "FAIL")
        print(Fore.RED + f"Error connecting: {str(e)}" + Style.RESET_ALL)

        print(f"{Fore.YELLOW}Check updates in Telegram: https://t.me/Event_Horizon_Shell{Style.RESET_ALL}")

    except Exception as e:
        logger.log(f"Unexpected error during update check: {str(e)}", "FAIL")
        print(Fore.RED + f"Error: {str(e)}" + Style.RESET_ALL)


# Progress bar
def _progress_bar(current, total, bar_length=50):
    from colorama import Fore, Style

    percent = float(current) * 100 / total
    arrow = '=' * int(percent / 100 * bar_length - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    if percent < 30:
        color = Fore.RED
    elif percent < 70:
        color = Fore.YELLOW
    else:
        color = Fore.GREEN

    return f"{color}[{arrow}{spaces}] {percent:.1f}%{Style.RESET_ALL}"


# Check internet speed
def speed_test():
    """
    Perform internet speed test by downloading a test file.
    
    Measures download speed from a remote server and displays results
    in KB/s with download statistics and progress bar.
    """
    import requests
    from colorama import Fore, Style

    logger.log("Starting internet speed test", "INFO")
    print(Fore.CYAN + "⣦ Checking internet speed..." + Style.RESET_ALL)

    speed_test_test_url = "https://speedtest.selectel.ru/100MB"
    try:
        speed_test_start_time = time.time()

        with requests.get(speed_test_test_url, stream=True, timeout=30) as response:
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            chunk_size = 8192

            print(f"\nFile size: {total_size / 1024 / 1024:.1f} MB")
            print("Downloading...\n")

            for chunk in response.iter_content(chunk_size=chunk_size):
                downloaded += len(chunk)

                if downloaded % (chunk_size * 10) == 0 or downloaded == total_size:
                    print(f"\r{_progress_bar(downloaded, total_size)}", end="", flush=True)

        speed_test_end_time = time.time()
        total_time = max(0.1, speed_test_end_time - speed_test_start_time)
        speed_kbs = downloaded / total_time / 1024
        speed_mbs = speed_kbs / 1024

        logger.log(f"Speed test completed: {speed_kbs:.1f} KB/s ({speed_mbs:.1f} MB/s)", "OK")
        print(f"\n\n{Fore.GREEN}Downloaded: {downloaded / 1024 / 1024:.1f} MB{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Time: {total_time:.2f} seconds{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Speed: {speed_kbs:.1f} KB/s ({speed_mbs:.1f} MB/s){Style.RESET_ALL}")

    except requests.exceptions.Timeout:
        logger.log("Speed test timeout - connection too slow", "FAIL")
        print(
            Fore.RED
            + "\n⣏!⣽ Connection timeout - check your internet connection"
            + Style.RESET_ALL
        )
    except requests.exceptions.ConnectionError:
        logger.log("Speed test connection error - no internet", "FAIL")
        print(
            Fore.RED
            + "\n⣏!⣽ Connection error - check your internet connection"
            + Style.RESET_ALL
        )
    except Exception as e:
        logger.log(f"Unexpected error during speed test: {str(e)}", "FAIL")
        print(Fore.RED + f"\n⣏!⣽ Error: {str(e)}" + Style.RESET_ALL)

    input("\nPress Enter to continue")
    clear_screen()
