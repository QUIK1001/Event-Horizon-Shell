import random
import sys
import time

from .chklib_clrscr import clear_screen
from .constants import config
from .logger import logger

# Timer
def timer():
    """
    Countdown timer with seconds input.
    """
    from colorama import Fore, Style

    logger.log("Starting timer", "INFO")

    try:
        timer_time = int(input("enter clock time in seconds> "))
        logger.log(f"Timer set for {timer_time} seconds", "INFO")

        print(timer_time)
        for i in range(timer_time):
            time.sleep(1)
            print(timer_time - i - 1)

        logger.log("Timer finished", "OK")
        print(Fore.GREEN + "Timer finished!" + Style.RESET_ALL)
        input("Press Enter to continue")
        clear_screen()

    except ValueError as e:
        logger.log("Invalid input for timer", "FAIL", e)
        print(Fore.RED + "⣏!⣽ Please enter a valid number!" + Style.RESET_ALL)
    except Exception as e:
        logger.log("Unexpected error in timer", "FAIL", e)
        print(Fore.RED + f"⣏!⣽ Error: {e}" + Style.RESET_ALL)


# Changelog
def changelog():
    """
    Display changelog and version information.
    """
    from colorama import Fore, Style
    from art import text2art

    logger.log("Displaying changelog", "INFO")

    try:
        clear_screen()
        print(text2art(config.SHELL_VERSION))
        print(Fore.RED + config.CHANGELOG + Style.RESET_ALL)
        logger.log("Changelog displayed successfully", "OK")
        input("Press Enter to continue")
        clear_screen()

    except Exception as e:
        logger.log("Error displaying changelog", "FAIL", e)
        print(Fore.RED + f"⣏!⣽ Error: {e}" + Style.RESET_ALL)


# Echo
def echo():
    """
    Simple echo command - repeats user input.
    """
    logger.log("Starting echo function", "INFO")

    try:
        echo_text = input("⣦Enter your text: ")
        logger.log(f"Echoing text: {echo_text}", "INFO")
        print("⣦Your text:", echo_text)
        logger.log("Echo completed", "OK")
        input("Press Enter to continue")
        clear_screen()

    except Exception as e:
        logger.log("Error in echo function", "FAIL", e)
        print(Fore.RED + f"⣏!⣽ Error: {e}" + Style.RESET_ALL)


# ASCII arts
def ascii_arts():
    """
    Display random ASCII art from collection.
    """
    logger.log("Displaying ASCII art", "INFO")

    try:
        cats = [
            """ /|_/|
( o.o )
 > ^ < """,
            """  /|_/|
 =( °w° )=
  )   (  """,
        ]
        galaxie = [
"         .                      .    \n"
"         .     `                ;    \n"
"         :                  - --+-- -\n"
"         !           .          !    \n"
"    '    |        .             .    \n"
"        _|_         +     :          \n"
"      ,` | `.                  .     \n"
"--- --+-<#>-+- ---  --  -            \n"
"      `._|_,'                     '  \n"
"         T              '        .    \n"
"         |        `              !  \n"
"         !                 `    /|\ \n"
"         :         . :         / | \ \n"
"         .       *                   \n"
        ]
        diskette = [
     " __________________\n "
    "|# :   _      _ :#|\n"
    " |  : |  | | / /:  |\n"
    " |  : |_ |_| \_\:  |\n"
    " |  : |_ | | /_/:  |\n"
    " |  :___________:  |\n"
    " |     _________   |\n"
    " |    | __      |  |\n"
    " |    ||  |     |  |\n"
    " \____||__|_____|__|\n"
        ]
        ehs = [
            ":::::::::: :::    :::  ::::::::  \n"
            ":+:        :+:    :+: :+:    :+: \n"
            "+:+        +:+    +:+ +:+        \n"
            "+#++:++#   +#++:++#++ +#++:++#++ \n"
            "+#+        +#+    +#+        +#+ \n"
            "#+#        #+#    #+# #+#    #+# \n"
            "########## ###    ###  ########  \n"
        ]

        ascii_all_arts = cats + galaxie + diskette + ehs
        selected_art = random.choice(ascii_all_arts)
        print(selected_art)
        logger.log("ASCII art displayed", "OK")
        input("Press Enter to continue")
        clear_screen()

    except Exception as e:
        logger.log("Error displaying ASCII art", "FAIL", e)
        print(Fore.RED + f"⣏!⣽ Error: {e}" + Style.RESET_ALL)


# shell info
def shell_info():
    """
    Display Event Horizon shell information and ASCII logo.
    
    Shows version, author, and colored ASCII art representation.
    """
    from colorama import Fore, Style

    logger.log("Displaying shell info", "INFO")

    try:
        clear_screen()
        print(
            Fore.GREEN + "             ⣀⣴⣿⣿⣿⣿⣿⣿⣶⣄\n"
            "            ⣶⣿⣿⡿⠛⠉⠉⠉⠛⢿⣿⣿⣷⣄\n"
            "        ⢀⣴⣿⣿⣿⠋⡀⠀⠀⠀⠀⠀⠀⠙⣿⣿⣿⣿⣦⡀\n"
            "⠀⠀⣀⣰⣶⣿⣿⣿⣿⣿⣿⣀⣀⣀⣀⣀⣀⣰⣰⣶⣿⣿⣿⣿⣿⣿⣷⣦⣤⣀\n"
            " ⠉⠉⠉⠈⠉⠛⠛⠛⠛⣿⣿⡽⣏⠉⠉⠉⠉⠉⣽⣿⠛⠉⠉⠉⠉⠉⠉⠉⠁\n"
            "             ⠉⠻⣷⣦⣄⣀⣠⣴⣾⡿⠃\n"
            "                 ⠛⠿⠿⠿⠟⠋"
        )
        print("Event Horizon")
        print(f"shell ver> {config.SHELL_VERSION}")
        print("by quik" + Style.RESET_ALL)
        logger.log("Shell info displayed successfully", "OK")
        input("Press Enter to continue")
        clear_screen()

    except Exception as e:
        logger.log("Error displaying shell info", "FAIL", e)
        print(Fore.RED + f"⣏!⣽ Error: {e}" + Style.RESET_ALL)


# Shutdown
def shutdown():
    """
    shutdown the application
    """
    from colorama import Fore, Style

    logger.log("Initiating shutdown", "INFO")

    try:
        print(Fore.RED + "⠋Shutting down..." + Style.RESET_ALL)
        time.sleep(1)
        logger.log("Shutdown completed", "OK")
        clear_screen()
        sys.exit()

    except Exception as e:
        logger.log("Error during shutdown", "FAIL", e)
        print(Fore.RED + f"⣏!⣽ Error: {e}" + Style.RESET_ALL)


# OS info
def os_info():
    """
    Display operating system information.
    
    Shows system specs including OS version, CPU details, RAM size,
    Python version, and architecture information.
    """
    from colorama import Fore, Style
    import platform
    import psutil

    logger.log("Displaying OS information", "INFO")

    try:
        print(f"{Style.BRIGHT} -System information- ")
        print(f"{Fore.BLUE} OS: {platform.system()} {platform.release()} (Build {platform.version()})")
        print(f"{Fore.CYAN} CPU: {psutil.cpu_count(logical=False)} physical cores, {psutil.cpu_count(logical=True)} logical cores,",
              f"@ {psutil.cpu_freq().current/1000:.1f}GHz" if psutil.cpu_freq() else "None")
        mem = psutil.virtual_memory()
        print(f"{Fore.LIGHTCYAN_EX} RAM: {mem.total/(1024**3):.1f}GB")
        print(f"{Fore.LIGHTBLUE_EX} Python: {platform.python_version()} ({platform.python_implementation()})")
        print(f"{Style.DIM} Architecture: {platform.architecture()[0]}" + Style.RESET_ALL)
        logger.log("OS information displayed successfully", "OK")

    except Exception as e:
        logger.log("Error displaying OS information", "FAIL", e)
        print(f"⣏!⣽ Error: {str(e)}")

    input("Press Enter to continue")

# Toggle logs
def toggle_logs():
    """
    Toggle logging on/off
    """
    from .constants import config
    from colorama import Fore, Style

    config.ENABLE_LOGGING = not config.ENABLE_LOGGING

    status = "ENABLED" if config.ENABLE_LOGGING else "DISABLED"
    color = Fore.GREEN if config.ENABLE_LOGGING else Fore.RED

    print(f"{color}Logging {status}{Style.RESET_ALL}")

    if not config.ENABLE_LOGGING:
        print(f"{Fore.YELLOW}[!!] Note: Logging will be re-enabled on next restart{Style.RESET_ALL}")

    input("Press Enter to continue")
    clear_screen()

# Toggle themes
def toggle_theme():
    from .constants import config
    from .theme_manager import apply_theme
    from colorama import Fore, Style

    themes = list(config.THEMES.keys())
    current_index = themes.index(config.CURRENT_THEME)

    next_index = (current_index + 1) % len(themes)
    config.CURRENT_THEME = themes[next_index]
    apply_theme(config)

    print(f"{Fore.GREEN}Theme changed to: {config.CURRENT_THEME}{Style.RESET_ALL}\n")
    print(f"{Fore.CYAN}Prompt color{Style.RESET_ALL}")
    print(f"{Fore.RED}Error color{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Success color{Style.RESET_ALL}")
    print(f"{Fore.BLUE}Info color{Style.RESET_ALL}")

    input("Press Enter to continue")
    clear_screen()

# Welcome
def welcome():
    from rich.console import Console
    import time
    console = Console()

    steps = 30

    for i in range(steps):
        color_code = 232 + int(23 * i / (steps - 1))
        console.print(f"[color({color_code})]Welcome![/]", end="\r")
        time.sleep(0.05)

# ping
def ping():
    import socket
    import time
    from colorama import Fore, Style

    def ping_check(hostname):
        try:
            if '://' in hostname:
                hostname = hostname.split('://')[1]
            hostname = hostname.split('/')[0]
            hostname = hostname.split(':')[0]

            print(f"{Fore.YELLOW}Checking: {hostname}...")

            start = time.time()
            ip = socket.gethostbyname(hostname)
            dns_time = time.time() - start

            ports_to_try = [443, 80, 22]
            connection_time = None

            for port in ports_to_try:
                try:
                    start = time.time()
                    socket1 = socket.create_connection((ip, port), timeout=5)
                    connection_time = time.time() - start
                    socket1.close()
                    print(f"{Fore.RED}Connected on port {port}")
                    break
                except:
                    continue

            if connection_time is None:
                raise Exception(f"{Fore.RED}Could not connect to any common ports")

            print(f"{Fore.GREEN}\nIP: {ip}")
            print(f"{Fore.GREEN}DNS resolution: {dns_time:.3f}s")
            print(f"{Fore.GREEN}Connection time: {connection_time:.3f}s")
            return True

        except socket.gaierror as e:
            error_code = e.args[0]
            if error_code == socket.EAI_NONAME:
                print(f"{Fore.RED}Error: Hostname '{hostname}' not found")
            elif error_code == socket.EAI_AGAIN:
                print(f"{Fore.RED}Error: Temporary DNS failure for '{hostname}'")
            else:
                print(f"{Fore.RED}DNS Error: {e}")
            return False
        except Exception as e:
            print(f"{Fore.RED}Error: {e}")
            return False

    host = input(f"{Fore.LIGHTYELLOW_EX}Enter host name> ")
    result = ping_check(host)
    status = f"{Fore.GREEN}Success" if result else f"{Fore.RED}Failed"
    print(status)
    input(f"\n{Fore.BLUE}Press Enter to continue{Style.RESET_ALL}")

# password generator
def password_gen():
    import random
    import string
    from colorama import Fore, Style
    print(f"{Fore.CYAN}=== Password Generator ==={Style.RESET_ALL}")
    length = int(input("Password length: "))
    use_digits = input("Include digits? (y/n): ").lower() == 'y'
    use_special = input("Include special characters? (y/n): ").lower() == 'y'
    
    chars = string.ascii_letters
    if use_digits:
        chars += string.digits
    if use_special:
        chars += "!@#$%^&*()_+-="
    
    password = ''.join(random.choice(chars) for _ in range(length))
    print(f"{Fore.GREEN}Generated password: {password}{Style.RESET_ALL}")
    
    input("Press Enter to continue...")
