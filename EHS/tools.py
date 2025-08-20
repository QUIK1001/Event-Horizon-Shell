import os
import random
import sys
import time
import subprocess

from .chklib_clrscr import clear_screen
from .constants import CHANGELOG, SHELL_VERSION

# Timer
def timer():
    """
    Countdown timer with seconds input.
    """    
    from colorama import Fore, Style
    timer_time = int(input("enter clock time in seconds> "))
    print(timer_time)
    for i in range(timer_time):
        time.sleep(1)
        print(timer_time - i - 1)
    print(Fore.GREEN + "Timer finished!" + Style.RESET_ALL)
    input("Press Enter to continue")
    clear_screen()


# Changelog
def changelog():
    """
    Display changelog and version information.
    """
    from colorama import Fore, Style
    from art import text2art
    clear_screen()
    print(text2art(SHELL_VERSION))
    print(Fore.RED + CHANGELOG + Style.RESET_ALL)
    input("Press Enter to continue")
    clear_screen()


# Echo
def echo():
    """
    Simple echo command - repeats user input.
    """
    echo_text = input("⣦Enter your text: ")
    print("⣦Your text:", echo_text)
    input("Press Enter to continue")
    clear_screen()


# ASCII arts
def ascii_arts():
    """
    Display random ASCII art from collection.
    """
    ascii_cats = [
        """ /|_/|
( o.o )
 > ^ < """,
        """  /|_/|
 =( °w° )=
  )   (  """,
    ]
    ascii_galaxie = [
        """    . * .
  * . * . *
. * . * . * .
  * . * . *
    . * ."""
    ]
    ascii_all_arts = ascii_cats + ascii_galaxie
    print(random.choice(ascii_all_arts))
    input("Press Enter to continue")
    clear_screen()


# shell info
def shell_info():
    """
    Display Event Horizon shell information and ASCII logo.
    
    Shows version, author, and colored ASCII art representation.
    """
    from colorama import Fore, Style
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
    print(f"shell ver> {SHELL_VERSION}")
    print("by quik" + Style.RESET_ALL)
    input("Press Enter to continue")
    clear_screen()


# Shutdown
def shutdown():
    """
    shutdown the application
    """
    from colorama import Fore, Style
    print(Fore.RED + "⠋Shutting down..." + Style.RESET_ALL)
    time.sleep(1)
    clear_screen()
    sys.exit()

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
    try:
        print(f"{Style.BRIGHT} -System information- ")
        print(f"{Fore.BLUE} OS: {platform.system()} {platform.release()} (Build {platform.version()})")
        print(f"{Fore.CYAN} CPU: {psutil.cpu_count(logical=False)} physical cores, {psutil.cpu_count(logical=True)} logical cores,", 
              f"@ {psutil.cpu_freq().current/1000:.1f}GHz" if psutil.cpu_freq() else "None")
        mem = psutil.virtual_memory()
        print(f"{Fore.LIGHTCYAN_EX} RAM: {mem.total/(1024**3):.1f}GB")
        print(f"{Fore.LIGHTBLUE_EX} Python: {platform.python_version()} ({platform.python_implementation()})")
        print(f"{Style.DIM} Architecture: {platform.architecture()[0]}" + Style.RESET_ALL)
    except Exception as e:
        print(f"⣏!⣽ Error: {str(e)}")
    input("Press Enter to continue")

