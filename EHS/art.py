import sys
import time

from .chklib_clrscr import clear_screen
from .constants import SHELL_VERSION

# Loading screen


def event_horizon():
    """
    Display the Event Horizon loading screen animation.

    Shows an ASCII art animation with loading symbols and version information.
    """
    from colorama import Back, Fore, Style
    eh_art_symbols = ["-", "\\", "|", "/"]
    eh_ascii_art_eh = [
        "             ⣀⣴⣿⣿⣿⣿⣿⣿⣶⣄          ",
        "            ⣶⣿⣿⡿⠛⠉⠉⠉⠛⢿⣿⣿⣷⣄       ",
        "        ⢀⣴⣿⣿⣿⠋⡀⠀⠀⠀⠀⠀⠀⠙⣿⣿⣿⣿⣦⡀     ",
        "⠀⠀⣀⣰⣶⣿⣿⣿⣿⣿⣿⣀⣀⣀⣀⣀⣀⣰⣰⣶⣿⣿⣿⣿⣿⣿⣷⣦⣤⣀   ",
        " ⠉⠉⠉⠈⠉⠛⠛⠛⠛⣿⣿⡽⣏⠉⠉⠉⠉⠉⣽⣿⠛⠉⠉⠉⠉⠉⠉⠉⠁   ",
        "             ⠉⠻⣷⣦⣄⣀⣠⣴⣾⡿⠃         ",
        "                 ⠛⠿⠿⠿⠟⠋          ",
    ]
    for eh_step in range(10, 0, -1):
        clear_screen()
        eh_spaces = " " * eh_step
        for line in eh_ascii_art_eh:
            print(Fore.GREEN + eh_spaces + line + eh_spaces + Style.RESET_ALL)
        time.sleep(0.1)
    clear_screen()
    for eh_line in eh_ascii_art_eh:
        print(Back.WHITE + Fore.BLACK + Style.BRIGHT + eh_line + Style.RESET_ALL)
    print(
        Fore.RED
        + " " * 10
        + "Event Horizon\n"
        + " " * 10
        + SHELL_VERSION
        + Style.RESET_ALL
    )
    for i in range(8):
        sys.stdout.write(
            Fore.GREEN
            + "\r"
            + " " * 10
            + "loading "
            + eh_art_symbols[i % 4]
            + Style.RESET_ALL
        )
        time.sleep(0.5)
    clear_screen()
