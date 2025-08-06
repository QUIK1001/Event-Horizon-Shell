import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime

import psutil
import requests
from colorama import Back, Fore, Style
from packaging import version

SHELL_VERSION = "v1.0.7"
REQUIRED_LIB = ["psutil", "requests", "colorama", "packaging"]

# Check lib


def check_lib():
    missing_packages = []
    for package in REQUIRED_LIB:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(Fore.RED + "⣏!⣽ Missing dependencies:" + Style.RESET_ALL)
        for pkg in missing_packages:
            print(f" - {pkg}")

        install = input("Install missing packages? (Y/N) ").upper()
        if install == "Y":
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", *missing_packages]
                )
                print(
                    Fore.GREEN
                    + "Dependencies installed successfully!"
                    + Style.RESET_ALL
                )
                return True
            except Exception as e:
                print(Fore.RED + f"⣏!⣽ Failed to install: {str(e)}" + Style.RESET_ALL)
                return False
        else:
            print(
                Fore.RED
                + "⣏!⣽ Some features may not work without dependencies."
                + Style.RESET_ALL
            )
            return False
    return True


if not check_lib():
    sys.exit(1)

# Check updates

GITHUB_LATEST_UPDATE = (
    "https://github.com/QUIK1001/Event-Horizon-Shell/blob/main/check_update"
)


def check_updates():
    try:
        response = requests.get(GITHUB_LATEST_UPDATE, timeout=3)
        response.raise_for_status()
        latest_version = response.text.strip()
        if version.parse(latest_version) > version.parse(SHELL_VERSION):
            print(
                Fore.RED
                + f"Update! {SHELL_VERSION} < {latest_version}"
                + Style.RESET_ALL
            )
            print(
                Fore.CYAN
                + "Download: https://github.com/QUIK1001/Event-Horizon-Shell/blob/main/check_update"
                + Style.RESET_ALL
            )
            return True

        print(Fore.GREEN + "Actual version" + Style.RESET_ALL)
        return False
    except requests.exceptions.RequestException as e:
        print(Fore.RED + f"Error connecting: {str(e)}" + Style.RESET_ALL)
        return False


# Clear


def clear_screen():
    os.system("clear" if os.name == "posix" else "cls")


# Loading screen


def event_horizon():
    art_symbols = ["-", "\\", "|", "/"]
    ascii_art_eh = [
        "             ⣀⣴⣿⣿⣿⣿⣿⣿⣶⣄          ",
        "            ⣶⣿⣿⡿⠛⠉⠉⠉⠛⢿⣿⣿⣷⣄       ",
        "        ⢀⣴⣿⣿⣿⠋⡀⠀⠀⠀⠀⠀⠀⠙⣿⣿⣿⣿⣦⡀     ",
        "⠀⠀⣀⣰⣶⣿⣿⣿⣿⣿⣿⣀⣀⣀⣀⣀⣀⣰⣰⣶⣿⣿⣿⣿⣿⣿⣷⣦⣤⣀   ",
        " ⠉⠉⠉⠈⠉⠛⠛⠛⠛⣿⣿⡽⣏⠉⠉⠉⠉⠉⣽⣿⠛⠉⠉⠉⠉⠉⠉⠉⠁   ",
        "             ⠉⠻⣷⣦⣄⣀⣠⣴⣾⡿⠃         ",
        "                 ⠛⠿⠿⠿⠟⠋          ",
    ]
    for step in range(10, 0, -1):
        clear_screen()
        spaces = " " * step
        for line in ascii_art_eh:
            print(Fore.GREEN + spaces + line + spaces + Style.RESET_ALL)
        time.sleep(0.1)
    clear_screen()
    for line in ascii_art_eh:
        print(Back.WHITE + Fore.BLACK + Style.BRIGHT + line + Style.RESET_ALL)
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
            + art_symbols[i % 4]
            + Style.RESET_ALL
        )
        time.sleep(0.5)
    clear_screen()
    print("---=Event Horizon=---")


# Calc


def calc():
    while True:
        try:
            calc_num1 = int(input("⣦First number> "))
            calc_act = input("⣦Action +,-,*,/> ")
            calc_num2 = int(input("⣦Second number> "))
            if calc_act == "+":
                print(
                    calc_num1,
                    calc_act,
                    calc_num2,
                    Fore.GREEN + "equals> " + Style.RESET_ALL,
                    calc_num1 + calc_num2,
                )
            elif calc_act == "-":
                print(
                    calc_num1,
                    calc_act,
                    calc_num2,
                    Fore.GREEN + "equals> " + Style.RESET_ALL,
                    calc_num1 - calc_num2,
                )
            elif calc_act == "*":
                print(
                    calc_num1,
                    calc_act,
                    calc_num2,
                    Fore.GREEN + "equals> " + Style.RESET_ALL,
                    calc_num1 * calc_num2,
                )
            elif calc_act == "/":
                if calc_num2 != 0:
                    print(
                        calc_num1,
                        calc_act,
                        calc_num2,
                        Fore.GREEN + "equals> " + Style.RESET_ALL,
                        calc_num1 / calc_num2,
                    )
                else:
                    print(Fore.RED + "⣏!⣽ DIV/0!" + Style.RESET_ALL)
        except ValueError:
            print(Fore.RED + "⣏!⣽ Numbers only!" + Style.RESET_ALL)
            continue
        if input("⣦Exit? Y/N> ").upper() == "Y":
            clear_screen()
            break


# Echo


def echo():
    echo_text = input("⣦Enter your text: ")
    print("⣦Your text:", echo_text)
    input("Press Enter to continue")
    clear_screen()


# ASCII arts


def ascii_arts():
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
    all_ascii_arts = ascii_cats + ascii_galaxie
    print(random.choice(all_ascii_arts))
    input("Press Enter to continue")
    clear_screen()


# OS info


def os_info():
    clear_screen()
    print(Fore.GREEN + "             ⣀⣴⣿⣿⣿⣿⣿⣿⣶⣄")
    print("            ⣶⣿⣿⡿⠛⠉⠉⠉⠛⢿⣿⣿⣷⣄")
    print("        ⢀⣴⣿⣿⣿⠋⡀⠀⠀⠀⠀⠀⠀⠙⣿⣿⣿⣿⣦⡀")
    print("⠀⠀⣀⣰⣶⣿⣿⣿⣿⣿⣿⣀⣀⣀⣀⣀⣀⣰⣰⣶⣿⣿⣿⣿⣿⣿⣷⣦⣤⣀")
    print(" ⠉⠉⠉⠈⠉⠛⠛⠛⠛⣿⣿⡽⣏⠉⠉⠉⠉⠉⣽⣿⠛⠉⠉⠉⠉⠉⠉⠉⠁")
    print("             ⠉⠻⣷⣦⣄⣀⣠⣴⣾⡿⠃")
    print("                 ⠛⠿⠿⠿⠟⠋")
    print("Event Horizon")
    print(f"shell ver> {SHELL_VERSION}")
    print("by quik" + Style.RESET_ALL)
    input("Press Enter to continue")
    clear_screen()


# Shutdown


def shutdown():
    print(Fore.RED + "⠋Shutting down..." + Style.RESET_ALL)
    time.sleep(1)
    clear_screen()
    sys.exit()


# get prompt (for shell)


def get_prompt():
    current_dir = os.getcwd()
    home_dir = os.path.expanduser("~")
    if current_dir.startswith(home_dir):
        current_dir = current_dir.replace(home_dir, "~", 1)
    return Fore.CYAN + f"#{current_dir}> " + Style.RESET_ALL


# Shell


def shell():
    clear_screen()
    print("Event Horizon shell")
    print(
        Fore.GREEN
        + "⣦ Type 'help' for commands, type 'exit' for exit\n"
        + Style.RESET_ALL
    )
    while True:
        command = input(get_prompt())
        # Help
        if command == "help":
            print("⣦help-show help")
            print("⣦clear-clear screen")
            print("⣦info-shell info")
            print("⣦exit-exit to menu")
            print("⣦mkdir-create folder")
            print("⣦rmdir-remove folder\n/?-for reference")
            print("⣦time-show current time")
            print("⣦perf-show CPU & RAM usage")
            print("⣦dir-list directory contents")
            print("⣦cd-change directory")
        # Exit
        elif command == "exit":
            clear_screen()
            break
        # Clear
        elif command == "clear":
            clear_screen()
            print("⣦Event Horizon shell\n")
        # Info
        elif command == "info":
            print(Fore.GREEN + "\n⣦Event Horizon")
            print(f"⣦shell ver> {SHELL_VERSION}")
            print("⣦by quik\n" + Style.RESET_ALL)
        # MKdir
        elif command == "mkdir":
            shell_mkdir = input("Enter folder name> ")
            choice = input(
                "Create in current dir (Y) or specify path (N)? Y/N> "
            ).upper()
            if choice == "Y":
                shell_parent_mkdir = os.getcwd()
            else:
                shell_parent_mkdir = input("Enter full path: ").strip()

            shell_mk_path = os.path.join(shell_parent_mkdir, shell_mkdir)
            try:
                os.mkdir(shell_mk_path)
                print(
                    Fore.GREEN
                    + f"Folder '{shell_mkdir}' created in '{shell_parent_mkdir}'"
                    + Style.RESET_ALL
                )
            except FileExistsError:
                print(
                    Fore.RED
                    + f"⣏!⣽ Folder '{shell_mk_path}' already exists!"
                    + Style.RESET_ALL
                )
            except PermissionError:
                print(Fore.RED + "⣏!⣽ Permission denied!" + Style.RESET_ALL)
        # RMdir reference
        elif command == "rmdir /?":
            print(
                Fore.GREEN + "rmdir| prefix",
                Fore.RED
                + "/all "
                + Fore.GREEN
                + "deletes all contents of the folder"
                + Style.RESET_ALL,
            )
        # RMdir
        elif command == "rmdir":
            shell_rm_path = input("Enter folder path to delete: ")
            expanded_path = os.path.expanduser(shell_rm_path)
            if not os.path.isabs(expanded_path):
                expanded_path = os.path.abspath(expanded_path)
            if not os.path.exists(expanded_path):
                print(Fore.RED + "⣏!⣽ folder doesn't exist!" + Style.RESET_ALL)
            else:
                try:
                    os.rmdir(expanded_path)
                    print(
                        Fore.GREEN
                        + f"Folder '{expanded_path}' deleted successfully!"
                        + Style.RESET_ALL
                    )
                except Exception as e:
                    print(
                        Fore.RED + f"⣏!⣽ Error deleting folder: {e}" + Style.RESET_ALL
                    )
        # dir
        elif command == "dir":
            try:
                current_dir = os.getcwd()
                print(Fore.CYAN + f"Contents of '{current_dir}':" + Style.RESET_ALL)
                for item in os.listdir(current_dir):
                    item_path = os.path.join(current_dir, item)
                    if os.path.isdir(item_path):
                        print(Fore.BLUE + f"[DIR]  {item}" + Style.RESET_ALL)
                    else:
                        print(Fore.GREEN + f"[FILE] {item}" + Style.RESET_ALL)
            except Exception as e:
                print(Fore.RED + f"⣏!⣽ Error: {str(e)}" + Style.RESET_ALL)

        # cd
        elif command == "cd":
            new_dir = input("Enter directory path> ").strip()
            if not new_dir:
                print(Fore.RED + "⣏!⣽ Path cannot be empty!" + Style.RESET_ALL)
                continue
            try:
                os.chdir(os.path.expanduser(new_dir))
                print(
                    Fore.GREEN + f"Current directory: {os.getcwd()}" + Style.RESET_ALL
                )
            except FileNotFoundError:
                print(
                    Fore.RED
                    + f"⣏!⣽ Directory '{new_dir}' does not exist!"
                    + Style.RESET_ALL
                )
            except PermissionError:
                print(
                    Fore.RED
                    + f"⣏!⣽ Permission denied! Cannot access '{new_dir}'"
                    + Style.RESET_ALL
                )
            except Exception as e:
                print(Fore.RED + f"⣏!⣽ Error: {str(e)}" + Style.RESET_ALL)
        # Time
        elif command == "time":
            shell_time = datetime.now()
            shell__time = shell_time.strftime(
                Fore.BLUE + "%Y-%m-%d %H:%M:%S:%MS" + Style.RESET_ALL
            )
            print(shell__time)
        # RMdir /all
        elif command == "rmdir /all":
            shell_rm_a_path = input("Enter folder path to delete: ")
            expanded_path = os.path.expanduser(shell_rm_a_path)
            if not os.path.isabs(expanded_path):
                expanded_path = os.path.abspath(expanded_path)
            if not os.path.exists(expanded_path):
                print(Fore.RED + "⣏!⣽ folder doesn't exist!" + Style.RESET_ALL)
            else:
                try:
                    shutil.rmtree(expanded_path)
                    print(
                        Fore.GREEN
                        + f"Folder '{expanded_path}' deleted successfully!"
                        + Style.RESET_ALL
                    )
                except Exception as e:
                    print(
                        Fore.RED + f"⣏!⣽ Error deleting folder: {e}" + Style.RESET_ALL
                    )
        # Perf
        elif command == "perf":
            clear_screen()
            print(
                Fore.RED
                + "⣦ System monitor started. Press Ctrl+C to stop."
                + Style.RESET_ALL
            )
            input("Press Enter to continue")
            try:
                while True:
                    clear_screen()
                    print(
                        Fore.RED
                        + f"CPU:{psutil.cpu_percent()}% \nRAM: {psutil.virtual_memory().percent}%"
                        + Style.RESET_ALL
                    )
                    time.sleep(1)
            except KeyboardInterrupt:
                print(Fore.GREEN + "\n⣦ Monitor stopped." + Style.RESET_ALL)
                clear_screen()
                print("⣦Event Horizon shell\n")
        else:
            print("⣏!⣽ invalid_choice")


# Timer


def timer():
    timer_time = int(input("enter clock time in seconds> "))
    print(timer_time)
    for i in range(timer_time):
        time.sleep(1)
        print(timer_time - i - 1)
    print(Fore.GREEN + "Timer finished!" + Style.RESET_ALL)
    input("Press Enter to continue")
    clear_screen()


# Сheck internet speed


def speed_test():
    print(Fore.CYAN + "⣦ Checking internet speed..." + Style.RESET_ALL)
    test_url = "https://speedtest.selectel.ru/100MB"
    try:
        start_test_time = time.time()
        response = requests.get(test_url, timeout=30)
        response.raise_for_status()
        end_time = time.time()
        downloaded = len(response.content)
        total_time = max(0.1, end_time - start_test_time)
        speed_kbs = downloaded / total_time / 1024
        print(
            Fore.GREEN + f"\nDownloaded: {downloaded / 1024:.1f} KB" + Style.RESET_ALL
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


# Menu


def menu():
    if not check_lib():
        print(
            Fore.RED + "⣏!⣽ Critical dependencies missing. Exiting." + Style.RESET_ALL
        )
        sys.exit(1)
    event_horizon()
    while True:
        print(
            Style.BRIGHT
            + "1. calc\n2. echo\n3. ASCII arts\n4. shell info\n5. shutdown\n6. shell\n7. timer\n8. check updates\n9. check internet speed"
            + Style.RESET_ALL
        )
        choice = input(Fore.RED + "select> " + Style.RESET_ALL)
        if choice == "1":
            clear_screen()
            calc()
        elif choice == "2":
            clear_screen()
            echo()
        elif choice == "3":
            clear_screen()
            ascii_arts()
        elif choice == "4":
            clear_screen()
            os_info()
        elif choice == "5":
            clear_screen()
            shutdown()
        elif choice == "6":
            shell()
        elif choice == "7":
            clear_screen()
            timer()
        elif choice == "8":
            clear_screen()
            check_updates()
            input("Press Enter to continue")
            clear_screen()
        elif choice == "9":
            clear_screen()
            speed_test()
        else:
            print("⣏!⣽ invalid_choice")
            time.sleep(1)
            clear_screen()


if __name__ == "__main__":
    menu()
