import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime

SHELL_VERSION = "v1.0.8"
REQUIRED_LIB = ["psutil", "requests", "colorama", "packaging"]

# Check lib


def check_lib():
    time.sleep(0.2)
    chk_lib_missing_packages = []
    for package in REQUIRED_LIB:
        try:
            __import__(package)
        except ImportError:
            chk_lib_missing_packages.append(package)

    if chk_lib_missing_packages:
        print("⣏!⣽ Missing dependencies:")
        for pkg in chk_lib_missing_packages:
            print(f" - {pkg}")

        chk_lib_install = input("Install missing packages? (Y/N) ").upper()
        if chk_lib_install == "Y":
            try:
                subprocess.check_call(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "--user",
                        *chk_lib_missing_packages,
                    ]
                )
                print("\n\nPlease restart the EHS.")
                time.sleep(1)
                sys.exit(0)
            except Exception as e:
                print(f"⣏!⣽ Failed to install: {str(e)}")
                return False
        else:
            print("You can install the packages manually with:")
            print(f"pip install --user {' '.join(chk_lib_missing_packages)}")
            time.sleep(1)
            return False
    return True


if not check_lib():
    sys.exit(1)

from packaging import version
from colorama import Back, Fore, Style
import requests
import psutil

# Check updates

GITHUB_LATEST_UPDATE = (
    "https://raw.githubusercontent.com/QUIK1001/Event-Horizon-Shell/main/check_update"
)


def check_updates():
    try:
        chk_upd_response = requests.get(GITHUB_LATEST_UPDATE, timeout=3)
        chk_upd_response.raise_for_status()
        chk_upd_latest_version = chk_upd_response.text.strip()
        if version.parse(chk_upd_latest_version) > version.parse(
                SHELL_VERSION):
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
            print(
                Fore.LIGHTBLACK_EX +
                "Are you from the future? :D" +
                Style.RESET_ALL)
            time.sleep(0.3)
            raise FutureWarning(
                Fore.RED + "Wait until this version comes out :)" + Style.RESET_ALL
            )
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
        print(
            Back.WHITE +
            Fore.BLACK +
            Style.BRIGHT +
            eh_line +
            Style.RESET_ALL)
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
    print("---=Event Horizon=---")


# Calc


def calc():
    while True:
        try:
            calc_num1 = float(input("⣦First number> "))
            calc_act = input("⣦Action +,-,*,/,**,%,//> ")
            calc_num2 = float(input("⣦Second number> "))
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
            elif calc_act == "**":
                print(
                    calc_num1,
                    calc_act,
                    calc_num2,
                    Fore.GREEN + "equals> " + Style.RESET_ALL,
                    calc_num1**calc_num2,
                )
            elif calc_act == "%":
                print(
                    calc_num1,
                    calc_act,
                    calc_num2,
                    Fore.GREEN + "equals> " + Style.RESET_ALL,
                    calc_num1 % calc_num2,
                )
            elif calc_act == "//":
                print(
                    calc_num1,
                    calc_act,
                    calc_num2,
                    Fore.GREEN + "equals> " + Style.RESET_ALL,
                    calc_num1 // calc_num2,
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
    ascii_all_arts = ascii_cats + ascii_galaxie
    print(random.choice(ascii_all_arts))
    input("Press Enter to continue")
    clear_screen()


# shell info


def shell_info():
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
    print(Fore.RED + "⠋Shutting down..." + Style.RESET_ALL)
    time.sleep(1)
    clear_screen()
    sys.exit()


# get prompt (for shell)


def get_prompt():
    get_prompt_current_dir = os.getcwd()
    get_prompt_home_dir = os.path.expanduser("~")
    if get_prompt_current_dir.startswith(get_prompt_home_dir):
        get_prompt_current_dir = get_prompt_current_dir.replace(
            get_prompt_home_dir, "~", 1
        )
    return Fore.CYAN + f"#{get_prompt_current_dir}> " + Style.RESET_ALL


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
        shell_command = input(get_prompt())

        # Help

        if shell_command == "help":
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

        elif shell_command == "exit":
            clear_screen()
            break

        # Clear

        elif shell_command == "clear":
            clear_screen()
            print("⣦Event Horizon shell\n")

        # Info

        elif shell_command == "info":
            print(Fore.GREEN + "\n⣦Event Horizon")
            print(f"⣦shell ver> {SHELL_VERSION}")
            print("⣦by quik\n" + Style.RESET_ALL)

        # MKdir

        elif shell_command == "mkdir":
            mkdir_dir = input("Enter folder name> ")
            mkdir_choice = input(
                "Create in current dir (Y) or specify path (N)? Y/N> "
            ).upper()
            if mkdir_choice == "Y":
                mkdir_parent_dir = os.getcwd()
            else:
                mkdir_parent_dir = input("Enter full path: ").strip()

            mkdir_path = os.path.join(mkdir_parent_dir, mkdir_dir)
            try:
                os.mkdir(mkdir_path)
                print(
                    Fore.GREEN
                    + f"Folder '{mkdir_dir}' created in '{mkdir_parent_dir}'"
                    + Style.RESET_ALL
                )
            except FileExistsError:
                print(
                    Fore.RED
                    + f"⣏!⣽ Folder '{mkdir_path}' already exists!"
                    + Style.RESET_ALL
                )
            except PermissionError:
                print(Fore.RED + "⣏!⣽ Permission denied!" + Style.RESET_ALL)

        # RMdir reference

        elif shell_command == "rmdir /?":
            print(
                Fore.GREEN + "rmdir| prefix",
                Fore.RED
                + "/all "
                + Fore.GREEN
                + "deletes all contents of the folder"
                + Style.RESET_ALL,
            )

        # RMdir

        elif shell_command == "rmdir":
            rmdir_path = input("Enter folder path to delete: ")
            rmdir_expanded_path = os.path.expanduser(rmdir_path)
            if not os.path.isabs(rmdir_expanded_path):
                rmdir_expanded_path = os.path.abspath(rmdir_expanded_path)
            if not os.path.exists(rmdir_expanded_path):
                print(Fore.RED + "⣏!⣽ folder doesn't exist!" + Style.RESET_ALL)
            else:
                try:
                    os.rmdir(rmdir_expanded_path)
                    print(
                        Fore.GREEN
                        + f"Folder '{rmdir_expanded_path}' deleted successfully!"
                        + Style.RESET_ALL
                    )
                except Exception as e:
                    print(
                        Fore.RED +
                        f"⣏!⣽ Error deleting folder: {e}" + Style.RESET_ALL
                    )
        # dir
        elif shell_command == "dir":
            try:
                current_dir = os.getcwd()
                print(
                    Fore.CYAN +
                    f"Contents of '{current_dir}':" +
                    Style.RESET_ALL)
                for dir_item in os.listdir(current_dir):
                    dir_item_path = os.path.join(current_dir, dir_item)
                    if os.path.isdir(dir_item_path):
                        print(
                            Fore.BLUE +
                            f"[DIR]  {dir_item}" +
                            Style.RESET_ALL)
                    else:
                        print(
                            Fore.GREEN +
                            f"[FILE] {dir_item}" +
                            Style.RESET_ALL)
            except Exception as e:
                print(Fore.RED + f"⣏!⣽ Error: {str(e)}" + Style.RESET_ALL)

        # cd

        elif shell_command == "cd":
            cd_new_dir = input("Enter directory path> ").strip()
            if not cd_new_dir:
                print(Fore.RED + "⣏!⣽ Path cannot be empty!" + Style.RESET_ALL)
                continue
            try:
                os.chdir(os.path.expanduser(cd_new_dir))
                print(
                    Fore.GREEN +
                    f"Current directory: {os.getcwd()}" + Style.RESET_ALL
                )
            except FileNotFoundError:
                print(
                    Fore.RED
                    + f"⣏!⣽ Directory '{cd_new_dir}' does not exist!"
                    + Style.RESET_ALL
                )
            except PermissionError:
                print(
                    Fore.RED
                    + f"⣏!⣽ Permission denied! Cannot access '{cd_new_dir}'"
                    + Style.RESET_ALL
                )
            except Exception as e:
                print(Fore.RED + f"⣏!⣽ Error: {str(e)}" + Style.RESET_ALL)

        # Time

        elif shell_command == "time":
            time_time = datetime.now()
            ru_format = time_time.strftime("%d.%m.%Y %H:%M:%S")
            iso_format = time_time.strftime("%Y-%m-%d %H:%M:%S")
            print(Fore.BLUE + f"RU: {ru_format}" + Style.RESET_ALL)
            print(Fore.GREEN + f"ISO: {iso_format}" + Style.RESET_ALL)

        # RMdir /all

        elif shell_command == "rmdir /all":
            rmdir_all_path = input("Enter folder path to delete: ")
            rmdir_all_expanded_path = os.path.expanduser(rmdir_all_path)
            if not os.path.isabs(rmdir_all_expanded_path):
                rmdir_all_expanded_path = os.path.abspath(rmdir_all_expanded_path)
            if not os.path.exists(rmdir_all_expanded_path):
                print(Fore.RED + "⣏!⣽ folder doesn't exist!" + Style.RESET_ALL)
            else:
                try:
                    input(
                        Fore.RED
                        + "Are you sure you want to delete the entire folder? \n"
                        "(Press Enter to delete)" + Style.RESET_ALL
                    )
                    shutil.rmtree(rmdir_all_expanded_path)
                    print(
                        Fore.GREEN
                        + f"Folder '{rmdir_all_expanded_path}' deleted successfully!"
                        + Style.RESET_ALL
                    )
                except Exception as e:
                    print(
                        Fore.RED +
                        f"⣏!⣽ Error deleting folder: {e}" + Style.RESET_ALL
                    )
        # Perf
        elif shell_command == "perf":
            clear_screen()
            print(
                Fore.RED
                + "⣦ System monitor started. Press Ctrl+C to stop."
                + Style.RESET_ALL
            )
            time.sleep(2.5)
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
            shell_info()
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
