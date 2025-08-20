import os
import shutil
import time
from datetime import datetime

from .chklib_clrscr import clear_screen
from .constants import SHELL_VERSION

# get prompt (for shell)

def get_prompt():
    """
    This function returns a formatted string representing the current directory.
    """
    from colorama import Fore, Style
    get_prompt_current_dir = os.getcwd()
    get_prompt_home_dir = os.path.expanduser("~")
    if get_prompt_current_dir.startswith(get_prompt_home_dir):
        get_prompt_current_dir = get_prompt_current_dir.replace(
            get_prompt_home_dir, "~", 1
        )
    return Fore.CYAN + f"#{get_prompt_current_dir}> " + Style.RESET_ALL


# Shell


def shell():
    """
    Event Horizon built-in shell with file system operations.
    
    Provides commands for directory navigation, file operations, system monitoring,
    and basic shell functionality with colored output and error handling.
    see more in help :)
    """
    from colorama import Fore, Style
    import psutil
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
            print("⣦rmfile-remove file")

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

        # mkdir

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

        # rmdir /all

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
                        + "(Press Enter to delete)" + Style.RESET_ALL
                    )
                    shutil.rmtree(rmdir_all_expanded_path)
                    print(
                        Fore.GREEN
                        + f"Folder '{rmdir_all_expanded_path}' deleted successfully!"
                        + Style.RESET_ALL
                    )
                except PermissionError:
                    print(Fore.RED + "⣏!⣽ Permission denied!" + Style.RESET_ALL)
                except Exception as e:
                    print(
                        Fore.RED + f"⣏!⣽ Error deleting folder: {e}" + Style.RESET_ALL
                    )

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
                except PermissionError:
                    print(Fore.RED + "⣏!⣽ Permission denied!" + Style.RESET_ALL)
                except Exception as e:
                    print(
                        Fore.RED + f"⣏!⣽ Error deleting folder: {e}" + Style.RESET_ALL
                    )

        # dir

        elif shell_command == "dir":
            try:
                current_dir = os.getcwd()
                print(Fore.CYAN + f"Contents of '{current_dir}':" + Style.RESET_ALL)
                for dir_item in os.listdir(current_dir):
                    dir_item_path = os.path.join(current_dir, dir_item)
                    if os.path.isdir(dir_item_path):
                        print(Fore.BLUE + f"[DIR]  {dir_item}" + Style.RESET_ALL)
                    else:
                        print(Fore.GREEN + f"[FILE] {dir_item}" + Style.RESET_ALL)
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
                    Fore.GREEN + f"Current directory: {os.getcwd()}" + Style.RESET_ALL
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
                
        # rmfile

        elif shell_command == "rmfile":
            rmfile_path = input("Enter file path to delete: ")
            rmfile_expanded_path = os.path.expanduser(rmfile_path)
            if not os.path.exists(rmfile_expanded_path):
                print(Fore.RED + f"⣏!⣽ Path is not a file or doesn't exist: {rmfile_expanded_path}" + Style.RESET_ALL)
                continue
            if os.path.isdir(rmfile_expanded_path):
                print(Fore.RED + "⣏!⣽ This is a directory, use rmdir instead" + Style.RESET_ALL)
                continue
            try:
                input(
                    Fore.RED
                    + "Are you sure you want to delete the file? \n"
                    + "(Press Enter to delete)" + Style.RESET_ALL
                )
                os.remove(rmfile_expanded_path)
                print(Fore.GREEN + f"File deleted: {rmfile_expanded_path}" + Style.RESET_ALL)
            except PermissionError:
                print(Fore.RED + f"⣏!⣽ Permission denied: {rmfile_expanded_path}" + Style.RESET_ALL)
            except Exception as e:
                print(Fore.RED + f"⣏!⣽ Error deleting file: {str(e)}" + Style.RESET_ALL)
        else:
            print("⣏!⣽ invalid_choice")
