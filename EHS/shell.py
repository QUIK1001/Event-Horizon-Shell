import os
import sys
import shutil
import time
from datetime import datetime

from .chklib_clrscr import clear_screen
from .constants import config
from .logger import logger

def get_prompt():
    """This function returns a formatted string representing the current directory."""
    from colorama import Fore, Style
    get_prompt_current_dir = os.getcwd()
    get_prompt_home_dir = os.path.expanduser("~")
    if get_prompt_current_dir.startswith(get_prompt_home_dir):
        get_prompt_current_dir = get_prompt_current_dir.replace(
            get_prompt_home_dir, "~", 1
        )
    return Fore.CYAN + f"#{get_prompt_current_dir}> " + Style.RESET_ALL

def shell():
    """Event Horizon built-in shell with file system operations."""
    from colorama import Fore, Style
    import psutil
    
    logger.log("Initializing shell", "INFO")
    clear_screen()
    print("Event Horizon shell")
    print(
        Fore.GREEN
        + "⣦ Type 'help' for commands, type 'exit' for exit\n"
        + Style.RESET_ALL
    )
    logger.log("Shell started successfully", "OK")

    while True:
        try:
            shell_command = input(get_prompt())
            logger.log(f"Command received: {shell_command}", "DEBUG")

            # Help
            if shell_command == "help":
                logger.log("Displaying help", "INFO")
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
                logger.log("Help displayed", "OK")

            # Exit
            elif shell_command == "exit":
                logger.log("Exiting shell", "INFO")
                clear_screen()
                logger.log("Shell exited", "OK")
                break

            # Clear
            elif shell_command == "clear":
                logger.log("Clearing screen", "INFO")
                clear_screen()
                print("⣦Event Horizon shell\n")
                logger.log("Screen cleared", "OK")

            # Info
            elif shell_command == "info":
                logger.log("Displaying shell info", "INFO")
                print(Fore.GREEN + "\n⣦Event Horizon")
                print(f"⣦shell ver> {config.SHELL_VERSION}")
                print("⣦by quik\n" + Style.RESET_ALL)
                logger.log("Shell info displayed", "OK")

            # mkdir
            elif shell_command == "mkdir":
                mkdir_dir = input("Enter folder name> ")
                mkdir_choice = input(
                    "Create in current dir (Y) or specify path (N)? Y/N> "
                ).upper()
                
                logger.log(f"Creating directory: {mkdir_dir}", "INFO")
                
                if mkdir_choice == "Y":
                    mkdir_parent_dir = os.getcwd()
                else:
                    mkdir_parent_dir = input("Enter full path: ").strip()

                mkdir_path = os.path.join(mkdir_parent_dir, mkdir_dir)
                try:
                    os.mkdir(mkdir_path)
                    logger.log(f"Directory created: {mkdir_path}", "OK")
                    print(
                        Fore.GREEN
                        + f"Folder '{mkdir_dir}' created in '{mkdir_parent_dir}'"
                        + Style.RESET_ALL
                    )
                except FileExistsError as e:
                    logger.log(f"Directory already exists: {mkdir_path}", "FAIL", e)
                    print(
                        Fore.RED
                        + f"⣏!⣽ Folder '{mkdir_path}' already exists!"
                        + Style.RESET_ALL
                    )
                except PermissionError as e:
                    logger.log(f"Permission denied for directory creation: {mkdir_path}", "FAIL", e)
                    print(Fore.RED + "⣏!⣽ Permission denied!" + Style.RESET_ALL)
                except Exception as e:
                    logger.log(f"Error creating directory: {mkdir_path}", "FAIL", e)
                    print(Fore.RED + f"⣏!⣽ Error: {e}" + Style.RESET_ALL)

            # rmdir /all
            elif shell_command == "rmdir /all":
                rmdir_all_path = input("Enter folder path to delete: ")
                rmdir_all_expanded_path = os.path.expanduser(rmdir_all_path)
                
                protected_paths = [
                    os.path.expanduser("~"),
                    "/", "\\", 
                    "C:\\", "C:/",
                    "/home", "/etc", "/usr", "/bin", "/sbin", "/lib",
                    "C:\\Windows", "C:\\Program Files", "C:\\Program Files (x86)"
                ]
                
                rmdir_all_abs_path = os.path.abspath(rmdir_all_expanded_path)
                
                for protected_path in protected_paths:
                    protected_abs = os.path.abspath(os.path.expanduser(protected_path))
                    if rmdir_all_abs_path == protected_abs or rmdir_all_abs_path.startswith(protected_abs + os.sep):
                        logger.log(f"Attempt to delete protected path: {rmdir_all_abs_path}", "CRITICAL")
                        print(f"{Fore.RED}⣏!⣽ CRITICAL: Cannot delete system or protected directory!{Style.RESET_ALL}")
                        print(f"{Fore.YELLOW}EHS will shut down immediately for system safety! 🤨{Style.RESET_ALL}")
                        time.sleep(2)
                        sys.exit(1)
                
                logger.log(f"Recursively removing directory: {rmdir_all_expanded_path}", "WARN")
                
                if not os.path.isabs(rmdir_all_expanded_path):
                    rmdir_all_expanded_path = os.path.abspath(rmdir_all_expanded_path)
                if not os.path.exists(rmdir_all_expanded_path):
                    logger.log(f"Directory does not exist: {rmdir_all_expanded_path}", "FAIL")
                    print(Fore.RED + "⣏!⣽ folder doesn't exist!" + Style.RESET_ALL)
                else:
                    try:
                        input(
                            Fore.RED
                            + "Are you sure you want to delete the entire folder? \n"
                            + "(Press Enter to delete)" + Style.RESET_ALL
                        )
                        shutil.rmtree(rmdir_all_expanded_path)
                        logger.log(f"Directory recursively removed: {rmdir_all_expanded_path}", "OK")
                        print(
                            Fore.GREEN
                            + f"Folder '{rmdir_all_expanded_path}' deleted successfully!"
                            + Style.RESET_ALL
                        )
                    except PermissionError as e:
                        logger.log(f"Permission denied for recursive removal: {rmdir_all_expanded_path}", "FAIL", e)
                        print(Fore.RED + "⣏!⣽ Permission denied!" + Style.RESET_ALL)
                    except Exception as e:
                        logger.log(f"Error in recursive directory removal: {rmdir_all_expanded_path}", "FAIL", e)
                        print(
                            Fore.RED + f"⣏!⣽ Error deleting folder: {e}" + Style.RESET_ALL
                        )

            # RMdir reference
            elif shell_command == "rmdir /?":
                logger.log("Displaying rmdir reference", "INFO")
                print(
                    Fore.GREEN + "rmdir| prefix",
                    Fore.RED
                    + "/all "
                    + Fore.GREEN
                    + "deletes all contents of the folder"
                    + Style.RESET_ALL,
                )
                logger.log("Rmdir reference displayed", "OK")

            # RMdir
            elif shell_command == "rmdir":
                rmdir_path = input("Enter folder path to delete: ")
                rmdir_expanded_path = os.path.expanduser(rmdir_path)
                
                logger.log(f"Removing directory: {rmdir_expanded_path}", "INFO")
                
                if not os.path.isabs(rmdir_expanded_path):
                    rmdir_expanded_path = os.path.abspath(rmdir_expanded_path)
                if not os.path.exists(rmdir_expanded_path):
                    logger.log(f"Directory does not exist: {rmdir_expanded_path}", "FAIL")
                    print(Fore.RED + "⣏!⣽ folder doesn't exist!" + Style.RESET_ALL)
                else:
                    try:
                        os.rmdir(rmdir_expanded_path)
                        logger.log(f"Directory removed: {rmdir_expanded_path}", "OK")
                        print(
                            Fore.GREEN
                            + f"Folder '{rmdir_expanded_path}' deleted successfully!"
                            + Style.RESET_ALL
                        )
                    except PermissionError as e:
                        logger.log(f"Permission denied for directory removal: {rmdir_expanded_path}", "FAIL", e)
                        print(Fore.RED + "⣏!⣽ Permission denied!" + Style.RESET_ALL)
                    except OSError as e:
                        logger.log(f"OS error during directory removal: {rmdir_expanded_path}", "FAIL", e)
                        print(Fore.RED + f"⣏!⣽ Error: Directory not empty or system error" + Style.RESET_ALL)
                    except Exception as e:
                        logger.log(f"Error removing directory: {rmdir_expanded_path}", "FAIL", e)
                        print(
                            Fore.RED + f"⣏!⣽ Error deleting folder: {e}" + Style.RESET_ALL
                        )

            # dir
            elif shell_command == "dir":
                try:
                    current_dir = os.getcwd()
                    logger.log(f"Listing directory contents: {current_dir}", "INFO")
                    print(Fore.CYAN + f"Contents of '{current_dir}':" + Style.RESET_ALL)
                    
                    items = os.listdir(current_dir)
                    dir_count = sum(1 for item in items if os.path.isdir(os.path.join(current_dir, item)))
                    file_count = len(items) - dir_count
                    
                    for dir_item in items:
                        dir_item_path = os.path.join(current_dir, dir_item)
                        if os.path.isdir(dir_item_path):
                            print(Fore.BLUE + f"[DIR]  {dir_item}" + Style.RESET_ALL)
                        else:
                            print(Fore.GREEN + f"[FILE] {dir_item}" + Style.RESET_ALL)
                    
                    logger.log(f"Directory listed: {dir_count} directories, {file_count} files", "OK")
                except PermissionError as e:
                    logger.log(f"Permission denied for directory listing: {current_dir}", "FAIL", e)
                    print(Fore.RED + f"⣏!⣽ Permission denied for directory listing!" + Style.RESET_ALL)
                except Exception as e:
                    logger.log(f"Error listing directory: {current_dir}", "FAIL", e)
                    print(Fore.RED + f"⣏!⣽ Error: {str(e)}" + Style.RESET_ALL)

            # cd
            elif shell_command == "cd":
                cd_new_dir = input("Enter directory path> ").strip()
                if not cd_new_dir:
                    logger.log("Empty directory path provided", "WARN")
                    print(Fore.RED + "⣏!⣽ Path cannot be empty!" + Style.RESET_ALL)
                    continue
                
                logger.log(f"Changing directory to: {cd_new_dir}", "INFO")
                
                try:
                    expanded_path = os.path.expanduser(cd_new_dir)
                    os.chdir(expanded_path)
                    current_dir = os.getcwd()
                    logger.log(f"Directory changed to: {current_dir}", "OK")
                    print(
                        Fore.GREEN + f"Current directory: {current_dir}" + Style.RESET_ALL
                    )
                except FileNotFoundError as e:
                    logger.log(f"Directory not found: {cd_new_dir}", "FAIL", e)
                    print(
                        Fore.RED
                        + f"⣏!⣽ Directory '{cd_new_dir}' does not exist!"
                        + Style.RESET_ALL
                    )
                except PermissionError as e:
                    logger.log(f"Permission denied for directory: {cd_new_dir}", "FAIL", e)
                    print(
                        Fore.RED
                        + f"⣏!⣽ Permission denied! Cannot access '{cd_new_dir}'"
                        + Style.RESET_ALL
                    )
                except NotADirectoryError as e:
                    logger.log(f"Path is not a directory: {cd_new_dir}", "FAIL", e)
                    print(Fore.RED + f"⣏!⣽ Path is not a directory: '{cd_new_dir}'" + Style.RESET_ALL)
                except Exception as e:
                    logger.log(f"Error changing directory: {cd_new_dir}", "FAIL", e)
                    print(Fore.RED + f"⣏!⣽ Error: {str(e)}" + Style.RESET_ALL)

            # Time
            elif shell_command == "time":
                logger.log("Displaying current time", "INFO")
                time_time = datetime.now()
                ru_format = time_time.strftime("%d.%m.%Y %H:%M:%S")
                iso_format = time_time.strftime("%Y-%m-%d %H:%M:%S")
                print(Fore.BLUE + f"RU: {ru_format}" + Style.RESET_ALL)
                print(Fore.GREEN + f"ISO: {iso_format}" + Style.RESET_ALL)
                logger.log("Time displayed", "OK")

            # Perf
            elif shell_command == "perf":
                logger.log("Starting system monitor", "INFO")
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
                        cpu_usage = psutil.cpu_percent()
                        ram_usage = psutil.virtual_memory().percent
                        logger.log(f"System stats - CPU: {cpu_usage}%, RAM: {ram_usage}%", "DEBUG")
                        print(
                            Fore.RED
                            + f"CPU:{cpu_usage}% \nRAM: {ram_usage}%"
                            + Style.RESET_ALL
                        )
                        time.sleep(1)
                except KeyboardInterrupt:
                    logger.log("System monitor stopped by user", "INFO")
                    print(Fore.GREEN + "\n⣦ Monitor stopped." + Style.RESET_ALL)
                    clear_screen()
                    print("⣦Event Horizon shell\n")
                    logger.log("System monitor stopped", "OK")
                except Exception as e:
                    logger.log("Error in system monitor", "FAIL", e)
                    
            # rmfile
            elif shell_command == "rmfile":
                rmfile_path = input("Enter file path to delete: ")
                rmfile_expanded_path = os.path.expanduser(rmfile_path)
                
                logger.log(f"Removing file: {rmfile_expanded_path}", "WARN")
                
                if not os.path.exists(rmfile_expanded_path):
                    logger.log(f"File does not exist: {rmfile_expanded_path}", "FAIL")
                    print(Fore.RED + f"⣏!⣽ Path is not a file or doesn't exist: {rmfile_expanded_path}" + Style.RESET_ALL)
                    continue
                if os.path.isdir(rmfile_expanded_path):
                    logger.log(f"Path is a directory, not a file: {rmfile_expanded_path}", "FAIL")
                    print(Fore.RED + "⣏!⣽ This is a directory, use rmdir instead" + Style.RESET_ALL)
                    continue
                try:
                    input(
                        Fore.RED
                        + "Are you sure you want to delete the file? \n"
                        + "(Press Enter to delete)" + Style.RESET_ALL
                    )
                    os.remove(rmfile_expanded_path)
                    logger.log(f"File removed: {rmfile_expanded_path}", "OK")
                    print(Fore.GREEN + f"File deleted: {rmfile_expanded_path}" + Style.RESET_ALL)
                except PermissionError as e:
                    logger.log(f"Permission denied for file removal: {rmfile_expanded_path}", "FAIL", e)
                    print(Fore.RED + f"⣏!⣽ Permission denied: {rmfile_expanded_path}" + Style.RESET_ALL)
                except IsADirectoryError as e:
                    logger.log(f"Path is a directory: {rmfile_expanded_path}", "FAIL", e)
                    print(Fore.RED + f"⣏!⣽ Path is a directory, use rmdir instead" + Style.RESET_ALL)
                except Exception as e:
                    logger.log(f"Error removing file: {rmfile_expanded_path}", "FAIL", e)
                    print(Fore.RED + f"⣏!⣽ Error deleting file: {str(e)}" + Style.RESET_ALL)

            else:
                logger.log(f"Unknown command: {shell_command}", "WARN")
                print("⣏!⣽ invalid_choice")
                
        except KeyboardInterrupt:
            logger.log("Shell interrupted by user", "INFO")
            print("\n⣦ Use 'exit' to leave shell or Ctrl+D to force exit")
        except EOFError:
            logger.log("Shell EOF received", "INFO")
            print("\n⣦ Exiting shell...")
            break
        except Exception as e:
            logger.log("Unexpected error in shell", "FAIL", e)
            print(Fore.RED + f"⣏!⣽ Unexpected error: {e}" + Style.RESET_ALL)
