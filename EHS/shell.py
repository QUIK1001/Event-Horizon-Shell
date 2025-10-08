import os
import shutil
import time
from datetime import datetime

from .chklib_clrscr import clear_screen
from .constants import config
from .logger import logger
from .aliases import resolve_alias, show_aliases, add_alias, remove_alias

def get_prompt():
    """This function returns a formatted string representing the current directory."""
    from colorama import Fore, Style
    from .constants import config

    get_prompt_current_dir = os.getcwd()
    get_prompt_home_dir = os.path.expanduser("~")

    if get_prompt_current_dir.startswith(get_prompt_home_dir):
        get_prompt_current_dir = get_prompt_current_dir.replace(
            get_prompt_home_dir, "~", 1
        )

    theme = config.THEMES[config.CURRENT_THEME]
    return theme["prompt"] + f"#{get_prompt_current_dir}> " + Style.RESET_ALL

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
            original_command = shell_command
            shell_command = resolve_alias(shell_command)
            if shell_command != original_command:
                logger.log(f"Alias expanded: {original_command} -> {shell_command}", "DEBUG")


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
                print("⣦find-find files\dirs")
                print("⣦alias-manage command aliases")
                print("⣦move-move file/folder")
                print("⣦copy-copy file/folder")
                print("⣦rename-rename file/folder")
                print("⣦grep-text search in files")
                print("⣦wc-calculating text statistics")
                logger.log("Help displayed", "OK")

            # Alias
            elif shell_command == "alias":
                show_aliases()
                print("\nIf you want to add or remove an alias use "
                "'alias add <name> <command>' or 'alias remove <name> <command>'")

            elif shell_command.startswith("alias add "):
                parts = shell_command.split(" ", 3)
                if len(parts) >= 4:
                    add_alias(parts[2], parts[3])
                else:
                    print(Fore.RED + "Usage: alias add <name> <command>" + Style.RESET_ALL)

            elif shell_command.startswith("alias remove "):
                parts = shell_command.split(" ", 2)
                if len(parts) >= 3:
                    remove_alias(parts[2])
                else:
                    print(Fore.RED + "Usage: alias remove <name>" + Style.RESET_ALL)

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
                try:
                    rmdir_all_path = input("Enter folder path to delete: ").strip()
                    if not rmdir_all_path:
                        print(Fore.RED + "⣏!⣽ Path cannot be empty!" + Style.RESET_ALL)
                        continue
                    rmdir_all_expanded_path = os.path.abspath(os.path.expanduser(rmdir_all_path))

                    if not os.path.exists(rmdir_all_expanded_path):
                        print(Fore.RED + f"⣏!⣽ Folder doesn't exist: {rmdir_all_expanded_path}" + Style.RESET_ALL)
                        continue

                    if not os.path.isdir(rmdir_all_expanded_path):
                        print(Fore.RED + f"⣏!⣽ Not a directory: {rmdir_all_expanded_path}" + Style.RESET_ALL)
                        continue
                    print(f"{Fore.RED}will delete: {rmdir_all_expanded_path}{Style.RESET_ALL}")
                    print(f"{Fore.YELLOW}This action cannot be undone!{Style.RESET_ALL}")
                    confirmation = input(f"{Fore.RED}Type 'delete all' to confirm: {Style.RESET_ALL}")

                    if confirmation == "delete all":
                        shutil.rmtree(rmdir_all_expanded_path)
                        print(Fore.GREEN + "Folder deleted successfully!" + Style.RESET_ALL)
                    else:
                        print(Fore.YELLOW + "Deletion cancelled" + Style.RESET_ALL)

                except Exception as e:
                    print(Fore.RED + f"⣏!⣽ Error: {e}" + Style.RESET_ALL)

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

            # Find
            elif shell_command == "find":
                try:
                    search_pattern = input("Search pattern: ")
                    if not search_pattern:
                        print("Pattern cannot be empty!")
                        continue

                    start_dir = input("Start directory (Enter for /): ").strip()
                    if not start_dir:
                        start_dir = "/"
                        print("Using root directory: /")

                    if not os.path.exists(start_dir):
                        print(f"Directory '{start_dir}' not found!")
                        continue

                    found_count = 0
                    results = []

                    print(f"Searching for '{search_pattern}' in '{start_dir}'...")
                    print("Press Ctrl+C to stop")

                    try:
                        for root, dirs, files in os.walk(start_dir):
                            for dir_name in dirs:
                                if search_pattern in dir_name:
                                    results.append(os.path.join(root, dir_name))
                                    found_count += 1

                            for file_name in files:
                                if search_pattern in file_name:
                                    results.append(os.path.join(root, file_name))
                                    found_count += 1

                    except KeyboardInterrupt:
                        print("\nSearch stopped by user")

                    print(f"\nFound {found_count} results:")
                    for result in results:
                        print(f"  {result}")

                    if found_count == 0:
                        print("No results found")

                except Exception as e:
                    print(f"Error: {e}")

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
                        if os.name == 'nt':
                            disk = psutil.disk_usage(os.getcwd()[:3])
                        else:
                            disk = psutil.disk_usage('/')
                        print(f"{Fore.CYAN}⣦ Disk Usage:{Style.RESET_ALL}")
                        print(f"Total: {disk.total//(1024**3)}GB | Used: {disk.used//(1024**3)}GB")
                        print(f"Free: {disk.free//(1024**3)}GB | {disk.percent}% used")
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

            # Move
            elif shell_command == "move":
                try:
                    source = input("Source path: ").strip()
                    destination = input("Destination path: ").strip()

                    if not source or not destination:
                        print(Fore.RED + "⣏!⣽ Source and destination required!" + Style.RESET_ALL)
                        continue

                    source = os.path.expanduser(source)
                    destination = os.path.expanduser(destination)

                    if not os.path.exists(source):
                        print(Fore.RED + f"⣏!⣽ Source not found: {source}" + Style.RESET_ALL)
                        continue
                    if os.path.isdir(destination):
                        destination = os.path.join(destination, os.path.basename(source))
                    if os.path.exists(destination):
                        overwrite = input(f"{Fore.YELLOW}Destination already exists. Overwrite? (Y/N): {Style.RESET_ALL}").upper()
                        if overwrite != 'Y':
                            print(Fore.YELLOW + "Move cancelled" + Style.RESET_ALL)
                            continue
                        if os.path.isdir(destination):
                            shutil.rmtree(destination)
                        else:
                            os.remove(destination)

                    shutil.move(source, destination)
                    logger.log(f"Moved: {source} -> {destination}", "OK")
                    print(Fore.GREEN + f"Moved: {source} -> {destination}" + Style.RESET_ALL)

                except Exception as e:
                    logger.log(f"Move error: {e}", "FAIL")
                    print(Fore.RED + f"⣏!⣽ Error: {e}" + Style.RESET_ALL)

            # Copy
            elif shell_command == "copy":
                try:
                    source = input("Source path: ").strip()
                    destination = input("Destination path: ").strip()

                    if not source or not destination:
                        print(Fore.RED + "⣏!⣽ Source and destination required!" + Style.RESET_ALL)
                        continue

                    source = os.path.expanduser(source)
                    destination = os.path.expanduser(destination)

                    if not os.path.exists(source):
                        print(Fore.RED + f"⣏!⣽ Source not found: {source}" + Style.RESET_ALL)
                        continue
                    final_destination = destination
                    if os.path.isdir(destination):
                        final_destination = os.path.join(destination, os.path.basename(source))
                    if os.path.exists(final_destination):
                        overwrite = input(f"{Fore.YELLOW}Destination already exists. Overwrite? (Y/N): {Style.RESET_ALL}").upper()
                        if overwrite != 'Y':
                            print(Fore.YELLOW + "Copy cancelled" + Style.RESET_ALL)
                            continue
                        if os.path.isdir(final_destination):
                            shutil.rmtree(final_destination)
                        else:
                            os.remove(final_destination)

                    if os.path.isdir(source):
                        shutil.copytree(source, final_destination)
                    else:
                        shutil.copy2(source, final_destination)

                    logger.log(f"Copied: {source} -> {final_destination}", "OK")
                    print(Fore.GREEN + f"Copied: {source} -> {final_destination}" + Style.RESET_ALL)

                except Exception as e:
                    logger.log(f"Copy error: {e}", "FAIL")
                    print(Fore.RED + f"⣏!⣽ Error: {e}" + Style.RESET_ALL)

            # Rename
            elif shell_command == "rename":
                try:
                    old_name = input("Current path: ").strip()
                    new_name = input("New name: ").strip()

                    if not old_name or not new_name:
                        print(Fore.RED + "⣏!⣽ Both names required!" + Style.RESET_ALL)
                        continue

                    old_name = os.path.expanduser(old_name)

                    if not os.path.exists(old_name):
                        print(Fore.RED + f"⣏!⣽ File not found: {old_name}" + Style.RESET_ALL)
                        continue
                    if not os.path.isabs(new_name):
                        new_name = os.path.join(os.path.dirname(old_name), new_name)

                    os.rename(old_name, new_name)
                    logger.log(f"Renamed: {old_name} -> {new_name}", "OK")
                    print(Fore.GREEN + f"Renamed: {old_name} -> {new_name}" + Style.RESET_ALL)

                except Exception as e:
                    logger.log(f"Rename error: {e}", "FAIL")
                    print(Fore.RED + f"⣏!⣽ Error: {e}" + Style.RESET_ALL)

            # Grep
            elif shell_command == "grep":
                import re
                logger.log("Starting grep search", "INFO")
                    
                try:
                    pattern = input("⣦ Search pattern: ").strip()
                    if not pattern:
                        print(f"{Fore.RED}⣏!⣽ Pattern cannot be empty!{Style.RESET_ALL}")
                        continue
                        
                    filename = input("⣦ File to search: ").strip()
                    if not filename:
                        print(f"{Fore.RED}⣏!⣽ Filename cannot be empty!{Style.RESET_ALL}")
                        continue
                        
                    filename = os.path.expanduser(filename)
                        
                    if not os.path.exists(filename):
                        print(f"{Fore.RED}⣏!⣽ File not found: {filename}{Style.RESET_ALL}")
                        continue
                        
                    if os.path.isdir(filename):
                        print(f"{Fore.RED}⣏!⣽ Path is a directory, not a file{Style.RESET_ALL}")
                        continue
                        
                    case_sensitive = input("⣦ Case sensitive? (Y/N): ").upper() == 'Y'
                    use_regex = input("⣦ Use regex? (Y/N): ").upper() == 'Y'
                        
                    flags = 0 if case_sensitive else re.IGNORECASE
                        
                    matches_found = 0
                    print(f"\n{Fore.CYAN}Searching for '{pattern}' in {filename}:{Style.RESET_ALL}\n")
                        
                    with open(filename, 'r', encoding='utf-8', errors='ignore') as file:
                        for line_num, line in enumerate(file, 1):
                            line = line.rstrip('\n\r')
                                
                            if use_regex:
                                try:
                                    if re.search(pattern, line, flags):
                                        matches_found += 1
                                        print(f"{Fore.GREEN}{line_num}:{Style.RESET_ALL} {line}")
                                except re.error as e:
                                    print(f"{Fore.RED}⣏!⣽ Regex error: {e}{Style.RESET_ALL}")
                                    continue
                            else:
                                search_line = line if case_sensitive else line.lower()
                                search_pattern = pattern if case_sensitive else pattern.lower()
                                    
                                if search_pattern in search_line:
                                    matches_found += 1
                                    highlighted_line = line.replace(pattern, f"{Fore.RED}{pattern}{Style.RESET_ALL}")
                                    print(f"{Fore.GREEN}{line_num}:{Style.RESET_ALL} {highlighted_line}")
                        
                    if matches_found == 0:
                        print(f"{Fore.YELLOW}No matches found{Style.RESET_ALL}")
                    else:
                        print(f"\n{Fore.GREEN}Found {matches_found} matches{Style.RESET_ALL}")
                        
                    logger.log(f"Grep search completed: {matches_found} matches in {filename}", "OK")
                        
                except PermissionError:
                    logger.log(f"Permission denied for file: {filename}", "FAIL")
                    print(f"{Fore.RED}⣏!⣽ Permission denied: {filename}{Style.RESET_ALL}")
                    continue 

                except Exception as e:
                    logger.log(f"Error in grep: {str(e)}", "FAIL")
                    print(f"{Fore.RED}⣏!⣽ Error: {str(e)}{Style.RESET_ALL}")
                    continue

            # wc
            elif shell_command == "wc":
                from pathlib import Path
                try:
                    user_input = input("⣦ File to analyze: ").strip()
                    if not user_input:
                        print(f"{Fore.RED}⣏!⣽ Filename cannot be empty!{Style.RESET_ALL}")
                        continue

                    filename = Path(user_input)
                
                    if not filename.is_absolute():
                        filename = Path.cwd() / filename
                    
                    filename = filename.resolve()

                    if not filename.exists():
                        print(f"{Fore.RED}⣏!⣽ File not found: {filename}{Style.RESET_ALL}")
                        continue

                    if filename.is_dir():
                        print(f"{Fore.RED}⣏!⣽ Path is a directory, not a file{Style.RESET_ALL}")
                        continue

                    with open(filename, 'r', encoding='utf-8', errors='ignore') as file:
                        content = file.read()

                    lines = content.count('\n')
                    words = len(content.split())
                    characters = len(content)
                    bytes_size = filename.stat().st_size

                    print(f"\n{Fore.CYAN}File statistics for: {filename}{Style.RESET_ALL}")
                    print(f"{Fore.GREEN}Lines:    {lines:8}{Style.RESET_ALL}")
                    print(f"{Fore.BLUE}Words:    {words:8}{Style.RESET_ALL}")
                    print(f"{Fore.YELLOW}Chars:    {characters:8}{Style.RESET_ALL}")
                    print(f"{Fore.MAGENTA}Bytes:    {bytes_size:8}{Style.RESET_ALL}")

                    if lines > 0:
                        avg_words_per_line = words / lines
                        avg_chars_per_line = characters / lines
                        print(f"{Fore.CYAN}Avg words/line: {avg_words_per_line:.1f}{Style.RESET_ALL}")
                        print(f"{Fore.CYAN}Avg chars/line: {avg_chars_per_line:.1f}{Style.RESET_ALL}")

                except PermissionError:
                    print(f"{Fore.RED}⣏!⣽ Permission denied: {filename}{Style.RESET_ALL}")
                except Exception as e:
                    print(f"{Fore.RED}⣏!⣽ Error: {str(e)}{Style.RESET_ALL}")

            else:
                logger.log(f"Unknown command: {shell_command}", "WARN")
                print("⣏!⣽ invalid_choice")

        except KeyboardInterrupt:
            logger.log("Shell interrupted by user", "INFO")
            print("\n⣦ Use 'exit' to leave shell")
            continue

        except EOFError:
            logger.log("Shell EOF received", "INFO")
            print("\n⣦ Exiting shell...")
            break

        except Exception as e:
            logger.log("Unexpected error in shell", "FAIL", e)
            print(Fore.RED + f"⣏!⣽ Unexpected error: {e}" + Style.RESET_ALL)
