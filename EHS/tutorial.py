from colorama import Fore, Style
from art import text2art
from .chklib_clrscr import clear_screen
import time

def tutorial():
    """Display an interactive tutorial menu for EHS guide."""
    try:
        while True:
            print(text2art("EHS Guide"))
            print(Fore.GREEN + "\nWelcome to Event Horizon Shell Tutorial!")
            print("Learn about all available features and how to use them.\n" + Style.RESET_ALL)

            print(Fore.LIGHTYELLOW_EX + "Available tutorials:\n")
            print(
                " 1. Calculator (calc)\n",
                "2. Echo command\n",
                "3. ASCII Arts\n", 
                "4. Shell Information\n",
                "5. OS Information\n",
                "6. EHS Shell\n",
                "7. Timer\n",
                "8. Check Updates\n",
                "9. Internet Speed Test\n",
                "10. Weather\n",
                "11. Toggle Logs\n",
                "12. Text Editors\n",
                "13. Toggle Theme\n",
                "14. Shutdown\n" \
                "15. ping\n",
                "16. Exit Tutorial\n"
            )
            
            print(Style.RESET_ALL)
            
            try:
                choice = input(Fore.LIGHTRED_EX + "Select tutorial (1-16) >> " + Style.RESET_ALL).strip()
  
                if choice == "1":
                    clear_screen()
                    print(Fore.CYAN + "=== Calculator Tutorial ===" + Style.RESET_ALL)
                    print(Fore.YELLOW + "\nThe calculator supports basic arithmetic operations:")
                    print("  + Plus")
                    print("  - minus") 
                    print("  * Multiplication")
                    print("  / Division")
                    print("  ** Power")
                    print("  % Modulo")
                    print("  // Floor division\n")
                    print(Fore.GREEN + "Example usage:")
                    print("1. Select option 1 from main menu")
                    print("2. Enter first number (e.g., 10)")
                    print("3. Choose operation (e.g., +)")
                    print("4. Enter second number (e.g., 5)")
                    print("5. See result: 10 + 5 = 15")
                    
                elif choice == "2":
                    clear_screen()
                    print(Fore.CYAN + "=== Echo Tutorial ===" + Style.RESET_ALL)
                    print(Fore.YELLOW + "\nEcho repeats any text you enter.")
                    print(Fore.GREEN + "\nExample usage:")
                    print("1. Select option 2 from main menu") 
                    print("2. Enter any text (e.g., 'Hello World!')")
                    print("3. See output: ⣦Your text: Hello World!")
                    
                elif choice == "3":
                    clear_screen()
                    print(Fore.CYAN + "=== ASCII Arts Tutorial ===" + Style.RESET_ALL)
                    print(Fore.YELLOW + "\nDisplays random ASCII art from a collection.")
                    print(Fore.GREEN + "\nExample usage:")
                    print("1. Select option 3 from main menu")
                    print("2. View random ASCII art (cats, galaxies)")
                    print("3. Press Enter to return to menu")
                    
                elif choice == "4":
                    clear_screen()
                    print(Fore.CYAN + "=== Shell Information Tutorial ===" + Style.RESET_ALL)
                    print(Fore.YELLOW + "\nShows information about Event Horizon Shell.")
                    print(Fore.GREEN + "\nExample usage:")
                    print("1. Select option 4 from main menu")
                    print("2. View shell version, author, and ASCII logo")
                    print("3. Press Enter to return")
                    
                elif choice == "5":
                    clear_screen()
                    print(Fore.CYAN + "=== OS Information Tutorial ===" + Style.RESET_ALL)
                    print(Fore.YELLOW + "\nDisplays detailed system information:")
                    print("  • Operating System details")
                    print("  • CPU information (cores, frequency)")
                    print("  • RAM size and usage")
                    print("  • Python version")
                    print("  • System architecture")
                    print(Fore.GREEN + "\nExample usage:")
                    print("1. Select option 5 from main menu")
                    print("2. View all system specifications")
                    print("3. Press Enter to return")
                    
                elif choice == "6":
                    clear_screen()
                    def show_shell_tutorial():
                        print(Fore.CYAN + "=== EHS Shell Tutorial ===" + Style.RESET_ALL)
                        print(Fore.YELLOW + "\nShell with file operations:")
                        print("1.  • help - Show available commands")
                        print("2.  • clear - Clear screen")
                        print("3.  • info - Shell information")
                        print("4.  • exit - Return to main menu")
                        print("5.  • mkdir - Create directory")
                        print("6.  • rmdir - Remove directory")
                        print("7.  • dir - List directory contents")
                        print("8.  • cd - Change directory")
                        print("9.  • rmfile - Remove file")
                        print("10. • find - Search files/directories")
                        print("11. • alias - Manage command aliases")
                        print("12. • move - Move files/folders")
                        print("13. • copy - Copy files/folders")
                        print("14. • rename - Rename files/folders")
                        print("15. • time - Show current time")
                        print("16. • perf - System performance monitor")
                        print(Fore.GREEN + "\nExample usage:")
                        print("1. Select option 6 from main menu")
                        print("2. Type 'help' to see all commands")
                        print("3. Use commands like 'dir' or 'mkdir test'")
                        print("4. Type 'exit' to return to main menu")
                    show_shell_tutorial()
                    while True:
                        shell_choice = input(Fore.LIGHTRED_EX + "\nSelect shell tutorial (1-16) or '17' to exit >> " + Style.RESET_ALL).strip()
                        
                        if shell_choice == "17":
                            break
                            
                        elif shell_choice == "1":
                            clear_screen()
                            print(text2art("help"))
                            print(Fore.CYAN + "Command Tutorial" + Style.RESET_ALL)
                            print(Fore.YELLOW + "\nDisplays all available commands with brief descriptions")
                            print(Fore.GREEN + "Enter: help")
                            print("Output: Shows list of all commands with their functions")
                            
                        elif shell_choice == "2":
                            clear_screen()
                            print(text2art("clear"))
                            print(Fore.CYAN + "Command Tutorial" + Style.RESET_ALL)
                            print(Fore.YELLOW + "\nClears the terminal screen")
                            print(Fore.GREEN + "\nEnter: clear")
                            print("Output: Screen cleared")
                            
                        elif shell_choice == "3":
                            clear_screen()
                            print(text2art("info"))
                            print(Fore.CYAN + "Command Tutorial" + Style.RESET_ALL)
                            print(Fore.YELLOW + "\nDisplays information about the shell environment")
                            print(Fore.GREEN + "\nEnter: info")
                            print("Output: Shows shell version")
                            
                        elif shell_choice == "4":
                            clear_screen()
                            print(text2art("exit"))
                            print(Fore.CYAN + "Command Tutorial" + Style.RESET_ALL)
                            print(Fore.YELLOW + "\nExits the shell and returns to main menu")
                            print(Fore.GREEN + "\nEnter: exit")
                            print("Output: Returns to main menu interface")
                            
                        elif shell_choice == "5":
                            clear_screen()
                            print(text2art("mkdir"))
                            print(Fore.CYAN + "Command Tutorial" + Style.RESET_ALL)
                            print(Fore.YELLOW + "\nCreates a new directory")
                            print(Fore.GREEN + "\nEnter: mkdir->myfolder")
                            print("Output: Creates directory 'myfolder'")
                            
                        elif shell_choice == "6":
                            clear_screen()
                            print(text2art("rmdir"))
                            print(Fore.CYAN + "Command Tutorial" + Style.RESET_ALL)
                            print(Fore.YELLOW + "\nRemoves an empty directory")
                            print(Fore.GREEN + "\nEnter: rmdir->myfolder")
                            print("Output: Removes directory 'myfolder' if empty")
                            
                        elif shell_choice == "7":
                            clear_screen()
                            print(text2art("dir"))
                            print(Fore.CYAN + "Command Tutorial" + Style.RESET_ALL)
                            print(Fore.YELLOW + "\nLists files and directories in current location")
                            print(Fore.GREEN + "\nEnter: dir")
                            print("Output: Shows list of files and folders in current directory")
                            
                        elif shell_choice == "8":
                            clear_screen()
                            print(text2art("cd"))
                            print(Fore.CYAN + "Command Tutorial" + Style.RESET_ALL)
                            print(Fore.YELLOW + "\nChanges current working directory")
                            print(Fore.GREEN + "\nEnter: cd->myfolder1")
                            print("Output: Changes to 'myfolder1' directory")
                            
                        elif shell_choice == "9":
                            clear_screen()
                            print(text2art("rmfile"))
                            print(Fore.CYAN + "Command Tutorial" + Style.RESET_ALL)
                            print(Fore.YELLOW + "\nRemoves a file")
                            print(Fore.GREEN + "\nEnter: rmfile->menu.exe")
                            print("Output: Deletes file 'menu.exe'")
                            
                        elif shell_choice == "10":
                            clear_screen()
                            print(text2art("find"))
                            print(Fore.CYAN + "Command Tutorial" + Style.RESET_ALL)
                            print(Fore.YELLOW + "\nSearches for files or directories")
                            print(Fore.GREEN + "\nEnter: find->menu.exe")
                            print("Output: Finds all files/directories containing 'menu.exe'")
                            
                        elif shell_choice == "11":
                            clear_screen()
                            print(text2art("alias"))
                            print(Fore.CYAN + "Command Tutorial" + Style.RESET_ALL)
                            print(Fore.YELLOW + "\nCreates or manages command aliases")
                            print("Built-in aliases: ls->dir, cls->clear, rm->rmfile (more in shell->alias)")
                            print("• alias - Show all aliases")
                            print("• alias add <name> <command> - Add new alias")
                            print("• alias remove <name> - Remove alias")
                            print(Fore.GREEN + "\nEnter: alias add ls dir")
                            print("Output: Creates alias 'ls' for 'dir' command")       
                                    
                        elif shell_choice == "12":
                            clear_screen()
                            print(text2art("move"))
                            print(Fore.CYAN + "Command Tutorial" + Style.RESET_ALL)
                            print(Fore.YELLOW + "\nMoves files or directories to new location")
                            print(Fore.GREEN + "\nEnter: move->menu.exe->C:\\")
                            print("Output: Moves menu.exe to 'C:\\'")
                            
                        elif shell_choice == "13":
                            clear_screen()
                            print(text2art("copy"))
                            print(Fore.CYAN + "Command Tutorial" + Style.RESET_ALL)
                            print(Fore.YELLOW + "\nCopies files or directories")
                            print(Fore.GREEN + "\nEnter: copy->menu.exe->C:\\")
                            print("Output: Creates copy of menu.exe in 'C:\\'")
                            
                        elif shell_choice == "14":
                            clear_screen()
                            print(text2art("rename"))
                            print(Fore.CYAN + "Command Tutorial" + Style.RESET_ALL)
                            print(Fore.YELLOW + "\nRenames files or directories")
                            print(Fore.GREEN + "\nEnter: rename->menu.exe->EHS")
                            print("Output: Renames file from 'menu.exe' to 'EHS'")
                            
                        elif shell_choice == "15":
                            clear_screen()
                            print(text2art("time"))
                            print(Fore.CYAN + "Command Tutorial" + Style.RESET_ALL)
                            print(Fore.YELLOW + "\nShows current date and time")
                            print(Fore.GREEN + "\nEnter: time")
                            print("Output: Displays current date and time")
                            
                        elif shell_choice == "16":
                            clear_screen()
                            print(text2art("perf"))
                            print(Fore.CYAN + "Command Tutorial" + Style.RESET_ALL)
                            print(Fore.YELLOW + "\nMonitors system performance")
                            print(Fore.GREEN + "\nEnter: perf")
                            print("Output: Shows CPU usage, memory usage")
                            
                        else:
                            clear_screen()
                            print(text2art("Invalid\n choice!"))
                            print(Fore.RED + "Please select 1-16 or 17 to exit." + Style.RESET_ALL)
                            time.sleep(1.5)
                            clear_screen()
                            show_shell_tutorial()
                            continue

                        print(Fore.RED + "\nNote! The separator '->' means that you need to enter commands one by one" + Style.RESET_ALL)
                        input(Fore.LIGHTBLUE_EX + "\nPress Enter to continue..." + Style.RESET_ALL)
                        clear_screen()
                        
                        show_shell_tutorial()
                    
                elif choice == "7":
                    clear_screen()
                    print(Fore.CYAN + "=== Timer Tutorial ===" + Style.RESET_ALL)
                    print(Fore.YELLOW + "\nCountdown timer with seconds precision.")
                    print(Fore.GREEN + "\nExample usage:")
                    print("1. Select option 7 from main menu")
                    print("2. Enter time in seconds (e.g., 10)")
                    print("3. Watch countdown from 10 to 0")
                    print("4. Get notification when timer finishes")
                    
                elif choice == "8":
                    clear_screen()
                    print(Fore.CYAN + "=== Check Updates Tutorial ===" + Style.RESET_ALL)
                    print(Fore.YELLOW + "\nChecks for new versions of EHS.")
                    print(Fore.GREEN + "\nExample usage:")
                    print("1. Select option 8 from main menu")
                    print("2. System connects to GitHub to check version")
                    print("3. Shows if update is available")
                    print("4. Offers to open Telegram for download")
                    
                elif choice == "9":
                    clear_screen()
                    print(Fore.CYAN + "=== Internet Speed Test Tutorial ===" + Style.RESET_ALL)
                    print(Fore.YELLOW + "\nTests your internet download speed.")
                    print(Fore.GREEN + "\nExample usage:")
                    print("1. Select option 9 from main menu")
                    print("2. System downloads test file")
                    print("3. Shows progress bar and speed")
                    print("4. Displays total time and average speed")
                    
                elif choice == "10":
                    clear_screen()
                    print(Fore.CYAN + "=== Weather Tutorial ===" + Style.RESET_ALL)
                    print(Fore.YELLOW + "\nGets current weather for any city.")
                    print(Fore.GREEN + "\nExample usage:")
                    print("1. Select option 10 from main menu")
                    print("2. Enter city name (e.g., 'London' or 'Moscow')")
                    print("3. View temperature, humidity, wind speed, conditions")
                    
                elif choice == "11":
                    clear_screen()
                    print(Fore.CYAN + "=== Toggle Logs Tutorial ===" + Style.RESET_ALL)
                    print(Fore.YELLOW + "\nEnables/disables system logging.")
                    print("Logs are saved to ~/.ehs_logs/ehs_log.txt")
                    print(Fore.GREEN + "\nExample usage:")
                    print("1. Select option 11 from main menu")
                    print("2. System toggles logging on/off")
                    print("3. Shows current status (ENABLED/DISABLED)")
                    print("4. Note: will reset to 'Enabled' on restart")
                    
                elif choice == "12":
                    clear_screen()
                    print(Fore.CYAN + "=== Text Editors Tutorial ===" + Style.RESET_ALL)
                    print(Fore.YELLOW + "\nLaunches system text editors:")
                    print("Windows: Notepad, Wordpad")
                    print("Linux: Nano, Vim, Vi, Gedit")
                    print(Fore.GREEN + "\nExample usage:")
                    print("1. Select option 12 from main menu")
                    print("2. Enter filename to edit/create")
                    print("3. Choose from available editors")
                    print("4. Edit file and return to EHS when done")
                    
                elif choice == "13":
                    clear_screen()
                    print(Fore.CYAN + "=== Toggle Theme Tutorial ===" + Style.RESET_ALL)
                    print(Fore.YELLOW + "\nChanges color theme of EHS interface.")
                    print("Available themes: default, dark, light, matrix")
                    print(Fore.GREEN + "\nExample usage:")
                    print("1. Select option 13 from main menu")
                    print("2. System cycles through available themes")
                    print("3. Shows preview of colors for each theme")
                    print("4. Theme persists until changed again")
                    
                elif choice == "14":
                    clear_screen()
                    print(Fore.CYAN + "=== Shutdown Tutorial ===" + Style.RESET_ALL)
                    print(Fore.GREEN + "\nExample usage:")
                    print("1. Select option 14 from main menu")
                    print("2. *shutdown animation*")
                    print("3. Exits to terminal")

                elif choice == "15":
                    clear_screen()
                    print(Fore.CYAN + "=== ping Tutorial ===" + Style.RESET_ALL)
                    print(Fore.GREEN + "\nExample usage:")
                    print("1. Select option 15 from main menu")
                    print("2. Enter host name (e.g., 'google.com' or 'https://www.google.com' if you want :D)")
                    print("3. See the result")

                elif choice == "16":
                    print(Fore.YELLOW + "Exiting tutorial..." + Style.RESET_ALL)
                    time.sleep(0.3)
                    break
                    
                else:
                    print(Fore.RED + "Invalid choice! Please select 1-16." + Style.RESET_ALL)
                    time.sleep(1)
                    clear_screen()
                    continue
                
                input(Fore.LIGHTBLUE_EX + "\nPress Enter to continue..." + Style.RESET_ALL)
                clear_screen()
                
            except KeyboardInterrupt:
                print(Fore.YELLOW + "\n\nTutorial cancelled." + Style.RESET_ALL)
                time.sleep(1)
                break
            except EOFError:
                print(Fore.YELLOW + "\n\nEnd of input." + Style.RESET_ALL)
                time.sleep(1)
                break
    
    except Exception as e:
        print(Fore.RED + f"An unexpected error occurred: {e}" + Style.RESET_ALL)
        return None
