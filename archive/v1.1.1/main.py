import sys
import time

def menu():
    """
    Main menu interface for Event Horizon Shell.
    
    Handles user navigation between all available tools and features.
    Initializes dependencies and displays the main menu loop.
    """
    from EHS.chklib_clrscr import check_lib, clear_screen
    if not check_lib():
        print(" [FAIL]")
        print("Critical dependencies missing. Exiting.")
        sys.exit(1)
    print("Continue starting...")
    
    print("Importing libraries...\n", end="")
    print("Loading modules...\n", end="")
    from EHS.logger import logger 
    from EHS.weather import weather 
    from colorama import Fore, Style 
    from EHS.art import event_horizon
    from EHS.calc import calc
    from EHS.chkupd_speedtest import check_updates, speed_test
    from EHS.shell import shell
    from EHS.editors import text_editor  
    from EHS.tools import ascii_arts, changelog, echo, shell_info, shutdown, timer, os_info, toggle_logs, toggle_theme
    from EHS.theme_manager import apply_theme
    from EHS.constants import config  
    print("Starting EHS...\n", end="")
    event_horizon()
    logger.log("Displaying changelog...", "INFO")
    changelog()
    logger.log("Changelog displayed", "OK")
    clear_screen()    
    apply_theme(config)
    print("---=Event Horizon=---")
    while True:
        print(
            Style.BRIGHT +
            "1. calc\n2. echo\n3. ASCII arts\n"
            "4. shell info\n5. OS info\n6. shell"
            "\n7. timer\n8. check updates\n9. check internet speed"
            "\n10. weather\n11. toggle logs\n12. text editors"
            "\n13. toggle theme\n14. shutdown"
            + Style.RESET_ALL)
        
        choice = input(Fore.RED + "select> " + Style.RESET_ALL)
        logger.log(f"User selected option: {choice}", "DEBUG")
        
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
            os_info()
            clear_screen()
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
        elif choice == "10":
            clear_screen()
            weather()
        elif choice == "11":
            clear_screen()
            toggle_logs()  
        elif choice == "12":
            clear_screen()
            text_editor()  
        elif choice == "13":
            clear_screen()
            toggle_theme()
            clear_screen()
        elif choice == "14":
            clear_screen()
            shutdown()
        else:
            logger.log(f"Invalid menu choice: {choice}", "WARN")
            print("⣏!⣽ invalid_choice")
            time.sleep(1)
            clear_screen()


if __name__ == "__main__":
    menu()
