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
