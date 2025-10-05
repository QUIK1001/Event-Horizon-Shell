# Move to EHS/ folder and do not use such plugin names: 
# aliases.py, archiver.py, art.py, calc.py, chklib_clrscr.py, 
# chkupd_speedtest.py, constants.py, editors.py, init.py, 
# logger.py, plugin_sys.py, shell.py, theme_manager.py, 
# tools.py, tutorial.py, weather.py and main.py.

def main():   #← Important 
# edit this part ↓
    try:
        from colorama import Fore, Style
        print(f"{Fore.CYAN}=== Login ==={Style.RESET_ALL}")
        name = input("Enter your name> ")
        age = int(input("Enter your age> "))
        print(f"{Fore.GREEN}Your name> {name}. Your age> {age}{Style.RESET_ALL}")
    except ValueError:
        print(f"{Fore.RED}Error: Please enter a valid age!{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Unexpected error: {e}{Style.RESET_ALL}")

    input("Press Enter to continue...")


# Important ↓
if __name__ == "__main__":
    main()
