from .chklib_clrscr import clear_screen
from .logger import logger

def calc():
    """Calculator with basic arithmetic operations."""
    from colorama import Fore, Style
    
    logger.log("Starting calculator", "INFO")
    
    while True:
        try:
            calc_num1 = float(input("⣦First number> "))
            calc_act = input("⣦Action +,-,*,/,**,%,//> ")
            calc_num2 = float(input("⣦Second number> "))
            
            logger.log(f"Calculation: {calc_num1} {calc_act} {calc_num2}", "INFO")
            
            if calc_act == "+":
                result = calc_num1 + calc_num2
                logger.log(f"Addition result: {result}", "OK")
                print(
                    calc_num1,
                    calc_act,
                    calc_num2,
                    Fore.GREEN + "equals> " + Style.RESET_ALL,
                    result,
                )
            elif calc_act == "-":
                result = calc_num1 - calc_num2
                logger.log(f"Subtraction result: {result}", "OK")
                print(
                    calc_num1,
                    calc_act,
                    calc_num2,
                    Fore.GREEN + "equals> " + Style.RESET_ALL,
                    result,
                )
            elif calc_act == "*":
                result = calc_num1 * calc_num2
                logger.log(f"Multiplication result: {result}", "OK")
                print(
                    calc_num1,
                    calc_act,
                    calc_num2,
                    Fore.GREEN + "equals> " + Style.RESET_ALL,
                    result,
                )
            elif calc_act == "**":
                result = calc_num1**calc_num2
                logger.log(f"Exponentiation result: {result}", "OK")
                print(
                    calc_num1,
                    calc_act,
                    calc_num2,
                    Fore.GREEN + "equals> " + Style.RESET_ALL,
                    result,
                )
            elif calc_act == "%":
                result = calc_num1 % calc_num2
                logger.log(f"Modulo result: {result}", "OK")
                print(
                    calc_num1,
                    calc_act,
                    calc_num2,
                    Fore.GREEN + "equals> " + Style.RESET_ALL,
                    result,
                )
            elif calc_act == "//":
                if calc_num2 == 0:
                    logger.log("Division by zero attempted in floor division", "FAIL")
                    print(Fore.RED + "⣏!⣽ DIV/0!" + Style.RESET_ALL)
                    continue
                result = calc_num1 // calc_num2
                logger.log(f"Floor division result: {result}", "OK")
                print(
                    calc_num1,
                    calc_act,
                    calc_num2,
                    Fore.GREEN + "equals> " + Style.RESET_ALL,
                    result,
                )
            elif calc_act == "/":
                if calc_num2 == 0:
                    logger.log("Division by zero attempted", "FAIL")
                    print(Fore.RED + "⣏!⣽ DIV/0!" + Style.RESET_ALL)
                else:
                    result = calc_num1 / calc_num2
                    logger.log(f"Division result: {result}", "OK")
                    print(
                        calc_num1,
                        calc_act,
                        calc_num2,
                        Fore.GREEN + "equals> " + Style.RESET_ALL,
                        result,
                    )
            else:
                logger.log(f"Unknown operation: {calc_act}", "FAIL")
                print(Fore.RED + f"⣏!⣽ Unknown operation: {calc_act}" + Style.RESET_ALL)
                
        except OverflowError as e:
            logger.log("Overflow error in calculation", "FAIL", e)
            print(Fore.RED + "⣏!⣽ Enter a number less!" + Style.RESET_ALL)
        except ValueError as e:
            logger.log("Invalid input value", "FAIL", e)
            print(Fore.RED + "⣏!⣽ Numbers only!" + Style.RESET_ALL)
        except ZeroDivisionError as e:
            logger.log("Division by zero error", "FAIL", e)
            print(Fore.RED + "⣏!⣽ Division by zero!" + Style.RESET_ALL)
        except Exception as e:
            logger.log("Unexpected error in calculator", "FAIL", e)
            print(Fore.RED + f"⣏!⣽ Error: {e}" + Style.RESET_ALL)
            
        exit_choice = input("⣦Exit? Y/N> ").upper()
        if exit_choice == "Y":
            logger.log("Exiting calculator", "INFO")
            clear_screen()
            break
        else:
            logger.log("Continuing calculator", "DEBUG")
