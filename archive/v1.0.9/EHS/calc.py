from .chklib_clrscr import clear_screen
# Calc


def calc():
    """
    Calculator with basic arithmetic operations.
    
    Supports: +, -, *, /, ** (power), % (modulo), // (floor division)
    Handles division by zero and various error cases with colored messages.
    """
    from colorama import Fore, Style
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
        except OverflowError:
            print(Fore.RED + "⣏!⣽ Enter a number less!" + Style.RESET_ALL)
        except ValueError:
            print(Fore.RED + "⣏!⣽ Numbers only!" + Style.RESET_ALL)
            continue
        except Exception as e:
            print(Fore.RED + f"⣏!⣽ Error: {e}" + Style.RESET_ALL)
        if input("⣦Exit? Y/N> ").upper() == "Y":
            clear_screen()
            break


