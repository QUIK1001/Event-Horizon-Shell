import os
import random
import sys
import time


def clear_screen():
    os.system("clear" if os.name == "posix" else "cls")


def event_horizon():
    print("             ⣀⣴⣿⣿⣿⣿⣿⣿⣶⣄")
    print("            ⣶⣿⣿⡿⠛⠉⠉⠉⠛⢿⣿⣿⣷⣄")
    print("        ⢀⣴⣿⣿⣿⠋⡀⠀⠀⠀⠀⠀⠀⠙⣿⣿⣿⣿⣦⡀")
    print("⠀⠀⣀⣰⣶⣿⣿⣿⣿⣿⣿⣀⣀⣀⣀⣀⣀⣰⣰⣶⣿⣿⣿⣿⣿⣿⣷⣦⣤⣀")
    print(" ⠉⠉⠉⠈⠉⠛⠛⠛⠛⣿⣿⡽⣏⠉⠉⠉⠉⠉⣽⣿⠛⠉⠉⠉⠉⠉⠉⠉⠁")
    print("             ⠉⠻⣷⣦⣄⣀⣠⣴⣾⡿⠃")
    print("                 ⠛⠿⠿⠿⠟⠋")
    symbols = ["-", "\\", "|", "/"]
    for i in range(8):
        sys.stdout.write("\rloading " + symbols[i % 4])
        time.sleep(0.5)
    clear_screen()
    print("---=Event Horizon=---")


def calc():
    while True:
        try:
            calc_num1 = int(input("⣦First number> "))
            calc_act = input("⣦Action +,-,*,/> ")
            calc_num2 = int(input("⣦Second number> "))
            if calc_act == "+":
                print(calc_num1, calc_act, calc_num2, "equals> ", calc_num1 + calc_num2)
            elif calc_act == "-":
                print(calc_num1, calc_act, calc_num2, "equals> ", calc_num1 - calc_num2)
            elif calc_act == "*":
                print(calc_num1, calc_act, calc_num2, "equals> ", calc_num1 * calc_num2)
            elif calc_act == "/":
                if calc_num2 != 0:
                    print(
                        calc_num1,
                        calc_act,
                        calc_num2,
                        "equals> ",
                        calc_num1 / calc_num2,
                    )
                else:
                    print("⣏!⣽ DIV/0!")
        except ValueError:
            print("⣏!⣽ Numbers only!")
            continue
        if input("⣦Exit? Y/N> ") == "Y":
            clear_screen()
            break


def echo():
    echo_text = input("⣦Enter your text: ")
    print("⣦Your text:", echo_text)
    time.sleep(2)
    clear_screen()


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
    all_ascii_arts = ascii_cats + ascii_galaxie
    print(random.choice(all_ascii_arts))
    time.sleep(2)
    clear_screen()


def os_info():
    clear_screen()
    print("             ⣀⣴⣿⣿⣿⣿⣿⣿⣶⣄")
    print("            ⣶⣿⣿⡿⠛⠉⠉⠉⠛⢿⣿⣿⣷⣄")
    print("        ⢀⣴⣿⣿⣿⠋⡀⠀⠀⠀⠀⠀⠀⠙⣿⣿⣿⣿⣦⡀")
    print("⠀⠀⣀⣰⣶⣿⣿⣿⣿⣿⣿⣀⣀⣀⣀⣀⣀⣰⣰⣶⣿⣿⣿⣿⣿⣿⣷⣦⣤⣀")
    print(" ⠉⠉⠉⠈⠉⠛⠛⠛⠛⣿⣿⡽⣏⠉⠉⠉⠉⠉⣽⣿⠛⠉⠉⠉⠉⠉⠉⠉⠁")
    print("             ⠉⠻⣷⣦⣄⣀⣠⣴⣾⡿⠃")
    print("                 ⠛⠿⠿⠿⠟⠋")
    print("Event Horizon")
    print("OS ver> v1.0.2")
    print("by quik")
    time.sleep(2.5)
    clear_screen()


def shutdown():
    print("⠋Shutting down...")
    time.sleep(1)
    clear_screen()
    sys.exit()


def shell():
    clear_screen()
    print("Event Horizon shell")
    print("⣦ Type 'help' for commands, type 'exit' for exit\n")
    while True:
        cmd = input("#> ")
        if cmd == "help":
            print("⣦help-show help")
            print("⣦clear-clear screen")
            print("⣦info-OS info")
            print("⣦exit-exit to menu\n")
            print("⣦mkdir-create folder")
            print("⣦rmdir-remove folder")
        elif cmd == "exit":
            clear_screen()
            break
        elif cmd == "clear":
            clear_screen()
            print("⣦Event Horizon shell\n")
        elif cmd == "info":
            print("\n⣦Event Horizon")
            print("⣦OS Ver> v1.0.2")
            print("⣦by quik\n")
        elif cmd == "mkdir":
            shell_mkdir = input("enter folder name> ")
            shell_par_mkdir = input("create in current dir? Y/N> ")
            if shell_par_mkdir == "Y":
                shell_par_mkdir = "."
            else:
                shell_par_mkdir = os.path.expanduser("~")
            shell_mk_path = os.path.join(shell_par_mkdir, shell_mkdir)
            try:
                os.mkdir(shell_mk_path)
                print("folder", {shell_mkdir}, "created in", {shell_par_mkdir})
            except FileExistsError:
                print("⣏!⣽", {shell_mk_path}, "already exists!")
        elif cmd == "rmdir":
            shell_rm_path = input("Enter folder path to delete: ")
            if not os.path.exists(shell_rm_path):
                print("⣏!⣽ folder does not exist!")
                return
            else:
                os.rmdir(shell_rm_path)
                print("folder", {shell_rm_path}, "deleted successfully!")


def menu():
    event_horizon()
    while True:
        print("1. calc\n2. echo\n3. ASCII arts\n4. OS info\n5. shutdown\n6. shell")
        choice = input("select> ")
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
            os_info()
        elif choice == "5":
            clear_screen()
            shutdown()
        elif choice == "6":
            shell()
        else:
            print("⣏!⣽ invalid_choice")
            time.sleep(1)
            clear_screen()


menu()
