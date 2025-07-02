import os
import time
import random
import sys
import psutil
from datetime import datetime
from packaging import version
import requests
shell_version="v1.0.4"
github_lastest_update = "https://raw.githubusercontent.com/QUIK1001/Event-Horizon/main/check_update"
def check_updates():
    try:
        response=requests.get(github_lastest_update, timeout=3)
        response.raise_for_status()
        latest_version = response.text.strip()
        if version.parse(latest_version) > version.parse(shell_version):
            print(f"Update! {shell_version} < {latest_version}")
            print("Download: https://github.com/QUIK1001/Event-Horizon")
            return True
        else:
            print("Actual version")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Error connecting: {str(e)}")
        return False
        
def clear_screen():
    os.system("clear" if os.name == "posix" else "cls")
def event_horizon():
    print("             ⣀⣴⣿⣿⣿⣿⣿⣿⣶⣄")
    print('            ⣶⣿⣿⡿⠛⠉⠉⠉⠛⢿⣿⣿⣷⣄')
    print("        ⢀⣴⣿⣿⣿⠋⡀⠀⠀⠀⠀⠀⠀⠙⣿⣿⣿⣿⣦⡀")
    print("⠀⠀⣀⣰⣶⣿⣿⣿⣿⣿⣿⣀⣀⣀⣀⣀⣀⣰⣰⣶⣿⣿⣿⣿⣿⣿⣷⣦⣤⣀")
    print(" ⠉⠉⠉⠈⠉⠛⠛⠛⠛⣿⣿⡽⣏⠉⠉⠉⠉⠉⣽⣿⠛⠉⠉⠉⠉⠉⠉⠉⠁")
    print("             ⠉⠻⣷⣦⣄⣀⣠⣴⣾⡿⠃")
    print("                 ⠛⠿⠿⠿⠟⠋")
    symbols = ['-', '\\', '|', '/']
    for i in range(8):
        sys.stdout.write('\rloading ' + symbols[i % 4])
        time.sleep(0.5)
    clear_screen()
    print("---=Event Horizon=---")
def calc():
    while True:
        try:
            calc_num1=int(input("⣦First number> "))
            calc_act=input("⣦Action +,-,*,/> ")
            calc_num2=int(input("⣦Second number> "))
            if calc_act=="+":
                print(calc_num1, calc_act, calc_num2, "equals> ", calc_num1+calc_num2)
            elif calc_act=="-":
                print(calc_num1, calc_act, calc_num2, "equals> ", calc_num1-calc_num2)
            elif calc_act=="*":
                print(calc_num1, calc_act, calc_num2, "equals> ", calc_num1*calc_num2)
            elif calc_act=="/":
                if calc_num2!=0:
                    print(calc_num1, calc_act, calc_num2, "equals> ", calc_num1/calc_num2)
                else:
                    print("⣏!⣽ DIV/0!")
        except ValueError:
            print("⣏!⣽ Numbers only!")
            continue
        if input("⣦Exit? Y/N> ")=="Y":
            clear_screen()
            break
def echo():
    echo_text=input("⣦Enter your text: ")
    print("⣦Your text:", echo_text)
    time.sleep(2)
    clear_screen()
def ascii_arts():
    ascii_cats=[
        """ /|_/|  
( o.o ) 
 > ^ < """,
        """  /|_/|
 =( °w° )=
  )   (  """
    ]
    ascii_galaxie=[
        """    . * .
  * . * . *
. * . * . * .
  * . * . *
    . * ."""
    ]
    all_ascii_arts=ascii_cats+ascii_galaxie
    print(random.choice(all_ascii_arts))
    time.sleep(2)
    clear_screen()
def os_info():
    clear_screen()
    print("             ⣀⣴⣿⣿⣿⣿⣿⣿⣶⣄")
    print('            ⣶⣿⣿⡿⠛⠉⠉⠉⠛⢿⣿⣿⣷⣄')
    print("        ⢀⣴⣿⣿⣿⠋⡀⠀⠀⠀⠀⠀⠀⠙⣿⣿⣿⣿⣦⡀")
    print("⠀⠀⣀⣰⣶⣿⣿⣿⣿⣿⣿⣀⣀⣀⣀⣀⣀⣰⣰⣶⣿⣿⣿⣿⣿⣿⣷⣦⣤⣀")
    print(" ⠉⠉⠉⠈⠉⠛⠛⠛⠛⣿⣿⡽⣏⠉⠉⠉⠉⠉⣽⣿⠛⠉⠉⠉⠉⠉⠉⠉⠁")
    print("             ⠉⠻⣷⣦⣄⣀⣠⣴⣾⡿⠃")
    print("                 ⠛⠿⠿⠿⠟⠋")
    print("Event Horizon")
    print("shell ver> v1.0.4")
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
        cmd=input("#> ")
        if cmd=="help":
            print("⣦help-show help")
            print("⣦clear-clear screen")
            print("⣦info-OS info")
            print("⣦exit-exit to menu\n")
            print("⣦mkdir-create folder")
            print("⣦rmdir-remove folder")
            print("⣦time-show current time")
            print("⣦perf-show CPU & RAM usage")
        elif cmd=="exit":
            clear_screen()
            break
        elif cmd=="clear":
            clear_screen()
            print("⣦Event Horizon shell\n")
        elif cmd=="info":
            print("\n⣦Event Horizon")
            print("⣦shell ver> v1.0.4")
            print("⣦by quik\n")
        elif cmd=="mkdir":
            shell_mkdir=input("enter folder name> ")
            shell_par_mkdir=input("create in current dir? Y/N> ")
            if shell_par_mkdir=="Y":
                shell_par_mkdir="."
            else:
                shell_par_mkdir=os.path.expanduser("~")
            shell_mk_path=os.path.join(shell_par_mkdir, shell_mkdir)
            try:
                os.mkdir(shell_mk_path)
                print("folder", {shell_mkdir},"created in", {shell_par_mkdir})
            except FileExistsError:
                print("⣏!⣽", {shell_mk_path},"already exists!")
        elif cmd=="rmdir":
            shell_rm_path = input("Enter folder path to delete: ")
            if not os.path.exists(shell_rm_path):
                print("⣏!⣽ folder doesn't exist!")
                return
            else:
                os.rmdir(shell_rm_path)
                print("folder", {shell_rm_path},"deleted successfully!")
        elif cmd=="time":
            shell_time=datetime.now()
            shell_for_time=shell_time.strftime("%Y-%m-%d %H:%M:%S:%MS")
            print(shell_for_time)
        elif cmd=="perf":
            clear_screen()
            print("⣦ System monitor started. Press Ctrl+C to stop.")
            time.sleep(2.5)
            try:
                while True:
                    clear_screen()
                    print(f"CPU: {psutil.cpu_percent()}% \nRAM: {psutil.virtual_memory().percent}%")
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n⣦ Monitor stopped.")
                clear_screen()
                print("⣦Event Horizon shell\n")
def timer():
    timer_time = int(input("enter clock time in seconds> "))
    print(timer_time)
    for i in range(timer_time):
        time.sleep(1)
        print(timer_time-i-1)
    print("Timer finished!")
    time.sleep(1.5)
    clear_screen()
def menu():
    event_horizon()
    while True:
        print("1. calc\n2. echo\n3. ASCII arts\n4. shell info\n5. shutdown\n6. shell\n7. timer\n8. check updates")
        choice = input("select> ")
        if choice=="1":
            clear_screen()
            calc()
        elif choice=="2":
            clear_screen()
            echo()
        elif choice=="3":
            clear_screen()
            ascii_arts()
        elif choice=="4":
            clear_screen()
            os_info()
        elif choice=="5":
            clear_screen()
            shutdown()
        elif choice=="6":
            shell()
        elif choice=="7":
            clear_screen()
            timer()
        elif choice=="8":
            clear_screen()
            check_updates()
            time.sleep(2)
            clear_screen()
        else:
            print("⣏!⣽ invalid_choice")
            time.sleep(1)
            clear_screen()
menu()