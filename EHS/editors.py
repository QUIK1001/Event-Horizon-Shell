import subprocess
import os
from .chklib_clrscr import clear_screen
from .logger import logger
from colorama import Fore, Style

def text_editor():
    """
    Launch system text editor to create/edit text files with user choice
    """
    logger.log("Starting external text editor", "INFO")
    
    try:
        filename = input("Enter filename: ")
        filename = os.path.expanduser(filename)
        
        if not os.path.exists(filename):
            create_new = input(f"File '{filename}' doesn't exist. Create new? (Y/N): ").upper()
            if create_new != 'Y':
                logger.log("User cancelled file creation", "INFO")
                return
            open(filename, 'a').close()

        available_editors = []

        def is_tool_available(name):
            try:
                if os.name == 'nt':
                    result = subprocess.run(['where', name], capture_output=True, text=True, shell=True)
                    return result.returncode == 0
                else:
                    result = subprocess.run(['which', name], capture_output=True, text=True)
                    return result.returncode == 0
            except:
                return False

        if os.name == 'nt':
            editors_priority = [
                'notepad',
                'write',
            ]
        else:
            editors_priority = [
                'nano',
                'vim',
                'vi',
                'gedit',
            ]
        
        for editor in editors_priority:
            if is_tool_available(editor):
                available_editors.append(editor)
                if os.name != 'nt' and len(available_editors) >= 3:
                    break
        
        if not available_editors:
            print("No text editor found! Please install one (nano, vim, notepad, etc.)")
            logger.log("No text editor available", "FAIL")
            return

        if len(available_editors) > 1:
            print("\nAvailable text editors:")
            for i, editor in enumerate(available_editors, 1):
                print(f"{i}. {editor}")
            
            try:
                choice = input(f"Choose editor (1-{len(available_editors)}, Enter for default [{available_editors[0]}]): ")
                if choice.strip():
                    editor_index = int(choice) - 1
                    if 0 <= editor_index < len(available_editors):
                        editor_cmd = available_editors[editor_index]
                    else:
                        print("Invalid choice, using default.")
                        editor_cmd = available_editors[0]
                else:
                    editor_cmd = available_editors[0]
            except ValueError:
                print("Invalid input, using default.")
                editor_cmd = available_editors[0]
        else:
            editor_cmd = available_editors[0]
        
        print(f"Opening {filename} with {editor_cmd}...")

        try:
            subprocess.run([editor_cmd, filename], check=True)
            logger.log(f"File edited with {editor_cmd}: {filename}", "OK")
            
        except subprocess.CalledProcessError as e:
            logger.log(f"Editor {editor_cmd} failed: {str(e)}", "FAIL")
            print(f"Editor {editor_cmd} failed to open the file.")
        
    except KeyboardInterrupt:
        logger.log("Text editor cancelled by user", "INFO")
        print("\nOperation cancelled.")
    except PermissionError:
        logger.log(f"Permission denied for file operation: {PermissionError}", "FAIL")
        print(f"{Fore.RED}⣏!⣽ Permission denied: {filename}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Check file permissions or run as administrator\sudo{Style.RESET_ALL}")
    except Exception as e:
        logger.log(f"Error in text editor: {str(e)}", "FAIL")
        print(f"Error: {str(e)}")
    finally:
        input("Press Enter to continue")
        clear_screen()
