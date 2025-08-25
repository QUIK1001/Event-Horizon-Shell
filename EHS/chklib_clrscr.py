import os
import subprocess
import sys
import time
from .logger import logger

REQUIRED_LIB = ["psutil", "requests", "colorama", "packaging", "art"]

# Clear screen
def clear_screen():
    """
    Clear the terminal screen.
    """
    os.system("clear" if os.name == "posix" else "cls")

# Check lib
def check_lib():
    """
    Check and install missing Python dependencies.
    Verifies required libraries are installed and offers to install them
    using pip or upm package managers. Exits if critical dependencies missing.
    """
    logger.log("Checking required libraries", "INFO")
    time.sleep(0.2)
    chk_lib_missing_packages = []
    
    for package in REQUIRED_LIB:
        try:
            __import__(package)
            logger.log(f"Library found: {package}", "DEBUG")
        except ImportError:
            logger.log(f"Library missing: {package}", "WARN")
            chk_lib_missing_packages.append(package)

    if chk_lib_missing_packages:
        logger.log(f"Missing dependencies: {chk_lib_missing_packages}", "FAIL")
        print("⣏!⣽ Missing dependencies:")
        for pkg in chk_lib_missing_packages:
            print(f" - {pkg}")

        chk_lib_install = input("Install missing packages? (Y/N) ").upper()
        if chk_lib_install == "Y":
            try:
                logger.log(f"Attempting pip install: {chk_lib_missing_packages}", "INFO")
                print(f"\nTrying <pip install --user {' '.join(chk_lib_missing_packages)}>\n\n")
                subprocess.check_call(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "--user",
                        *chk_lib_missing_packages,
                    ]
                )
                logger.log("Pip installation successful", "OK")
                print("\n\nPlease restart the EHS.")
                time.sleep(1)
                sys.exit(0)
            except Exception as e:
                logger.log(f"Pip installation failed: {str(e)}", "FAIL")
                print(f"\n⣏!⣽ <pip install --user {' '.join(chk_lib_missing_packages)}> failed:\n {str(e)}")
                print(f"\nTrying <upm add {' '.join(chk_lib_missing_packages)}>\n\n")
                try:
                    logger.log(f"Attempting upm install: {chk_lib_missing_packages}", "INFO")
                    subprocess.check_call(
                        [
                            "upm",
                            "add",
                            *chk_lib_missing_packages,
                        ]
                    )
                    logger.log("UPM installation successful", "OK")
                    print("\n\nPlease restart the EHS.")
                    time.sleep(1)
                    sys.exit(0)
                except Exception as e2:
                    logger.log(f"UPM installation failed: {str(e2)}", "FAIL")
                    print(f"\n⣏!⣽ upm add {' '.join(chk_lib_missing_packages)} failed:\n {str(e2)}")
                    return False
        else:
            logger.log("User declined to install missing packages", "WARN")
            print("\nYou can install the packages manually with:")
            print(f"pip install --user {' '.join(chk_lib_missing_packages)}")
            print("or")
            print(f"upm add {' '.join(chk_lib_missing_packages)}")
            time.sleep(1)
            return False
    
    logger.log("All required libraries are available", "OK")
    return True
