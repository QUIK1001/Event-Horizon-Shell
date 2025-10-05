import os
import zipfile
from .chklib_clrscr import clear_screen
from .logger import logger
from colorama import Fore, Style

def archiver():
    logger.log("Starting archiver utility", "INFO")
    
    original_dir = os.getcwd()
    
    while True:
        try:
            print(f"{Fore.CYAN}=== ZIP Archiver Utility ==={Style.RESET_ALL}")
            print(f"{Fore.BLUE}Current directory: {os.getcwd()}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}1. Create ZIP archive")
            print(f"2. Extract ZIP archive")
            print(f"3. List directory contents")
            print(f"4. Change directory")
            print(f"5. Show current directory")
            print(f"6. Back to main menu{Style.RESET_ALL}")
            
            choice = input(f"{Fore.GREEN}Select option (1-6): {Style.RESET_ALL}").strip()
            
            if choice == "1":
                create_zip_archive_interactive()
                input("Press Enter to continue...")
                clear_screen()
            elif choice == "2":
                extract_zip_archive_interactive()
                input("Press Enter to continue...")
                clear_screen()
            elif choice == "3":
                list_current_directory()
                input("Press Enter to continue...")
                clear_screen()
            elif choice == "4":
                change_directory()
                input("Press Enter to continue...")
                clear_screen()
            elif choice == "5":
                print(f"{Fore.BLUE}Current directory: {os.getcwd()}{Style.RESET_ALL}")
                input("Press Enter to continue...")
                clear_screen()
            elif choice == "6":
                logger.log("Exiting archiver utility", "INFO")
                os.chdir(original_dir)
                clear_screen()
                break
            else:
                print(f"{Fore.RED}Invalid choice! Please select 1-6.{Style.RESET_ALL}")
                input("Press Enter to continue...")
                clear_screen()
                
        except KeyboardInterrupt:
            print(f"{Fore.YELLOW}\nOperation cancelled.{Style.RESET_ALL}")
            os.chdir(original_dir)
            break
        except Exception as e:
            logger.log(f"Error in archiver utility: {str(e)}", "FAIL")
            print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
            input("Press Enter to continue...")
            clear_screen()

def get_archiver_prompt():
    from .constants import config
    
    current_dir = os.getcwd()
    home_dir = os.path.expanduser("~")
    
    if current_dir.startswith(home_dir):
        current_dir = current_dir.replace(home_dir, "~", 1)
    
    theme = config.THEMES[config.CURRENT_THEME]
    return theme["prompt"] + f"archiver#{current_dir}> " + Style.RESET_ALL

def change_directory():
    try:
        new_dir = input(f"{get_archiver_prompt()}cd> ").strip()
        if not new_dir:
            print(f"{Fore.RED}Path cannot be empty!{Style.RESET_ALL}")
            return
        
        expanded_path = os.path.expanduser(new_dir)
        
        if not os.path.exists(expanded_path):
            print(f"{Fore.RED}Directory not found: {expanded_path}{Style.RESET_ALL}")
            return
            
        if not os.path.isdir(expanded_path):
            print(f"{Fore.RED}Path is not a directory: {expanded_path}{Style.RESET_ALL}")
            return
            
        os.chdir(expanded_path)
        print(f"{Fore.GREEN}Changed to: {os.getcwd()}{Style.RESET_ALL}")
        logger.log(f"Directory changed to: {os.getcwd()}", "OK")
        
    except PermissionError as e:
        logger.log(f"Permission denied for directory: {new_dir}", "FAIL", e)
        print(f"{Fore.RED}Permission denied! Cannot access '{new_dir}'{Style.RESET_ALL}")
    except Exception as e:
        logger.log(f"Error changing directory: {new_dir}", "FAIL", e)
        print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")

def list_current_directory():
    try:
        current_dir = os.getcwd()
        print(f"{Fore.CYAN}Contents of '{current_dir}':{Style.RESET_ALL}")
        
        items = os.listdir(current_dir)
        dir_count = 0
        file_count = 0
        zip_files = []
        
        for item in items:
            item_path = os.path.join(current_dir, item)
            if os.path.isdir(item_path):
                print(f"{Fore.BLUE}[DIR]  {item}{Style.RESET_ALL}")
                dir_count += 1
            else:
                if item.lower().endswith('.zip'):
                    print(f"{Fore.GREEN}[ZIP]  {item}{Style.RESET_ALL}")
                    zip_files.append(item)
                else:
                    print(f"{Fore.WHITE}[FILE] {item}{Style.RESET_ALL}")
                file_count += 1
        
        print(f"\n{Fore.YELLOW}Summary: {dir_count} directories, {file_count} files{Style.RESET_ALL}")
        if zip_files:
            print(f"{Fore.GREEN}ZIP archives: {len(zip_files)}{Style.RESET_ALL}")
        
        logger.log(f"Directory listed: {dir_count} directories, {file_count} files", "OK")
        
    except PermissionError as e:
        logger.log(f"Permission denied for directory listing: {current_dir}", "FAIL", e)
        print(f"{Fore.RED}Permission denied for directory listing!{Style.RESET_ALL}")
    except Exception as e:
        logger.log(f"Error listing directory: {current_dir}", "FAIL", e)
        print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")

def create_zip_archive_interactive():
    try:
        print(f"\n{Fore.CYAN}=== Create ZIP Archive ==={Style.RESET_ALL}")
        print(f"{Fore.BLUE}Current directory: {os.getcwd()}{Style.RESET_ALL}")
        
        list_current_directory()
        print()
        
        zip_name = input(f"{get_archiver_prompt()}Enter archive name: ").strip()
        if not zip_name:
            print(f"{Fore.RED}Archive name cannot be empty!{Style.RESET_ALL}")
            return
            
        if not zip_name.endswith('.zip'):
            zip_name += '.zip'
        
        if os.path.exists(zip_name):
            overwrite = input(f"{Fore.YELLOW}Archive '{zip_name}' already exists. Overwrite? (Y/N): {Style.RESET_ALL}").upper()
            if overwrite != 'Y':
                print(f"{Fore.YELLOW}Operation cancelled.{Style.RESET_ALL}")
                return
        
        print(f"\n{Fore.YELLOW}Enter files to add (one per line):{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}- Use '*' for all files in current directory{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}- Use '*.ext' for all files with extension{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}- Use partial names with wildcards{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}- Empty line to finish{Style.RESET_ALL}")
        
        files_to_add = []
        while True:
            file_pattern = input(f"{get_archiver_prompt()}").strip()
            if not file_pattern:
                break
            
            if file_pattern == '*':
                all_files = [f for f in os.listdir('.') if os.path.isfile(f) and f != zip_name]
                files_to_add.extend(all_files)
                print(f"{Fore.GREEN}Added {len(all_files)} files{Style.RESET_ALL}")
                break
            else:
                files_to_add.append(file_pattern)
        
        if not files_to_add:
            print(f"{Fore.RED}No files specified!{Style.RESET_ALL}")
            return
        
        valid_files = []
        missing_files = []
        
        for pattern in files_to_add:
            if '*' in pattern or '?' in pattern:
                import glob
                matched_files = glob.glob(pattern)
                if matched_files:
                    valid_files.extend([f for f in matched_files if os.path.isfile(f) and f != zip_name])
                else:
                    missing_files.append(pattern)
            else:
                if os.path.exists(pattern) and os.path.isfile(pattern) and pattern != zip_name:
                    valid_files.append(pattern)
                else:
                    missing_files.append(pattern)
        
        valid_files = list(set(valid_files))
        
        if missing_files:
            print(f"{Fore.RED}Files not found:{Style.RESET_ALL}")
            for missing in missing_files:
                print(f"  - {missing}")
            
            if not valid_files:
                print(f"{Fore.RED}No valid files to archive!{Style.RESET_ALL}")
                return
                
            continue_anyway = input(f"{Fore.YELLOW}Continue with available files? (Y/N): {Style.RESET_ALL}").upper()
            if continue_anyway != 'Y':
                return
        
        if not valid_files:
            print(f"{Fore.RED}No valid files to archive!{Style.RESET_ALL}")
            return
        
        print(f"\n{Fore.CYAN}Files to be added to archive:{Style.RESET_ALL}")
        for file_path in valid_files:
            file_size = os.path.getsize(file_path)
            print(f"  - {file_path} ({file_size} bytes)")
        
        confirm = input(f"\n{Fore.YELLOW}Create archive with {len(valid_files)} files? (Y/N): {Style.RESET_ALL}").upper()
        if confirm != 'Y':
            print(f"{Fore.YELLOW}Operation cancelled.{Style.RESET_ALL}")
            return
        
        try:
            with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
                total_size = 0
                for file_path in valid_files:
                    arcname = os.path.basename(file_path)
                    zipf.write(file_path, arcname)
                    file_size = os.path.getsize(file_path)
                    total_size += file_size
                    print(f"{Fore.GREEN}Added: {file_path} ({file_size} bytes){Style.RESET_ALL}")

            archive_size = os.path.getsize(zip_name)
            compression_ratio = (1 - archive_size / total_size) * 100 if total_size > 0 else 0
            
            print(f"\n{Fore.CYAN}Archive created successfully!{Style.RESET_ALL}")
            print(f"{Fore.BLUE}Archive: {zip_name}{Style.RESET_ALL}")
            print(f"{Fore.BLUE}Total files: {len(valid_files)}{Style.RESET_ALL}")
            print(f"{Fore.BLUE}Original size: {total_size} bytes ({total_size/1024:.2f} KB){Style.RESET_ALL}")
            print(f"{Fore.BLUE}Archive size: {archive_size} bytes ({archive_size/1024:.2f} KB){Style.RESET_ALL}")
            print(f"{Fore.BLUE}Compression: {compression_ratio:.1f}%{Style.RESET_ALL}")
            
            logger.log(f"ZIP archive created: {zip_name} with {len(valid_files)} files", "OK")
            
        except Exception as e:
            logger.log(f"Error creating ZIP archive: {str(e)}", "FAIL")
            print(f"{Fore.RED}Error creating archive: {str(e)}{Style.RESET_ALL}")
            if os.path.exists(zip_name):
                os.remove(zip_name)
    
    except KeyboardInterrupt:
        print(f"{Fore.YELLOW}\nOperation cancelled.{Style.RESET_ALL}")
    except Exception as e:
        logger.log(f"Error in create_zip_archive_interactive: {str(e)}", "FAIL")
        print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")

def extract_zip_archive_interactive():
    try:
        print(f"\n{Fore.CYAN}=== Extract ZIP Archive ==={Style.RESET_ALL}")
        print(f"{Fore.BLUE}Current directory: {os.getcwd()}{Style.RESET_ALL}")

        zip_files = [f for f in os.listdir('.') if f.lower().endswith('.zip')]
        if zip_files:
            print(f"{Fore.YELLOW}Available ZIP files:{Style.RESET_ALL}")
            for i, zip_file in enumerate(zip_files, 1):
                file_size = os.path.getsize(zip_file)
                print(f"  {i}. {zip_file} ({file_size} bytes)")
            print()

        zip_path = input(f"{get_archiver_prompt()}Enter ZIP file name: ").strip()
        if not zip_path:
            print(f"{Fore.RED}ZIP file name cannot be empty!{Style.RESET_ALL}")
            return
        
        if not zip_path.endswith('.zip'):
            zip_path += '.zip'
        
        if not os.path.exists(zip_path):
            print(f"{Fore.RED}Archive not found: {zip_path}{Style.RESET_ALL}")
            return
        
        if not zipfile.is_zipfile(zip_path):
            print(f"{Fore.RED}Not a valid ZIP file: {zip_path}{Style.RESET_ALL}")
            return

        extract_path = input(f"{get_archiver_prompt()}Enter extraction path (Enter for current): ").strip()
        if not extract_path:
            extract_path = "."
        
        extract_path = os.path.expanduser(extract_path)

        if not os.path.exists(extract_path):
            create_dir = input(f"{Fore.YELLOW}Directory doesn't exist. Create it? (Y/N): {Style.RESET_ALL}").upper()
            if create_dir == 'Y':
                os.makedirs(extract_path, exist_ok=True)
                print(f"{Fore.GREEN}Directory created: {extract_path}{Style.RESET_ALL}")
            else:
                return

        try:
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                file_list = zipf.namelist()
                total_size = sum(zipf.getinfo(f).file_size for f in file_list)
                compressed_size = sum(zipf.getinfo(f).compress_size for f in file_list)
                
                print(f"\n{Fore.CYAN}Archive contents ({len(file_list)} files):{Style.RESET_ALL}")
                for file_in_zip in file_list:
                    info = zipf.getinfo(file_in_zip)
                    size_str = f"{info.file_size} bytes" if info.file_size < 1024 else f"{info.file_size/1024:.1f} KB"
                    print(f"  - {file_in_zip} ({size_str})")
                
                print(f"\n{Fore.BLUE}Total size: {total_size} bytes ({total_size/1024:.2f} KB){Style.RESET_ALL}")
                print(f"{Fore.BLUE}Compressed: {compressed_size} bytes ({compressed_size/1024:.2f} KB){Style.RESET_ALL}")

                existing_files = []
                for file_in_zip in file_list:
                    extract_file_path = os.path.join(extract_path, file_in_zip)
                    if os.path.exists(extract_file_path):
                        existing_files.append(file_in_zip)
                
                if existing_files:
                    print(f"\n{Fore.YELLOW}Warning: {len(existing_files)} files already exist:{Style.RESET_ALL}")
                    for existing in existing_files[:5]:
                        print(f"  - {existing}")
                    if len(existing_files) > 5:
                        print(f"  ... and {len(existing_files) - 5} more")
                    
                    overwrite = input(f"{Fore.YELLOW}Overwrite existing files? (Y/N/A=all/S=skip existing): {Style.RESET_ALL}").upper()
                    if overwrite == 'N':
                        return
                    elif overwrite == 'S':
                        file_list = [f for f in file_list if f not in existing_files]
                        if not file_list:
                            print(f"{Fore.YELLOW}No files to extract (all already exist).{Style.RESET_ALL}")
                            return
     
                confirm = input(f"\n{Fore.YELLOW}Extract {len(file_list)} files? (Y/N): {Style.RESET_ALL}").upper()
                if confirm != 'Y':
                    print(f"{Fore.YELLOW}Operation cancelled.{Style.RESET_ALL}")
                    return

                print(f"\n{Fore.CYAN}Extracting...{Style.RESET_ALL}")
                extracted_count = 0
                for file_in_zip in file_list:
                    try:
                        zipf.extract(file_in_zip, extract_path)
                        extracted_count += 1
                        print(f"{Fore.GREEN}Extracted: {file_in_zip}{Style.RESET_ALL}")
                    except Exception as e:
                        print(f"{Fore.RED}Error extracting {file_in_zip}: {str(e)}{Style.RESET_ALL}")
                
                print(f"\n{Fore.CYAN}Extraction completed!{Style.RESET_ALL}")
                print(f"{Fore.BLUE}Extracted to: {extract_path}{Style.RESET_ALL}")
                print(f"{Fore.BLUE}Files extracted: {extracted_count}/{len(file_list)}{Style.RESET_ALL}")
                
                logger.log(f"ZIP archive extracted: {zip_path} to {extract_path} ({extracted_count} files)", "OK")
                
        except Exception as e:
            logger.log(f"Error extracting ZIP archive: {str(e)}", "FAIL")
            print(f"{Fore.RED}Error extracting archive: {str(e)}{Style.RESET_ALL}")
    
    except KeyboardInterrupt:
        print(f"{Fore.YELLOW}\nOperation cancelled.{Style.RESET_ALL}")
    except Exception as e:
        logger.log(f"Error in extract_zip_archive_interactive: {str(e)}", "FAIL")
        print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")

def create_zip_archive(zip_name, files):
    try:
        with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in files:
                expanded_path = os.path.expanduser(file_path)
                if os.path.exists(expanded_path):
                    arcname = os.path.basename(expanded_path)
                    zipf.write(expanded_path, arcname)
        return True
    except Exception as e:
        logger.log(f"Error creating ZIP: {str(e)}", "FAIL")
        return False

def extract_zip_archive(zip_name, extract_path):
    try:
        zip_path = os.path.expanduser(zip_name)
        extract_path = os.path.expanduser(extract_path)
        
        if not os.path.exists(extract_path):
            os.makedirs(extract_path, exist_ok=True)
            
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            zipf.extractall(extract_path)
        return True
    except Exception as e:
        logger.log(f"Error extracting ZIP: {str(e)}", "FAIL")
        return False
