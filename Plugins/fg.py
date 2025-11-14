import argparse
import os
import hashlib
from pathlib import Path
import shutil
import json
from datetime import datetime
import sys
import mimetypes
from collections import defaultdict
import fnmatch
import zipfile
import tarfile
import tempfile
import csv
import logging
from typing import Dict, List, Optional, Tuple
import io
import webbrowser
from tkinter import filedialog, messagebox, ttk
import tkinter as tk
from tkinter import scrolledtext
import threading
import subprocess
from PIL import Image, ImageTk
import platform

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

class EnhancedFileOrganizer:
    def __init__(self):
        self.extensions = self.get_ext_cfg()
        self.stats = defaultdict(lambda: {'count': 0, 'size': 0})
        self.log_buffer = []

    def get_ext_cfg(self):
        return {
            'images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg', '.ico', '.raw', '.heic', '.psd', '.ai', '.eps', '.indd', '.sketch', '.cr2', '.nef', '.arw', '.dng', '.ico', '.icns', '.tga', '.psb'],
            'documents': ['.pdf', '.doc', '.docx', '.txt', '.md', '.rtf', '.odt', '.xls', '.xlsx', '.ppt', '.pptx', '.csv', '.tsv', '.ods', '.odp', '.pages', '.numbers', '.key', '.tex', '.latex', '.wpd', '.wps', '.oxps', '.xps'],
            'code': ['.py', '.js', '.java', '.cpp', '.c', '.h', '.hpp', '.html', '.css', '.php', '.rb', '.go', '.rs', '.ts', '.json', '.xml', '.yml', '.yaml', '.ini', '.cfg', '.conf', '.config', '.sql', '.sh', '.bat', '.ps1', '.swift', '.kt', '.dart', '.lua', '.pl', '.r', '.m', '.asm', '.s', '.vue', '.jsx', '.tsx', '.scss', '.less', '.sass', '.coffee'],
            'archives': ['.zip', '.rar', '.tar', '.gz', '.7z', '.bz2', '.xz', '.iso', '.dmg', '.pkg', '.deb', '.rpm', '.msi', '.apk', '.jar', '.war', '.egg', '.whl', '.cpio', '.z', '.lz', '.lzma', '.lzo'],
            'audio': ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma', '.aiff', '.ape', '.opus', '.mid', '.midi', '.amr', '.3ga', '.ac3', '.dts', '.ra', '.rm', '.voc', '.8svx'],
            'video': ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.mpg', '.mpeg', '.3gp', '.m2ts', '.ts', '.mts', '.vob', '.ogv', '.divx', '.xvid', '.asf', '.rmvb', '.avchd', '.hevc', '.vp9'],
            'executables': ['.exe', '.msi', '.dmg', '.pkg', '.deb', '.rpm', '.apk', '.app', '.bat', '.sh', '.cmd', '.com', '.scr', '.msc', '.jar', '.bin', '.run', '.out', '.appimage'],
            'fonts': ['.ttf', '.otf', '.woff', '.woff2', '.eot', '.pfb', '.pfm', '.afm', '.pfa', '.bdf', '.fnt', '.fon', '.ttc', '.dfont'],
            'databases': ['.db', '.sqlite', '.mdb', '.accdb', '.sql', '.dbf', '.mdf', '.ndf', '.ldf', '.frm', '.myd', '.myi', '.ibd', '.wal'],
            'backups': ['.bak', '.backup', '.old', '.tmp', '.temp', '.crdownload', '.part', '.partial', '.save', '.sav', '.autosave'],
            'ebooks': ['.epub', '.mobi', '.azw', '.azw3', '.fb2', '.lit', '.lrf', '.pdb', '.pml', '.rb', '.snb', '.tcr'],
            'cad': ['.dwg', '.dxf', '.stl', '.obj', '.blend', '.max', '.3ds', '.fbx', '.dae', '.iges', '.step', '.stp', '.iges', '.igs', '.x_t', '.x_b', '.sat', '.sldprt', '.sldasm', '.prt', '.asm', '.ipt', '.iam'],
            'virtual_machines': ['.vmdk', '.ova', '.ovf', '.vdi', '.vhd', '.vhdx', '.qcow2', '.vmem', '.nvram', '.vmx', '.vmxf', '.vmsd', '.vmtm', '.vmss'],
            'configs': ['.ini', '.cfg', '.conf', '.config', '.properties', '.prop', '.settings', '.prefs', '.plist', '.reg', '.inf', '.desktop'],
            'logs': ['.log', '.txtlog', '.error', '.debug', '.trace', '.audit', '.event', '.history', '.cache', '.cached', '.journal'],
            'torrents': ['.torrent'],
            'subtitles': ['.srt', '.sub', '.vtt', '.ass', '.ssa', '.smi', '.sbv', '.mpl'],
            'presentations': ['.ppt', '.pptx', '.key', '.odp', '.pps', '.ppsx', '.sxi'],
            'spreadsheets': ['.xls', '.xlsx', '.ods', '.numbers', '.csv', '.tsv', '.dif'],
            'emails': ['.eml', '.msg', '.pst', '.ost', '.mbox', '.mbx', '.emlx'],
            'gis': ['.shp', '.kml', '.kmz', '.gpx', '.geojson', '.topojson', '.mif', '.tab'],
            'scientific': ['.fits', '.root', '.h5', '.hdf5', '.nc', '.cdf', '.mat', '.sav'],
            'game_files': ['.pak', '.pak2', '.wad', '.bsp', '.map', '.mdl', '.vmt', '.vtf', '.unitypackage', '.uasset', '.umap', '.blend', '.ma', '.mb'],
            'source_control': ['.git', '.svn', '.hg', '.cvs', '.bzr', '.gitignore', '.gitattributes'],
            'docker': ['.dockerfile', '.dockerignore'],
            'certificates': ['.pem', '.crt', '.cer', '.key', '.pfx', '.p12', '.der', '.csr'],
            'scripts': ['.sh', '.bash', '.zsh', '.fish', '.ps1', '.psm1', '.psd1', '.vbs', '.ahk', '.au3', '.scpt', '.applescript', '.jsx', '.tsx'],
            'web_assets': ['.html', '.htm', '.css', '.scss', '.less', '.sass', '.js', '.jsx', '.ts', '.tsx', '.vue', '.php', '.asp', '.aspx', '.jsp', '.mustache', '.hbs', '.ejs', '.pug', '.twig']
        }

    def _smart_categorize(self, file_path: Path) -> str:
        ext = file_path.suffix.lower()
        for category, exts in self.extensions.items():
            if ext in exts:
                return category
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type:
            mime_major = mime_type.split('/')[0]
            mime_map = {
                'image': 'images',
                'text': 'documents',
                'application': 'code',
                'audio': 'audio',
                'video': 'video'
            }
            return mime_map.get(mime_major, 'other')
        return 'unknown'

    def log_message(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_buffer.append(f"[{timestamp}] {message}")
        if len(self.log_buffer) > 1000:
            self.log_buffer.pop(0)

    def get_logs(self) -> List[str]:
        return self.log_buffer.copy()

    def org_files(self, directory, organize_by_date=False, date_format="%Y/%m", dry_run=False, copy=False, backup_dir=None, progress_callback=None, log_callback=None):
        directory = Path(directory)
        self.stats.clear()
        current_script = Path(__file__).resolve()
        files_to_process = [f for f in directory.iterdir() if f.is_file() and f.resolve() != current_script]
        total_files = len(files_to_process)
        processed = 0

        for file_path in files_to_process:
            category = self._smart_categorize(file_path)
            if organize_by_date:
                mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                date_folder = mtime.strftime(date_format)
                category_dir = directory / category / date_folder
            else:
                category_dir = directory / category

            category_dir.mkdir(parents=True, exist_ok=True)
            dest_path = category_dir / file_path.name
            counter = 1
            original_dest = dest_path
            while dest_path.exists():
                new_name = f"{file_path.stem}_{counter}{file_path.suffix}"
                dest_path = category_dir / new_name
                counter += 1

            if not dry_run:
                file_size = file_path.stat().st_size
                if copy:
                    shutil.copy2(str(file_path), dest_path)
                    if log_callback:
                        log_callback(f"СКОПИРОВАНО: {file_path.name} -> {dest_path}")
                else:
                    if backup_dir:
                        backup_dir_path = Path(backup_dir)
                        backup_dir_path.mkdir(parents=True, exist_ok=True)
                        backup_path = backup_dir_path / file_path.name
                        counter_b = 1
                        while backup_path.exists():
                            backup_name = f"{file_path.stem}_backup_{counter_b}{file_path.suffix}"
                            backup_path = backup_dir_path / backup_name
                            counter_b += 1
                        shutil.copy2(str(file_path), backup_path)
                        if log_callback:
                            log_callback(f"РЕЗЕРВНАЯ КОПИЯ: {file_path.name} -> {backup_path}")

                    shutil.move(str(file_path), dest_path)
                    if log_callback:
                        log_callback(f"ПЕРЕМЕЩЕНО: {file_path.name} -> {dest_path}")

                if dest_path != original_dest:
                    if log_callback:
                        log_callback(f"ПЕРЕИМЕНОВАНО: {file_path.name} -> {dest_path.name}")

                self.stats[category]['count'] += 1
                self.stats[category]['size'] += file_size
            else:
                self.stats[category]['count'] += 1
                self.stats[category]['size'] += file_path.stat().st_size
                if log_callback:
                    log_callback(f"ПРОБНЫЙ РЕЖИМ: {file_path.name} -> {dest_path}")

            processed += 1
            if progress_callback:
                progress_callback(processed, total_files)

    def find_dup(self, directory, delete=False, min_size=0, algorithm='md5', interactive=False, progress_callback=None, log_callback=None):
        directory = Path(directory)
        hash_func = {
            'md5': hashlib.md5,
            'sha1': hashlib.sha1,
            'sha256': hashlib.sha256
        }.get(algorithm, hashlib.md5)

        size_groups = defaultdict(list)
        total_files = 0
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                total_files += 1
        processed = 0

        for file_path in directory.rglob('*'):
            if file_path.is_file():
                try:
                    size = file_path.stat().st_size
                    if size >= min_size:
                        size_groups[size].append(file_path)
                except OSError:
                    pass
            processed += 1
            if progress_callback:
                progress_callback(processed, total_files)

        duplicates = []
        for size, files in size_groups.items():
            if len(files) > 1:
                hashes = defaultdict(list)
                for file_path in files:
                    try:
                        file_hash = self._calculate_hash(file_path, hash_func)
                        file_info = {
                            'path': file_path,
                            'size': size,
                            'hash': file_hash,
                            'mtime': datetime.fromtimestamp(file_path.stat().st_mtime)
                        }
                        hashes[file_hash].append(file_info)
                    except (IOError, OSError):
                        pass

                for file_hash, file_list in hashes.items():
                    if len(file_list) > 1:
                        file_list.sort(key=lambda x: x['mtime'])
                        original = file_list[0]
                        dup_copies = file_list[1:]
                        for dup in dup_copies:
                            if interactive:
                                if log_callback:
                                    log_callback(f"НАЙДЕН ДУБЛИКАТ: {dup['path']} (оригинал: {original['path']})")
                                duplicates.append((dup['path'], original['path']))
                            else:
                                if log_callback:
                                    log_callback(f"НАЙДЕН ДУБЛИКАТ: {dup['path']} (оригинал: {original['path']})")
                                duplicates.append((dup['path'], original['path']))

        if duplicates and delete:
            deleted_count = self._delete_duplicates([d[0] for d in duplicates], log_callback)
            if log_callback:
                log_callback(f"УДАЛЕНО ДУБЛИКАТОВ: {deleted_count}")
            return deleted_count

        return len(duplicates)

    def _calculate_hash(self, file_path, hash_func, chunk_size=8192):
        hash_obj = hash_func()
        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()

    def _delete_duplicates(self, duplicates, log_callback=None):
        deleted_count = 0
        for dup in duplicates:
            try:
                dup.unlink()
                deleted_count += 1
                if log_callback:
                    log_callback(f"УДАЛЕН: {dup}")
            except OSError as e:
                if log_callback:
                    log_callback(f"ОШИБКА УДАЛЕНИЯ {dup}: {e}")
        return deleted_count

    def clean_empty_dirs(self, directory, recursive=True, progress_callback=None, log_callback=None):
        directory = Path(directory)
        empty_dirs = []
        if recursive:
            walk_iter = list(directory.rglob('*'))
        else:
            walk_iter = list(directory.iterdir())

        total_dirs = len([d for d in walk_iter if d.is_dir()])
        processed = 0

        for dir_path in walk_iter:
            if dir_path.is_dir() and not any(dir_path.iterdir()):
                empty_dirs.append(dir_path)

        cleaned_count = 0
        for empty_dir in empty_dirs:
            try:
                empty_dir.rmdir()
                cleaned_count += 1
                if log_callback:
                    log_callback(f"УДАЛЕНА ПУСТАЯ ПАПКА: {empty_dir}")
            except OSError as e:
                if log_callback:
                    log_callback(f"ОШИБКА УДАЛЕНИЯ ПАПКИ {empty_dir}: {e}")
            processed += 1
            if progress_callback:
                progress_callback(processed, total_dirs)
        
        return cleaned_count

    def get_file_stats(self, directory, detailed=False, export_csv=None, export_html=None, progress_callback=None):
        directory = Path(directory)
        self.stats.clear()
        all_files = []
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                all_files.append(file_path)
        total_files = len(all_files)
        processed = 0

        for file_path in all_files:
            category = self._smart_categorize(file_path)
            self.stats[category]['count'] += 1
            self.stats[category]['size'] += file_path.stat().st_size
            processed += 1
            if progress_callback:
                progress_callback(processed, total_files)

        if export_csv:
            with open(export_csv, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['File', 'Size_Bytes', 'Size_MB', 'Category', 'Extension'])
                for file_path in all_files:
                    size = file_path.stat().st_size
                    category = self._smart_categorize(file_path)
                    writer.writerow([
                        str(file_path),
                        size,
                        size / (1024 * 1024),
                        category,
                        file_path.suffix.lower()
                    ])

        if export_html:
            html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Отчёт Filer</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #4CAF50; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        .summary { background-color: #e7f3ff; padding: 15px; border-radius: 5px; margin: 20px 0; }
    </style>
</head>
<body>
    <h1>📊 Отчёт о файлах</h1>
    <div class="summary">
        <h3>Сводка</h3>
"""
            for category, data in self.stats.items():
                html_content += f"        <p><strong>{category}:</strong> {data['count']} файлов, {self._format_size(data['size'])}</p>\n"
            
            html_content += """
    </div>
    <table>
        <tr><th>Файл</th><th>Размер (Байт)</th><th>Размер (МБ)</th><th>Категория</th><th>Расширение</th></tr>
"""
            for file_path in all_files:
                size = file_path.stat().st_size
                category = self._smart_categorize(file_path)
                html_content += f"""
        <tr>
            <td>{file_path}</td>
            <td>{size}</td>
            <td>{size / (1024 * 1024):.2f}</td>
            <td>{category}</td>
            <td>{file_path.suffix.lower()}</td>
        </tr>
"""
            html_content += """
    </table>
</body>
</html>
"""
            with open(export_html, 'w', encoding='utf-8') as f:
                f.write(html_content)

    def find_files(self, directory, patterns, case_sensitive=False, recursive=True, progress_callback=None, log_callback=None):
        directory = Path(directory)
        matches = []
        if recursive:
            walk_iter = list(directory.rglob('*'))
        else:
            walk_iter = list(directory.iterdir())
        total_files = len(walk_iter)
        processed = 0

        for file_path in walk_iter:
            if file_path.is_file():
                filename = file_path.name if case_sensitive else file_path.name.lower()
                search_patterns = patterns if case_sensitive else [p.lower() for p in patterns]
                for pattern in search_patterns:
                    if fnmatch.fnmatch(filename, pattern):
                        matches.append(file_path)
                        if log_callback:
                            log_callback(f"НАЙДЕН: {file_path}")
                        break
            processed += 1
            if progress_callback:
                progress_callback(processed, total_files)
        return matches

    def rename_files(self, directory, pattern, replacement, dry_run=False, progress_callback=None, log_callback=None):
        directory = Path(directory)
        files = list(directory.iterdir())
        total_files = len(files)
        processed = 0
        renamed_count = 0

        for file_path in files:
            if file_path.is_file():
                new_name = file_path.name.replace(pattern, replacement)
                if new_name != file_path.name:
                    new_path = file_path.parent / new_name
                    if not dry_run:
                        try:
                            file_path.rename(new_path)
                            renamed_count += 1
                            if log_callback:
                                log_callback(f"ПЕРЕИМЕНОВАН: {file_path.name} -> {new_name}")
                        except OSError as e:
                            if log_callback:
                                log_callback(f"ОШИБКА ПЕРЕИМЕНОВАНИЯ {file_path.name}: {e}")
                    else:
                        renamed_count += 1
                        if log_callback:
                            log_callback(f"ПРОБНЫЙ РЕЖИМ: {file_path.name} -> {new_name}")
            processed += 1
            if progress_callback:
                progress_callback(processed, total_files)
        return renamed_count

    def extract_archives(self, directory, delete_after=False, archive_types=None, progress_callback=None, log_callback=None):
        directory = Path(directory)
        if archive_types is None:
            archive_types = ['.zip', '.rar', '.tar', '.gz', '.7z']
        
        archives = [f for f in directory.iterdir() if f.is_file() and f.suffix.lower() in archive_types]
        total_archives = len(archives)
        processed = 0
        extracted_count = 0

        for file_path in archives:
            extract_dir = file_path.parent / file_path.stem
            extract_dir.mkdir(exist_ok=True)
            try:
                if file_path.suffix.lower() == '.zip':
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_dir)
                elif file_path.suffix.lower() in ['.tar', '.gz']:
                    with tarfile.open(file_path, 'r:*') as tar_ref:
                        tar_ref.extractall(extract_dir)
                extracted_count += 1
                if log_callback:
                    log_callback(f"ИЗВЛЕЧЕН: {file_path} -> {extract_dir}")
                
                if delete_after:
                    file_path.unlink()
                    if log_callback:
                        log_callback(f"УДАЛЕН АРХИВ: {file_path}")
            except (zipfile.BadZipFile, tarfile.ReadError) as e:
                if log_callback:
                    log_callback(f"ОШИБКА ИЗВЛЕЧЕНИЯ {file_path}: {e}")
            processed += 1
            if progress_callback:
                progress_callback(processed, total_archives)
        return extracted_count

    def _format_size(self, size_bytes):
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"

    def save_cfg(self, config_path):
        config = self.extensions
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    def load_cfg(self, config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return None

    def bulk_delete(self, directory, patterns, case_sensitive=False, recursive=True, min_size=0, max_size=float('inf'), progress_callback=None, log_callback=None):
        directory = Path(directory)
        files_to_delete = []
        
        if recursive:
            walk_iter = list(directory.rglob('*'))
        else:
            walk_iter = list(directory.iterdir())
        
        total_files = len(walk_iter)
        processed = 0

        for file_path in walk_iter:
            if file_path.is_file():
                file_size = file_path.stat().st_size
                filename = file_path.name if case_sensitive else file_path.name.lower()
                search_patterns = patterns if case_sensitive else [p.lower() for p in patterns]
                
                matches_pattern = any(fnmatch.fnmatch(filename, pattern) for pattern in search_patterns)
                matches_size = min_size <= file_size <= max_size
                
                if matches_pattern and matches_size:
                    files_to_delete.append(file_path)
            
            processed += 1
            if progress_callback:
                progress_callback(processed, total_files)

        deleted_count = 0
        for file_path in files_to_delete:
            try:
                file_path.unlink()
                deleted_count += 1
                if log_callback:
                    log_callback(f"УДАЛЕН: {file_path}")
            except OSError as e:
                if log_callback:
                    log_callback(f"ОШИБКА УДАЛЕНИЯ {file_path}: {e}")

        return deleted_count


class ModernFileOrganizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Filer v1.0-gui")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        
        self.bg_color = "#2b2b2b"
        self.fg_color = "#ffffff"
        self.accent_color = "#3498db"
        self.success_color = "#27ae60"
        self.warning_color = "#f39c12"
        self.danger_color = "#e74c3c"
        
        self.root.configure(bg=self.bg_color)
        
        try:
            self.root.iconbitmap("icon.ico")
        except:
            pass
        
        self.organizer = EnhancedFileOrganizer()
        self.setup_styles()
        self.setup_ui()

    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use("clam")
        
        self.style.configure(".", 
                           background=self.bg_color,
                           foreground=self.fg_color,
                           fieldbackground=self.bg_color)
        
        self.style.configure("TFrame", background=self.bg_color)
        self.style.configure("TLabel", background=self.bg_color, foreground=self.fg_color, font=("Segoe UI", 10))
        self.style.configure("TButton", font=("Segoe UI", 10), padding=8)
        self.style.configure("TCheckbutton", background=self.bg_color, foreground=self.fg_color, font=("Segoe UI", 10))
        self.style.configure("TRadiobutton", background=self.bg_color, foreground=self.fg_color, font=("Segoe UI", 10))
        self.style.configure("TEntry", font=("Segoe UI", 10), fieldbackground="#3c3c3c", foreground=self.fg_color)
        self.style.configure("TProgressbar", thickness=20, background=self.accent_color)
        self.style.configure("TNotebook", background=self.bg_color)
        self.style.configure("TNotebook.Tab", background="#3c3c3c", foreground=self.fg_color, padding=[10, 5])
        self.style.map("TNotebook.Tab", background=[("selected", self.accent_color)])
        self.style.configure("Custom.Treeview", 
                        background="#3c3c3c",
                        foreground="#ffffff",
                        fieldbackground="#3c3c3c",
                        rowheight=25)
        self.style.configure("Custom.Treeview.Heading",
                        background="#2b2b2b",
                        foreground="#ffffff",
                        relief="flat",
                        font=("Segoe UI", 10, "bold"))
        self.style.map("Custom.Treeview",
                  background=[('selected', '#3498db')],
                  foreground=[('selected', 'white')])
        self.style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"), foreground=self.accent_color)
        self.style.configure("Subtitle.TLabel", font=("Segoe UI", 12, "bold"), foreground=self.warning_color)
        
        self.style.configure("Accent.TButton", background=self.accent_color, foreground="white")
        self.style.map("Accent.TButton", background=[("active", "#2980b9")])
        
        self.style.configure("Success.TButton", background=self.success_color, foreground="white")
        self.style.map("Success.TButton", background=[("active", "#229954")])
        
        self.style.configure("Warning.TButton", background=self.warning_color, foreground="white")
        self.style.map("Warning.TButton", background=[("active", "#e67e22")])
        
        self.style.configure("Danger.TButton", background=self.danger_color, foreground="white")
        self.style.map("Danger.TButton", background=[("active", "#c0392b")])

    def setup_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.setup_org_tab()
        self.setup_dup_tab()
        self.setup_stat_tab()
        self.setup_find_tab()
        self.setup_rename_tab()
        self.setup_extract_tab()
        self.setup_clean_tab()
        self.setup_delete_tab()
        self.setup_file_manager_tab()
        self.setup_logs_tab()
        self.setup_about_tab()

        self.setup_status_bar()

    def create_console(self, parent, height=8):
        """Создает мини-консоль для вкладки"""
        console_frame = ttk.LabelFrame(parent, text="Консоль операций")
        console_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        console = scrolledtext.ScrolledText(console_frame, height=height, wrap=tk.WORD, 
                                          bg="#1e1e1e", fg="#00ff00", 
                                          font=("Consolas", 9), state=tk.DISABLED)
        console.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        return console

    def log_to_console(self, console, message):
        """Добавляет сообщение в мини-консоль"""
        console.config(state=tk.NORMAL)
        console.insert(tk.END, f"{message}\n")
        console.see(tk.END)
        console.config(state=tk.DISABLED)

    def clear_console(self, console):
        """Очищает мини-консоль"""
        console.config(state=tk.NORMAL)
        console.delete(1.0, tk.END)
        console.config(state=tk.DISABLED)

    def setup_status_bar(self):
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_var = tk.StringVar(value="Готов")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_label.pack(fill=tk.X, padx=5, pady=2)

    def create_scrolled_frame(self, parent):
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        canvas = tk.Canvas(main_frame, bg=self.bg_color, highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
        
        content_frame = ttk.Frame(canvas)
        
        content_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=content_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        return content_frame

    def setup_org_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="📁 Организация")

        main_container = ttk.Frame(frame)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        settings_frame = ttk.Frame(main_container)
        settings_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(settings_frame, text="Организация файлов", style="Title.TLabel").pack(anchor=tk.W, pady=(0, 20))

        dir_frame = ttk.Frame(settings_frame)
        dir_frame.pack(fill=tk.X, pady=5)
        ttk.Label(dir_frame, text="Целевая директория:").pack(side=tk.LEFT, padx=5)
        self.org_dir_var = tk.StringVar(value=str(Path.cwd()))
        ttk.Entry(dir_frame, textvariable=self.org_dir_var, width=70).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(dir_frame, text="Обзор", command=lambda: self.browse_dir(self.org_dir_var), style="Accent.TButton").pack(side=tk.LEFT, padx=5)

        options_frame = ttk.LabelFrame(settings_frame, text="Параметры организации")
        options_frame.pack(fill=tk.X, pady=10)

        options_row1 = ttk.Frame(options_frame)
        options_row1.pack(fill=tk.X, pady=5)
        self.org_by_date_var = tk.BooleanVar()
        ttk.Checkbutton(options_row1, text="Организовать по дате", variable=self.org_by_date_var).pack(side=tk.LEFT, padx=5)
        self.org_dry_run_var = tk.BooleanVar()
        ttk.Checkbutton(options_row1, text="Пробный запуск", variable=self.org_dry_run_var).pack(side=tk.LEFT, padx=5)
        self.org_copy_var = tk.BooleanVar()
        ttk.Checkbutton(options_row1, text="Копировать вместо перемещения", variable=self.org_copy_var).pack(side=tk.LEFT, padx=5)

        backup_frame = ttk.Frame(settings_frame)
        backup_frame.pack(fill=tk.X, pady=5)
        ttk.Label(backup_frame, text="Резервная директория (опционально):").pack(side=tk.LEFT, padx=5)
        self.org_backup_dir_var = tk.StringVar()
        ttk.Entry(backup_frame, textvariable=self.org_backup_dir_var, width=70).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(backup_frame, text="Обзор", command=lambda: self.browse_dir(self.org_backup_dir_var), style="Accent.TButton").pack(side=tk.LEFT, padx=5)

        action_frame = ttk.Frame(settings_frame)
        action_frame.pack(fill=tk.X, pady=10)
        ttk.Button(action_frame, text="🔄 Запустить организацию", command=self.run_org, style="Success.TButton").pack(side=tk.LEFT, padx=5)
        self.org_progress = ttk.Progressbar(action_frame, mode='determinate')
        self.org_progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        console_frame = ttk.Frame(main_container)
        console_frame.pack(fill=tk.BOTH, expand=True)
        self.org_console = self.create_console(console_frame)

    def setup_dup_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="🔍 Дубликаты")

        main_container = ttk.Frame(frame)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        settings_frame = ttk.Frame(main_container)
        settings_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(settings_frame, text="Поиск дубликатов", style="Title.TLabel").pack(anchor=tk.W, pady=(0, 20))

        dir_frame = ttk.Frame(settings_frame)
        dir_frame.pack(fill=tk.X, pady=5)
        ttk.Label(dir_frame, text="Директория:").pack(side=tk.LEFT, padx=5)
        self.dup_dir_var = tk.StringVar(value=str(Path.cwd()))
        ttk.Entry(dir_frame, textvariable=self.dup_dir_var, width=70).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(dir_frame, text="Обзор", command=lambda: self.browse_dir(self.dup_dir_var), style="Accent.TButton").pack(side=tk.LEFT, padx=5)

        options_frame = ttk.LabelFrame(settings_frame, text="Параметры поиска")
        options_frame.pack(fill=tk.X, pady=10)

        options_row1 = ttk.Frame(options_frame)
        options_row1.pack(fill=tk.X, pady=5)
        self.dup_del_var = tk.BooleanVar()
        ttk.Checkbutton(options_row1, text="Удалять дубликаты", variable=self.dup_del_var).pack(side=tk.LEFT, padx=5)
        self.dup_interactive_var = tk.BooleanVar()
        ttk.Checkbutton(options_row1, text="Интерактивный режим", variable=self.dup_interactive_var).pack(side=tk.LEFT, padx=5)

        options_row2 = ttk.Frame(options_frame)
        options_row2.pack(fill=tk.X, pady=5)
        ttk.Label(options_row2, text="Мин. размер (байт):").pack(side=tk.LEFT, padx=5)
        self.dup_min_size_var = tk.IntVar(value=0)
        ttk.Entry(options_row2, textvariable=self.dup_min_size_var, width=15).pack(side=tk.LEFT, padx=5)

        algo_frame = ttk.Frame(options_frame)
        algo_frame.pack(fill=tk.X, pady=5)
        ttk.Label(algo_frame, text="Алгоритм хеширования:").pack(side=tk.LEFT, padx=5)
        self.dup_algo_var = tk.StringVar(value="md5")
        ttk.Radiobutton(algo_frame, text="MD5", variable=self.dup_algo_var, value="md5").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(algo_frame, text="SHA1", variable=self.dup_algo_var, value="sha1").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(algo_frame, text="SHA256", variable=self.dup_algo_var, value="sha256").pack(side=tk.LEFT, padx=10)

        action_frame = ttk.Frame(settings_frame)
        action_frame.pack(fill=tk.X, pady=10)
        ttk.Button(action_frame, text="🔍 Найти дубликаты", command=self.run_find_dup, style="Accent.TButton").pack(side=tk.LEFT, padx=5)
        self.dup_progress = ttk.Progressbar(action_frame, mode='determinate')
        self.dup_progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        console_frame = ttk.Frame(main_container)
        console_frame.pack(fill=tk.BOTH, expand=True)
        self.dup_console = self.create_console(console_frame)

    def setup_stat_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="📊 Статистика")

        main_container = ttk.Frame(frame)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        settings_frame = ttk.Frame(main_container)
        settings_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(settings_frame, text="Статистика файлов", style="Title.TLabel").pack(anchor=tk.W, pady=(0, 20))

        dir_frame = ttk.Frame(settings_frame)
        dir_frame.pack(fill=tk.X, pady=5)
        ttk.Label(dir_frame, text="Директория:").pack(side=tk.LEFT, padx=5)
        self.stat_dir_var = tk.StringVar(value=str(Path.cwd()))
        ttk.Entry(dir_frame, textvariable=self.stat_dir_var, width=70).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(dir_frame, text="Обзор", command=lambda: self.browse_dir(self.stat_dir_var), style="Accent.TButton").pack(side=tk.LEFT, padx=5)

        export_frame = ttk.LabelFrame(settings_frame, text="Экспорт отчётов")
        export_frame.pack(fill=tk.X, pady=10)

        csv_frame = ttk.Frame(export_frame)
        csv_frame.pack(fill=tk.X, pady=5)
        ttk.Label(csv_frame, text="CSV отчёт:").pack(side=tk.LEFT, padx=5)
        self.stat_csv_var = tk.StringVar()
        ttk.Entry(csv_frame, textvariable=self.stat_csv_var, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(csv_frame, text="Обзор", command=lambda: self.browse_file_save(self.stat_csv_var, [("CSV", "*.csv")]), style="Accent.TButton").pack(side=tk.LEFT, padx=5)

        html_frame = ttk.Frame(export_frame)
        html_frame.pack(fill=tk.X, pady=5)
        ttk.Label(html_frame, text="HTML отчёт:").pack(side=tk.LEFT, padx=5)
        self.stat_html_var = tk.StringVar()
        ttk.Entry(html_frame, textvariable=self.stat_html_var, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(html_frame, text="Обзор", command=lambda: self.browse_file_save(self.stat_html_var, [("HTML", "*.html")]), style="Accent.TButton").pack(side=tk.LEFT, padx=5)

        action_frame = ttk.Frame(settings_frame)
        action_frame.pack(fill=tk.X, pady=10)
        ttk.Button(action_frame, text="📈 Получить статистику", command=self.run_stat, style="Success.TButton").pack(side=tk.LEFT, padx=5)
        self.stat_progress = ttk.Progressbar(action_frame, mode='determinate')
        self.stat_progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        console_frame = ttk.Frame(main_container)
        console_frame.pack(fill=tk.BOTH, expand=True)
        self.stat_console = self.create_console(console_frame)

    def setup_find_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="🔎 Поиск")

        main_container = ttk.Frame(frame)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        settings_frame = ttk.Frame(main_container)
        settings_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(settings_frame, text="Поиск файлов", style="Title.TLabel").pack(anchor=tk.W, pady=(0, 20))

        dir_frame = ttk.Frame(settings_frame)
        dir_frame.pack(fill=tk.X, pady=5)
        ttk.Label(dir_frame, text="Директория:").pack(side=tk.LEFT, padx=5)
        self.find_dir_var = tk.StringVar(value=str(Path.cwd()))
        ttk.Entry(dir_frame, textvariable=self.find_dir_var, width=70).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(dir_frame, text="Обзор", command=lambda: self.browse_dir(self.find_dir_var), style="Accent.TButton").pack(side=tk.LEFT, padx=5)

        patterns_frame = ttk.Frame(settings_frame)
        patterns_frame.pack(fill=tk.X, pady=5)
        ttk.Label(patterns_frame, text="Шаблоны (через пробел):").pack(side=tk.LEFT, padx=5)
        self.find_patterns_var = tk.StringVar()
        ttk.Entry(patterns_frame, textvariable=self.find_patterns_var, width=70).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        options_frame = ttk.LabelFrame(settings_frame, text="Параметры поиска")
        options_frame.pack(fill=tk.X, pady=10)

        options_row = ttk.Frame(options_frame)
        options_row.pack(fill=tk.X, pady=5)
        self.find_case_var = tk.BooleanVar()
        ttk.Checkbutton(options_row, text="Чувствительность к регистру", variable=self.find_case_var).pack(side=tk.LEFT, padx=5)
        self.find_recursive_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_row, text="Рекурсивный поиск", variable=self.find_recursive_var).pack(side=tk.LEFT, padx=5)

        action_frame = ttk.Frame(settings_frame)
        action_frame.pack(fill=tk.X, pady=10)
        ttk.Button(action_frame, text="🔍 Найти файлы", command=self.run_find, style="Accent.TButton").pack(side=tk.LEFT, padx=5)
        self.find_progress = ttk.Progressbar(action_frame, mode='determinate')
        self.find_progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        console_frame = ttk.Frame(main_container)
        console_frame.pack(fill=tk.BOTH, expand=True)
        self.find_console = self.create_console(console_frame)

    def setup_rename_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="✏️ Переименование")

        main_container = ttk.Frame(frame)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        settings_frame = ttk.Frame(main_container)
        settings_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(settings_frame, text="Пакетное переименование", style="Title.TLabel").pack(anchor=tk.W, pady=(0, 20))

        dir_frame = ttk.Frame(settings_frame)
        dir_frame.pack(fill=tk.X, pady=5)
        ttk.Label(dir_frame, text="Директория:").pack(side=tk.LEFT, padx=5)
        self.rename_dir_var = tk.StringVar(value=str(Path.cwd()))
        ttk.Entry(dir_frame, textvariable=self.rename_dir_var, width=70).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(dir_frame, text="Обзор", command=lambda: self.browse_dir(self.rename_dir_var), style="Accent.TButton").pack(side=tk.LEFT, padx=5)

        pattern_frame = ttk.Frame(settings_frame)
        pattern_frame.pack(fill=tk.X, pady=5)
        ttk.Label(pattern_frame, text="Шаблон для поиска:").pack(side=tk.LEFT, padx=5)
        self.rename_pattern_var = tk.StringVar()
        ttk.Entry(pattern_frame, textvariable=self.rename_pattern_var, width=70).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        replacement_frame = ttk.Frame(settings_frame)
        replacement_frame.pack(fill=tk.X, pady=5)
        ttk.Label(replacement_frame, text="Замена:").pack(side=tk.LEFT, padx=5)
        self.rename_replacement_var = tk.StringVar()
        ttk.Entry(replacement_frame, textvariable=self.rename_replacement_var, width=70).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        dry_run_frame = ttk.Frame(settings_frame)
        dry_run_frame.pack(fill=tk.X, pady=5)
        self.rename_dry_run_var = tk.BooleanVar()
        ttk.Checkbutton(dry_run_frame, text="Пробный запуск", variable=self.rename_dry_run_var).pack(side=tk.LEFT, padx=5)

        action_frame = ttk.Frame(settings_frame)
        action_frame.pack(fill=tk.X, pady=10)
        ttk.Button(action_frame, text="✏️ Переименовать файлы", command=self.run_rename, style="Accent.TButton").pack(side=tk.LEFT, padx=5)
        self.rename_progress = ttk.Progressbar(action_frame, mode='determinate')
        self.rename_progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        console_frame = ttk.Frame(main_container)
        console_frame.pack(fill=tk.BOTH, expand=True)
        self.rename_console = self.create_console(console_frame)

    def setup_extract_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="📦 Архивы")

        main_container = ttk.Frame(frame)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        settings_frame = ttk.Frame(main_container)
        settings_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(settings_frame, text="Извлечение архивов", style="Title.TLabel").pack(anchor=tk.W, pady=(0, 20))

        dir_frame = ttk.Frame(settings_frame)
        dir_frame.pack(fill=tk.X, pady=5)
        ttk.Label(dir_frame, text="Директория:").pack(side=tk.LEFT, padx=5)
        self.extract_dir_var = tk.StringVar(value=str(Path.cwd()))
        ttk.Entry(dir_frame, textvariable=self.extract_dir_var, width=70).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(dir_frame, text="Обзор", command=lambda: self.browse_dir(self.extract_dir_var), style="Accent.TButton").pack(side=tk.LEFT, padx=5)

        options_frame = ttk.LabelFrame(settings_frame, text="Параметры извлечения")
        options_frame.pack(fill=tk.X, pady=10)

        del_frame = ttk.Frame(options_frame)
        del_frame.pack(fill=tk.X, pady=5)
        self.extract_del_after_var = tk.BooleanVar()
        ttk.Checkbutton(del_frame, text="Удалить архивы после извлечения", variable=self.extract_del_after_var).pack(side=tk.LEFT, padx=5)

        types_frame = ttk.Frame(options_frame)
        types_frame.pack(fill=tk.X, pady=5)
        ttk.Label(types_frame, text="Типы архивов:").pack(side=tk.LEFT, padx=5)
        
        self.archive_vars = {}
        archive_types = ['.zip', '.rar', '.tar', '.gz', '.7z']
        for arch_type in archive_types:
            var = tk.BooleanVar(value=True)
            self.archive_vars[arch_type] = var
            ttk.Checkbutton(types_frame, text=arch_type, variable=var).pack(side=tk.LEFT, padx=10)

        action_frame = ttk.Frame(settings_frame)
        action_frame.pack(fill=tk.X, pady=10)
        ttk.Button(action_frame, text="📦 Извлечь архивы", command=self.run_extract, style="Accent.TButton").pack(side=tk.LEFT, padx=5)
        self.extract_progress = ttk.Progressbar(action_frame, mode='determinate')
        self.extract_progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        console_frame = ttk.Frame(main_container)
        console_frame.pack(fill=tk.BOTH, expand=True)
        self.extract_console = self.create_console(console_frame)

    def setup_clean_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="🧹 Очистка")

        main_container = ttk.Frame(frame)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        settings_frame = ttk.Frame(main_container)
        settings_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(settings_frame, text="Очистка пустых папок", style="Title.TLabel").pack(anchor=tk.W, pady=(0, 20))

        dir_frame = ttk.Frame(settings_frame)
        dir_frame.pack(fill=tk.X, pady=5)
        ttk.Label(dir_frame, text="Директория:").pack(side=tk.LEFT, padx=5)
        self.clean_dir_var = tk.StringVar(value=str(Path.cwd()))
        ttk.Entry(dir_frame, textvariable=self.clean_dir_var, width=70).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(dir_frame, text="Обзор", command=lambda: self.browse_dir(self.clean_dir_var), style="Accent.TButton").pack(side=tk.LEFT, padx=5)

        recursive_frame = ttk.Frame(settings_frame)
        recursive_frame.pack(fill=tk.X, pady=5)
        self.clean_recursive_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(recursive_frame, text="Рекурсивно", variable=self.clean_recursive_var).pack(side=tk.LEFT, padx=5)

        action_frame = ttk.Frame(settings_frame)
        action_frame.pack(fill=tk.X, pady=10)
        ttk.Button(action_frame, text="🧹 Очистить пустые папки", command=self.run_clean, style="Accent.TButton").pack(side=tk.LEFT, padx=5)
        self.clean_progress = ttk.Progressbar(action_frame, mode='determinate')
        self.clean_progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        console_frame = ttk.Frame(main_container)
        console_frame.pack(fill=tk.BOTH, expand=True)
        self.clean_console = self.create_console(console_frame)

    def setup_delete_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="🗑️ Удаление")

        main_container = ttk.Frame(frame)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        settings_frame = ttk.Frame(main_container)
        settings_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(settings_frame, text="Пакетное удаление файлов", style="Title.TLabel").pack(anchor=tk.W, pady=(0, 20))

        dir_frame = ttk.Frame(settings_frame)
        dir_frame.pack(fill=tk.X, pady=5)
        ttk.Label(dir_frame, text="Директория:").pack(side=tk.LEFT, padx=5)
        self.delete_dir_var = tk.StringVar(value=str(Path.cwd()))
        ttk.Entry(dir_frame, textvariable=self.delete_dir_var, width=70).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(dir_frame, text="Обзор", command=lambda: self.browse_dir(self.delete_dir_var), style="Accent.TButton").pack(side=tk.LEFT, padx=5)

        patterns_frame = ttk.Frame(settings_frame)
        patterns_frame.pack(fill=tk.X, pady=5)
        ttk.Label(patterns_frame, text="Шаблоны для удаления (через пробел):").pack(side=tk.LEFT, padx=5)
        self.delete_patterns_var = tk.StringVar()
        ttk.Entry(patterns_frame, textvariable=self.delete_patterns_var, width=70).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        options_frame = ttk.LabelFrame(settings_frame, text="Параметры удаления")
        options_frame.pack(fill=tk.X, pady=10)

        search_frame = ttk.Frame(options_frame)
        search_frame.pack(fill=tk.X, pady=5)
        self.delete_case_var = tk.BooleanVar()
        ttk.Checkbutton(search_frame, text="Чувствительность к регистру", variable=self.delete_case_var).pack(side=tk.LEFT, padx=5)
        self.delete_recursive_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(search_frame, text="Рекурсивный поиск", variable=self.delete_recursive_var).pack(side=tk.LEFT, padx=5)

        size_frame = ttk.Frame(options_frame)
        size_frame.pack(fill=tk.X, pady=5)
        ttk.Label(size_frame, text="Размер файла от:").pack(side=tk.LEFT, padx=5)
        self.delete_min_size_var = tk.IntVar(value=0)
        ttk.Entry(size_frame, textvariable=self.delete_min_size_var, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(size_frame, text="до:").pack(side=tk.LEFT, padx=5)
        self.delete_max_size_var = tk.StringVar(value="")
        ttk.Entry(size_frame, textvariable=self.delete_max_size_var, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(size_frame, text="байт").pack(side=tk.LEFT, padx=5)

        action_frame = ttk.Frame(settings_frame)
        action_frame.pack(fill=tk.X, pady=10)
        ttk.Button(action_frame, text="🗑️ Удалить файлы", command=self.run_delete, style="Danger.TButton").pack(side=tk.LEFT, padx=5)
        self.delete_progress = ttk.Progressbar(action_frame, mode='determinate')
        self.delete_progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        console_frame = ttk.Frame(main_container)
        console_frame.pack(fill=tk.BOTH, expand=True)
        self.delete_console = self.create_console(console_frame)

    def setup_file_manager_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="📂 Файловый менеджер")

        main_container = ttk.Frame(frame)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        top_frame = ttk.Frame(main_container)
        top_frame.pack(fill=tk.X, pady=(0, 10))

        path_frame = ttk.Frame(top_frame)
        path_frame.pack(fill=tk.X, pady=5)
        ttk.Label(path_frame, text="Текущая директория:").pack(side=tk.LEFT, padx=5)
        self.fm_dir_var = tk.StringVar(value=str(Path.cwd()))
        self.fm_path_entry = ttk.Entry(path_frame, textvariable=self.fm_dir_var, width=80)
        self.fm_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.fm_path_entry.bind('<Return>', lambda e: self.fm_refresh())
        
        ttk.Button(path_frame, text="Обзор", command=self.fm_browse, style="Accent.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(path_frame, text="🔄", command=self.fm_refresh, width=3, style="Accent.TButton").pack(side=tk.LEFT, padx=2)
        ttk.Button(path_frame, text="📁", command=self.fm_create_dir, width=3, style="Success.TButton").pack(side=tk.LEFT, padx=2)

        action_frame = ttk.Frame(top_frame)
        action_frame.pack(fill=tk.X, pady=5)
        ttk.Button(action_frame, text="📂 Открыть", command=self.fm_open_selected, style="Accent.TButton").pack(side=tk.LEFT, padx=2)
        ttk.Button(action_frame, text="✏️ Переименовать", command=self.fm_rename_selected, style="Warning.TButton").pack(side=tk.LEFT, padx=2)
        ttk.Button(action_frame, text="🗑️ Удалить", command=self.fm_delete_selected, style="Danger.TButton").pack(side=tk.LEFT, padx=2)
        ttk.Button(action_frame, text="ℹ️ Свойства", command=self.fm_properties, style="Accent.TButton").pack(side=tk.LEFT, padx=2)
        ttk.Button(action_frame, text="📊 Статистика", command=self.fm_stats, style="Success.TButton").pack(side=tk.LEFT, padx=2)

        middle_frame = ttk.Frame(main_container)
        middle_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        tree_frame = ttk.LabelFrame(middle_frame, text="Файлы и папки")
        tree_frame.pack(fill=tk.BOTH, expand=True, side=tk.LEFT, padx=(0, 5))

        self.fm_tree = ttk.Treeview(
            tree_frame, 
            columns=("size", "type", "modified", "permissions"), 
            show="headings",
            height=20,
            style="Custom.Treeview"
        )
        
        self.style.configure("Custom.Treeview", 
                            background="#3c3c3c",
                            foreground="#ffffff",
                            fieldbackground="#3c3c3c")
        self.style.configure("Custom.Treeview.Heading",
                            background="#2b2b2b",
                            foreground="#ffffff",
                            relief="flat")
        self.style.map("Custom.Treeview",
                    background=[('selected', '#3498db')],
                    foreground=[('selected', 'white')])
        
        self.fm_tree.heading("#0", text="Имя")
        self.fm_tree.heading("size", text="Размер")
        self.fm_tree.heading("type", text="Тип")
        self.fm_tree.heading("modified", text="Изменён")
        self.fm_tree.heading("permissions", text="Права")
        
        self.fm_tree.column("#0", width=300, anchor=tk.W)
        self.fm_tree.column("size", width=100, anchor=tk.E)
        self.fm_tree.column("type", width=80, anchor=tk.CENTER)
        self.fm_tree.column("modified", width=120, anchor=tk.CENTER)
        self.fm_tree.column("permissions", width=80, anchor=tk.CENTER)

        scrollbar_y = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.fm_tree.yview)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.fm_tree.configure(yscrollcommand=scrollbar_y.set)

        scrollbar_x = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.fm_tree.xview)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.fm_tree.configure(xscrollcommand=scrollbar_x.set)

        self.fm_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fm_tree.bind('<Double-1>', lambda e: self.fm_open_selected())
        self.fm_tree.bind('<Return>', lambda e: self.fm_open_selected())
        self.fm_tree.bind('<<TreeviewSelect>>', lambda e: self.fm_update_info())

        info_frame = ttk.LabelFrame(middle_frame, text="Информация")
        info_frame.pack(fill=tk.BOTH, expand=True, side=tk.RIGHT, padx=(5, 0))
        
        self.info_text = scrolledtext.ScrolledText(
            info_frame, 
            wrap=tk.WORD, 
            bg="#1e1e1e", 
            fg="#ffffff", 
            font=("Consolas", 9), 
            state=tk.DISABLED
        )
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        console_frame = ttk.Frame(main_container)
        console_frame.pack(fill=tk.BOTH, expand=True)
        self.fm_console = self.create_console(console_frame, height=6)

        self.fm_refresh()

    def setup_logs_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="📋 Логи")

        main_frame = ttk.Frame(frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(main_frame, text="Журнал операций", style="Title.TLabel").pack(anchor=tk.W, pady=(0, 10))

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=5)

        ttk.Button(button_frame, text="🔄 Обновить", command=self.update_logs, style="Accent.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="🧹 Очистить", command=self.clear_logs, style="Warning.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="💾 Сохранить", command=self.save_logs, style="Success.TButton").pack(side=tk.LEFT, padx=5)

        self.log_text = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, height=25, font=("Consolas", 10),
                                                bg="#1e1e1e", fg="#00ff00")
        self.log_text.pack(fill=tk.BOTH, expand=True)

        self.update_logs()

    def setup_about_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="ℹ️ О программе")

        main_frame = ttk.Frame(frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        ttk.Label(main_frame, text="Filer v1.0 - gui", style="Title.TLabel").pack(pady=10)

        about_text = """
GitHub: https://github.com/QUIK1001/Event-Horizon
Telegram: https://t.me/Event_Horizon_Shell

Внимание: Будьте осторожны при удалении файлов!
Все операции выполняются на ваш страх и риск.
        """

        about_display = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, height=25, font=("Segoe UI", 11),
                                                bg="#000000", fg="#FFFFFF")
        about_display.pack(fill=tk.BOTH, expand=True, pady=10)
        about_display.insert(tk.END, about_text)
        about_display.config(state=tk.DISABLED)


    def browse_dir(self, var):
        directory = filedialog.askdirectory(initialdir=var.get())
        if directory:
            var.set(directory)

    def browse_file_save(self, var, filetypes):
        file_path = filedialog.asksaveasfilename(defaultextension=filetypes[0][1][1:], filetypes=filetypes)
        if file_path:
            var.set(file_path)

    def log_message(self, message):
        self.organizer.log_message(message)
        self.update_logs()

    def update_logs(self):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        logs = self.organizer.get_logs()
        for log in logs[-1000:]:
            self.log_text.insert(tk.END, log + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def clear_logs(self):
        self.organizer.log_buffer.clear()
        self.update_logs()

    def save_logs(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(self.organizer.get_logs()))

    def run_in_thread(self, func, *args, **kwargs):
        progress_callback = kwargs.pop('progress_callback', None)
        log_callback = kwargs.pop('log_callback', None)
        completion_callback = kwargs.pop('completion_callback', None)
        console_callback = kwargs.pop('console_callback', None)

        def worker():
            try:
                self.status_var.set("Выполняется...")
                result = func(*args, progress_callback=progress_callback, log_callback=log_callback, **kwargs)
                self.root.after(0, lambda: self.status_var.set("Готов"))
                if completion_callback:
                    self.root.after(0, completion_callback, result)
            except Exception as e:
                self.root.after(0, lambda: self.status_var.set("Ошибка"))
                self.root.after(0, lambda: messagebox.showerror("Ошибка", str(e)))
                self.log_message(f"ОШИБКА: {e}")
                if console_callback:
                    self.root.after(0, lambda: self.log_to_console(console_callback, f"ОШИБКА: {e}"))

        thread = threading.Thread(target=worker)
        thread.daemon = True
        thread.start()

    def run_org(self):
        self.org_progress['value'] = 0
        self.clear_console(self.org_console)
        
        def log_callback(message):
            self.log_message(message)
            self.log_to_console(self.org_console, message)
            
        self.run_in_thread(
            self.organizer.org_files,
            self.org_dir_var.get(),
            organize_by_date=self.org_by_date_var.get(),
            dry_run=self.org_dry_run_var.get(),
            copy=self.org_copy_var.get(),
            backup_dir=self.org_backup_dir_var.get() if self.org_backup_dir_var.get() else None,
            progress_callback=lambda current, total: self.org_progress.config(value=(current / total) * 100) if total > 0 else None,
            log_callback=log_callback,
            console_callback=self.org_console
        )

    def run_find_dup(self):
        self.dup_progress['value'] = 0
        self.clear_console(self.dup_console)
        
        def log_callback(message):
            self.log_message(message)
            self.log_to_console(self.dup_console, message)
            
        def completion_callback(result):
            messagebox.showinfo("Результат", f"Найдено дубликатов: {result}")
            
        self.run_in_thread(
            self.organizer.find_dup,
            self.dup_dir_var.get(),
            delete=self.dup_del_var.get(),
            min_size=self.dup_min_size_var.get(),
            algorithm=self.dup_algo_var.get(),
            interactive=self.dup_interactive_var.get(),
            progress_callback=lambda current, total: self.dup_progress.config(value=(current / total) * 100) if total > 0 else None,
            log_callback=log_callback,
            completion_callback=completion_callback,
            console_callback=self.dup_console
        )

    def run_stat(self):
        self.stat_progress['value'] = 0
        self.clear_console(self.stat_console)
        
        def completion_callback(_):
            messagebox.showinfo("Готово", "Статистика собрана успешно!")
            self.log_to_console(self.stat_console, "Статистика собрана успешно!")
            
        self.run_in_thread(
            self.organizer.get_file_stats,
            self.stat_dir_var.get(),
            export_csv=self.stat_csv_var.get() if self.stat_csv_var.get() else None,
            export_html=self.stat_html_var.get() if self.stat_html_var.get() else None,
            progress_callback=lambda current, total: self.stat_progress.config(value=(current / total) * 100) if total > 0 else None,
            completion_callback=completion_callback,
            console_callback=self.stat_console
        )

    def run_find(self):
        self.find_progress['value'] = 0
        self.clear_console(self.find_console)
        
        def log_callback(message):
            self.log_message(message)
            self.log_to_console(self.find_console, message)
            
        def completion_callback(result):
            messagebox.showinfo("Результат", f"Найдено файлов: {len(result)}")
            
        self.run_in_thread(
            self.organizer.find_files,
            self.find_dir_var.get(),
            self.find_patterns_var.get().split(),
            case_sensitive=self.find_case_var.get(),
            recursive=self.find_recursive_var.get(),
            progress_callback=lambda current, total: self.find_progress.config(value=(current / total) * 100) if total > 0 else None,
            log_callback=log_callback,
            completion_callback=completion_callback,
            console_callback=self.find_console
        )

    def run_rename(self):
        self.rename_progress['value'] = 0
        self.clear_console(self.rename_console)
        
        def log_callback(message):
            self.log_message(message)
            self.log_to_console(self.rename_console, message)
            
        def completion_callback(result):
            messagebox.showinfo("Результат", f"Переименовано файлов: {result}")
            
        self.run_in_thread(
            self.organizer.rename_files,
            self.rename_dir_var.get(),
            self.rename_pattern_var.get(),
            self.rename_replacement_var.get(),
            dry_run=self.rename_dry_run_var.get(),
            progress_callback=lambda current, total: self.rename_progress.config(value=(current / total) * 100) if total > 0 else None,
            log_callback=log_callback,
            completion_callback=completion_callback,
            console_callback=self.rename_console
        )

    def run_extract(self):
        self.extract_progress['value'] = 0
        self.clear_console(self.extract_console)
        
        def log_callback(message):
            self.log_message(message)
            self.log_to_console(self.extract_console, message)
            
        def completion_callback(result):
            messagebox.showinfo("Результат", f"Извлечено архивов: {result}")
        
        selected_archives = [arch_type for arch_type, var in self.archive_vars.items() if var.get()]
        
        self.run_in_thread(
            self.organizer.extract_archives,
            self.extract_dir_var.get(),
            delete_after=self.extract_del_after_var.get(),
            archive_types=selected_archives,
            progress_callback=lambda current, total: self.extract_progress.config(value=(current / total) * 100) if total > 0 else None,
            log_callback=log_callback,
            completion_callback=completion_callback,
            console_callback=self.extract_console
        )

    def run_clean(self):
        self.clean_progress['value'] = 0
        self.clear_console(self.clean_console)
        
        def log_callback(message):
            self.log_message(message)
            self.log_to_console(self.clean_console, message)
            
        def completion_callback(result):
            messagebox.showinfo("Результат", f"Очищено пустых папок: {result}")
            
        self.run_in_thread(
            self.organizer.clean_empty_dirs,
            self.clean_dir_var.get(),
            recursive=self.clean_recursive_var.get(),
            progress_callback=lambda current, total: self.clean_progress.config(value=(current / total) * 100) if total > 0 else None,
            log_callback=log_callback,
            completion_callback=completion_callback,
            console_callback=self.clean_console
        )

    def run_delete(self):
        self.delete_progress['value'] = 0
        self.clear_console(self.delete_console)
        
        def log_callback(message):
            self.log_message(message)
            self.log_to_console(self.delete_console, message)
            
        def completion_callback(result):
            messagebox.showinfo("Результат", f"Удалено файлов: {result}")
        
        max_size = float('inf') if not self.delete_max_size_var.get() else int(self.delete_max_size_var.get())
        
        self.run_in_thread(
            self.organizer.bulk_delete,
            self.delete_dir_var.get(),
            self.delete_patterns_var.get().split(),
            case_sensitive=self.delete_case_var.get(),
            recursive=self.delete_recursive_var.get(),
            min_size=self.delete_min_size_var.get(),
            max_size=max_size,
            progress_callback=lambda current, total: self.delete_progress.config(value=(current / total) * 100) if total > 0 else None,
            log_callback=log_callback,
            completion_callback=completion_callback,
            console_callback=self.delete_console
        )

    def fm_browse(self):
        directory = filedialog.askdirectory(initialdir=self.fm_dir_var.get())
        if directory:
            self.fm_dir_var.set(directory)
            self.fm_refresh()
            self.log_to_console(self.fm_console, f"Переход в директорию: {directory}")

    def fm_refresh(self):
        for item in self.fm_tree.get_children():
            self.fm_tree.delete(item)

        path = Path(self.fm_dir_var.get())
        if not path.exists():
            return

        try:
            dirs = []
            files = []
            
            for item in path.iterdir():
                if item.is_dir():
                    dirs.append(item)
                else:
                    files.append(item)

            dirs.sort(key=lambda x: x.name.lower())
            files.sort(key=lambda x: x.name.lower())

            for directory in dirs:
                try:
                    mtime = datetime.fromtimestamp(directory.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                    permissions = self.get_permissions(directory)
                    full_path = str(directory.resolve())
                    item_id = self.fm_tree.insert(
                        "", 
                        tk.END, 
                        text=f"📁 {directory.name}", 
                        values=("", "Папка", mtime, permissions),
                        tags=('directory',),
                        iid=full_path 
                    )
                except OSError as e:
                    print(f"Ошибка при добавлении папки {directory}: {e}")

            for file in files:
                try:
                    size = file.stat().st_size
                    mtime = datetime.fromtimestamp(file.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                    permissions = self.get_permissions(file)
                    file_icon = self.get_file_icon(file.suffix.lower())
                    full_path = str(file.resolve())
                    item_id = self.fm_tree.insert(
                        "", 
                        tk.END, 
                        text=f"{file_icon} {file.name}", 
                        values=(self.organizer._format_size(size), file.suffix or "Файл", mtime, permissions),
                        tags=('file',),
                        iid=full_path
                    )
                except OSError as e:
                    print(f"Ошибка при добавлении файла {file}: {e}")
            
            self.fm_tree.tag_configure('directory', foreground='#3498db')
            self.fm_tree.tag_configure('file', foreground='#ffffff')
                    
        except OSError as e:
            messagebox.showerror("Ошибка", f"Не удалось прочитать директорию: {e}")
            self.log_to_console(self.fm_console, f"ОШИБКА: {e}")

    def get_permissions(self, path):
        """Получить права доступа в формате rwx"""
        try:
            stat_info = path.stat()
            permissions = stat_info.st_mode
            result = ""
            
            result += 'r' if permissions & 0o400 else '-'
            result += 'w' if permissions & 0o200 else '-'
            result += 'x' if permissions & 0o100 else '-'
            
            result += 'r' if permissions & 0o040 else '-'
            result += 'w' if permissions & 0o020 else '-'
            result += 'x' if permissions & 0o010 else '-'
            
            result += 'r' if permissions & 0o004 else '-'
            result += 'w' if permissions & 0o002 else '-'
            result += 'x' if permissions & 0o001 else '-'
            
            return result
        except:
            return "---------"

    def get_file_icon(self, extension):
        icons = {
            '.pdf': '📄', '.doc': '📄', '.docx': '📄', '.txt': '📄',
            '.jpg': '🖼️', '.jpeg': '🖼️', '.png': '🖼️', '.gif': '🖼️',
            '.mp3': '🎵', '.wav': '🎵', '.flac': '🎵',
            '.mp4': '🎬', '.avi': '🎬', '.mkv': '🎬',
            '.zip': '📦', '.rar': '📦', '.7z': '📦',
            '.exe': '⚙️', '.msi': '⚙️',
            '.py': '🐍', '.js': '📜', '.html': '🌐', '.css': '🎨'
        }
        return icons.get(extension, '📄')

    def fm_update_info(self):
        """Обновить информацию о выбранном файле/папке"""
        selected = self.fm_tree.selection()
        if not selected:
            self.info_text.config(state=tk.NORMAL)
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(tk.END, "Выберите файл или папку для просмотра информации")
            self.info_text.config(state=tk.DISABLED)
            return

        path_str = selected[0]
        
        try:
            path = Path(path_str)
            
            if not path.exists():
                self.info_text.config(state=tk.NORMAL)
                self.info_text.delete(1.0, tk.END)
                self.info_text.insert(tk.END, f"Файл/папка не существует:\n{path_str}")
                self.info_text.config(state=tk.DISABLED)
                return
                
            stat_info = path.stat()
            info = f"Имя: {path.name}\n"
            info += f"Путь: {path}\n"
            info += f"Тип: {'Папка' if path.is_dir() else 'Файл'}\n"
            
            if path.is_file():
                info += f"Размер: {self.organizer._format_size(stat_info.st_size)}\n"
                info += f"Расширение: {path.suffix or 'нет'}\n"
            
            info += f"Создан: {datetime.fromtimestamp(stat_info.st_ctime).strftime('%Y-%m-%d %H:%M:%S')}\n"
            info += f"Изменён: {datetime.fromtimestamp(stat_info.st_mtime).strftime('%Y-%m-%d %H:%M:%S')}\n"
            info += f"Права доступа: {self.get_permissions(path)}\n"
            info += f"Владелец: {stat_info.st_uid}\n"
            info += f"Группа: {stat_info.st_gid}"
            
            self.info_text.config(state=tk.NORMAL)
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(tk.END, info)
            self.info_text.config(state=tk.DISABLED)
            
        except Exception as e:
            self.info_text.config(state=tk.NORMAL)
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(tk.END, f"Ошибка получения информации: {e}\nПуть: {path_str}")
            self.info_text.config(state=tk.DISABLED)

    def fm_open_selected(self):
        selected = self.fm_tree.selection()
        if selected:
            path_str = selected[0]
            try:
                path = Path(path_str)
                if not path.exists():
                    messagebox.showerror("Ошибка", f"Файл/папка не существует: {path_str}")
                    return
                    
                if path.is_file():
                    try:
                        if os.name == 'nt':
                            os.startfile(path)
                        elif sys.platform == 'darwin':
                            subprocess.run(['open', path])
                        else:
                            subprocess.run(['xdg-open', path])
                        self.log_to_console(self.fm_console, f"Открыт файл: {path.name}")
                    except Exception as e:
                        messagebox.showerror("Ошибка", f"Не удалось открыть файл: {e}")
                        self.log_to_console(self.fm_console, f"ОШИБКА открытия файла: {e}")
                else:
                    self.fm_dir_var.set(path_str)
                    self.fm_refresh()
                    self.log_to_console(self.fm_console, f"Открыта папка: {path.name}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Неверный путь: {path_str}\nОшибка: {e}")

    def fm_delete_selected(self):
        selected = self.fm_tree.selection()
        if selected:
            path_str = selected[0]
            try:
                path = Path(path_str)
                if not path.exists():
                    messagebox.showerror("Ошибка", f"Файл/папка не существует: {path_str}")
                    return
                    
                if messagebox.askyesno("Удалить", f"Удалить '{path.name}'?"):
                    try:
                        if path.is_file():
                            path.unlink()
                            self.log_message(f"УДАЛЕН ФАЙЛ: {path}")
                            self.log_to_console(self.fm_console, f"Удален файл: {path.name}")
                        else:
                            shutil.rmtree(path)
                            self.log_message(f"УДАЛЕНА ПАПКА: {path}")
                            self.log_to_console(self.fm_console, f"Удалена папка: {path.name}")
                        self.fm_refresh()
                    except OSError as e:
                        messagebox.showerror("Ошибка", f"Не удалось удалить: {e}")
                        self.log_to_console(self.fm_console, f"ОШИБКА удаления: {e}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Неверный путь: {path_str}\nОшибка: {e}")

    def fm_rename_selected(self):
        selected = self.fm_tree.selection()
        if selected:
            path_str = selected[0]
            try:
                path = Path(path_str)
                if not path.exists():
                    messagebox.showerror("Ошибка", f"Файл/папка не существует: {path_str}")
                    return
                    
                new_name = tk.simpledialog.askstring("Переименовать", "Введите новое имя:", initialvalue=path.name)
                if new_name and new_name != path.name:
                    try:
                        new_path = path.parent / new_name
                        path.rename(new_path)
                        self.log_message(f"ПЕРЕИМЕНОВАНО: {path.name} -> {new_name}")
                        self.log_to_console(self.fm_console, f"Переименовано: {path.name} -> {new_name}")
                        self.fm_refresh()
                    except OSError as e:
                        messagebox.showerror("Ошибка", f"Не удалось переименовать: {e}")
                        self.log_to_console(self.fm_console, f"ОШИБКА переименования: {e}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Неверный путь: {path_str}\nОшибка: {e}")

    def fm_create_dir(self):
        dir_name = tk.simpledialog.askstring("Создать папку", "Введите имя новой папки:")
        if dir_name:
            try:
                new_dir = Path(self.fm_dir_var.get()) / dir_name
                new_dir.mkdir(exist_ok=True)
                self.log_message(f"СОЗДАНА ПАПКА: {new_dir}")
                self.log_to_console(self.fm_console, f"Создана папка: {dir_name}")
                self.fm_refresh()
            except OSError as e:
                messagebox.showerror("Ошибка", f"Не удалось создать папку: {e}")
                self.log_to_console(self.fm_console, f"ОШИБКА создания папки: {e}")

    def fm_properties(self):
        selected = self.fm_tree.selection()
        if selected:
            path_str = selected[0]
            path = Path(path_str)
            
            try:
                stat = path.stat()
                size = self.organizer._format_size(stat.st_size)
                mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                ctime = datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S")
                atime = datetime.fromtimestamp(stat.st_atime).strftime("%Y-%m-%d %H:%M:%S")
                
                info = f"Имя: {path.name}\n"
                info += f"Путь: {path}\n"
                info += f"Тип: {'Папка' if path.is_dir() else 'Файл'}\n"
                info += f"Размер: {size}\n"
                info += f"Создан: {ctime}\n"
                info += f"Изменён: {mtime}\n"
                info += f"Открыт: {atime}\n"
                info += f"Права доступа: {self.get_permissions(path)}\n"
                info += f"Владелец: {stat.st_uid}\n"
                info += f"Группа: {stat.st_gid}\n"
                info += f"Inode: {stat.st_ino}"
                
                messagebox.showinfo("Свойства", info)
            except OSError as e:
                messagebox.showerror("Ошибка", f"Не удалось получить свойства: {e}")

    def fm_stats(self):
        path = Path(self.fm_dir_var.get())
        try:
            file_count = 0
            dir_count = 0
            total_size = 0
            
            for item in path.rglob('*'):
                if item.is_file():
                    file_count += 1
                    total_size += item.stat().st_size
                else:
                    dir_count += 1
            
            stats = f"Директория: {path}\n"
            stats += f"Файлов: {file_count}\n"
            stats += f"Папок: {dir_count}\n"
            stats += f"Общий размер: {self.organizer._format_size(total_size)}"
            
            messagebox.showinfo("Статистика", stats)
            self.log_to_console(self.fm_console, f"Статистика для {path}: {file_count} файлов, {dir_count} папок, {self.organizer._format_size(total_size)}")
        except OSError as e:
            messagebox.showerror("Ошибка", f"Не удалось собрать статистику: {e}")
            self.log_to_console(self.fm_console, f"ОШИБКА статистики: {e}")


def main():
    root = tk.Tk()
    app = ModernFileOrganizerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()