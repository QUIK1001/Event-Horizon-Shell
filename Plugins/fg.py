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
import csv
from typing import Dict, List
import io
import webbrowser
from tkinter import filedialog, messagebox, ttk, simpledialog
import tkinter as tk
from tkinter import scrolledtext
import threading
import subprocess
import platform
import psutil
import chardet
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import tempfile
from PIL import Image, ImageTk

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

IS_WINDOWS = platform.system() == 'Windows'
IS_MAC = platform.system() == 'Darwin'
IS_LINUX = platform.system() == 'Linux'

class EnhancedFileOrganizer:
    def __init__(self):
        self.extensions = self.get_ext_cfg()
        self.stats = defaultdict(lambda: {'count': 0, 'size': 0})
        self.log_buffer = []
        self.setup_mime_types()

    def setup_mime_types(self):
        if IS_WINDOWS:
            mimetypes.add_type('application/x-msdownload', '.exe')
            mimetypes.add_type('application/x-msi', '.msi')
        elif IS_MAC:
            mimetypes.add_type('application/x-apple-diskimage', '.dmg')
            mimetypes.add_type('application/x-macbinary', '.bin')

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
                'application': self._categorize_application(mime_type),
                'audio': 'audio',
                'video': 'video'
            }
            return mime_map.get(mime_major, 'other')
        
        return 'unknown'

    def _categorize_application(self, mime_type: str) -> str:
        if 'zip' in mime_type or 'archive' in mime_type:
            return 'archives'
        elif 'pdf' in mime_type:
            return 'documents'
        elif 'executable' in mime_type or 'octet-stream' in mime_type:
            return 'executables'
        return 'code'

    def log_message(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        self.log_buffer.append(formatted_message)
        if len(self.log_buffer) > 1000:
            self.log_buffer.pop(0)

    def get_logs(self) -> List[str]:
        return self.log_buffer.copy()

    def org_files(self, directory, organize_by_date=False, date_format="%Y/%m", 
                  dry_run=False, copy=False, backup_dir=None, 
                  progress_callback=None, log_callback=None, 
                  custom_categories=None):
        try:
            directory = Path(directory)
            self.stats.clear()
            current_script = Path(__file__).resolve()
            
            files_to_process = []
            for f in directory.iterdir():
                if (f.is_file() and 
                    f.resolve() != current_script and 
                    not self._is_system_file(f)):
                    files_to_process.append(f)
            
            total_files = len(files_to_process)
            processed = 0

            categories = custom_categories if custom_categories else self.extensions

            for file_path in files_to_process:
                try:
                    category = self._smart_categorize_with_custom(file_path, categories)
                    
                    if organize_by_date:
                        mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                        date_folder = mtime.strftime(date_format)
                        category_dir = directory / category / date_folder
                    else:
                        category_dir = directory / category

                    category_dir.mkdir(parents=True, exist_ok=True)
                    dest_path = category_dir / file_path.name
                    
                    dest_path = self._resolve_naming_conflict(dest_path)
                    
                    if not dry_run:
                        file_size = file_path.stat().st_size
                        if copy:
                            shutil.copy2(str(file_path), dest_path)
                            if log_callback:
                                log_callback(f"СКОПИРОВАНО: {file_path.name} -> {dest_path}")
                        else:
                            if backup_dir:
                                self._create_backup(file_path, backup_dir, log_callback)
                            
                            if file_path.resolve() != dest_path.resolve():
                                shutil.move(str(file_path), dest_path)
                                if log_callback:
                                    log_callback(f"ПЕРЕМЕЩЕНО: {file_path.name} -> {dest_path}")
                            else:
                                if log_callback:
                                    log_callback(f"ФАЙЛ УЖЕ НА МЕСТЕ: {file_path.name}")

                        if dest_path.name != file_path.name:
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
                        
                except Exception as e:
                    if log_callback:
                        log_callback(f"ОШИБКА обработки {file_path.name}: {e}")
                    processed += 1
                    if progress_callback:
                        progress_callback(processed, total_files)
                        
        except Exception as e:
            if log_callback:
                log_callback(f"КРИТИЧЕСКАЯ ОШИБКА: {e}")
            raise

    def _smart_categorize_with_custom(self, file_path: Path, categories: Dict) -> str:
        ext = file_path.suffix.lower()
        for category, exts in categories.items():
            if ext in exts:
                return category
        return self._smart_categorize(file_path)

    def _is_system_file(self, file_path: Path) -> bool:
        system_patterns = [
            'Thumbs.db', '.DS_Store', 'desktop.ini',
            '.*',
        ]
        
        filename = file_path.name
        if any(fnmatch.fnmatch(filename, pattern) for pattern in system_patterns):
            return True
        
        if IS_WINDOWS:
            try:
                import ctypes
                FILE_ATTRIBUTE_SYSTEM = 0x4
                FILE_ATTRIBUTE_HIDDEN = 0x2
                attrs = ctypes.windll.kernel32.GetFileAttributesW(str(file_path))
                if attrs != -1 and (attrs & (FILE_ATTRIBUTE_SYSTEM | FILE_ATTRIBUTE_HIDDEN)):
                    return True
            except:
                pass
        
        return False

    def _resolve_naming_conflict(self, dest_path: Path) -> Path:
        if not dest_path.exists():
            return dest_path
            
        counter = 1
        original_dest = dest_path
        stem, suffix = dest_path.stem, dest_path.suffix
        
        while dest_path.exists():
            new_name = f"{stem}_{counter:03d}{suffix}"
            dest_path = original_dest.parent / new_name
            counter += 1
            if counter > 1000:
                break
                
        return dest_path

    def _create_backup(self, file_path: Path, backup_dir: str, log_callback=None):
        backup_dir_path = Path(backup_dir)
        backup_dir_path.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir_path / file_path.name
        
        backup_path = self._resolve_naming_conflict(backup_path)
        shutil.copy2(str(file_path), backup_path)
        
        if log_callback:
            log_callback(f"РЕЗЕРВНАЯ КОПИЯ: {file_path.name} -> {backup_path}")

    def find_dup(self, directory, delete=False, min_size=0, algorithm='md5', 
                 interactive=False, progress_callback=None, log_callback=None,
                 parallel=True):
        directory = Path(directory)
        hash_func = {
            'md5': hashlib.md5,
            'sha1': hashlib.sha1,
            'sha256': hashlib.sha256,
            'blake2b': hashlib.blake2b
        }.get(algorithm, hashlib.md5)

        file_info_list = []
        total_files = 0
        
        for file_path in directory.rglob('*'):
            if file_path.is_file() and not self._is_system_file(file_path):
                total_files += 1

        processed = 0
        for file_path in directory.rglob('*'):
            if file_path.is_file() and not self._is_system_file(file_path):
                try:
                    size = file_path.stat().st_size
                    if size >= min_size:
                        file_info_list.append({
                            'path': file_path,
                            'size': size,
                            'mtime': file_path.stat().st_mtime
                        })
                except OSError:
                    pass
            
            processed += 1
            if progress_callback:
                progress_callback(processed, total_files)

        size_groups = defaultdict(list)
        for file_info in file_info_list:
            size_groups[file_info['size']].append(file_info)

        duplicates = []
        
        if parallel:
            duplicates = self._find_duplicates_parallel(size_groups, hash_func, 
                                                       progress_callback, log_callback)
        else:
            duplicates = self._find_duplicates_sequential(size_groups, hash_func, 
                                                         progress_callback, log_callback)

        if duplicates and delete:
            deleted_count = self._delete_duplicates([d[0] for d in duplicates], log_callback)
            if log_callback:
                log_callback(f"УДАЛЕНО ДУБЛИКАТОВ: {deleted_count}")
            return deleted_count

        return len(duplicates)

    def _find_duplicates_parallel(self, size_groups, hash_func, progress_callback, log_callback):
        duplicates = []
        total_groups = len([g for g in size_groups.values() if len(g) > 1])
        processed_groups = 0
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_group = {}
            
            for size, files in size_groups.items():
                if len(files) > 1:
                    future = executor.submit(self._process_file_group, files, hash_func)
                    future_to_group[future] = (size, files)
            
            for future in as_completed(future_to_group):
                size, files = future_to_group[future]
                try:
                    group_duplicates = future.result()
                    duplicates.extend(group_duplicates)
                except Exception as e:
                    if log_callback:
                        log_callback(f"ОШИБКА обработки группы файлов размером {size}: {e}")
                
                processed_groups += 1
                if progress_callback:
                    progress_callback(processed_groups, total_groups)
        
        return duplicates

    def _find_duplicates_sequential(self, size_groups, hash_func, progress_callback, log_callback):
        duplicates = []
        total_groups = len([g for g in size_groups.values() if len(g) > 1])
        processed_groups = 0
        
        for size, files in size_groups.items():
            if len(files) > 1:
                group_duplicates = self._process_file_group(files, hash_func)
                duplicates.extend(group_duplicates)
                
            processed_groups += 1
            if progress_callback:
                progress_callback(processed_groups, total_groups)
        
        return duplicates

    def _process_file_group(self, files, hash_func):
        hashes = defaultdict(list)
        for file_info in files:
            try:
                file_hash = self._calculate_hash(file_info['path'], hash_func)
                file_info['hash'] = file_hash
                hashes[file_hash].append(file_info)
            except (IOError, OSError):
                continue

        group_duplicates = []
        for file_hash, file_list in hashes.items():
            if len(file_list) > 1:
                file_list.sort(key=lambda x: x['mtime'])
                original = file_list[0]
                for dup in file_list[1:]:
                    group_duplicates.append((dup['path'], original['path']))
        
        return group_duplicates

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
                if not os.access(dup, os.W_OK):
                    if log_callback:
                        log_callback(f"НЕТ ПРАВ ДЛЯ УДАЛЕНИЯ: {dup}")
                    continue
                    
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

        empty_dirs.sort(key=lambda x: len(x.parts), reverse=True)
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

    def get_file_stats(self, directory):
        try:
            directory = Path(directory)
            if not directory.exists():
                return {"error": f"Директория не существует: {directory}"}
            
            self.stats.clear()
            
            for file_path in directory.rglob('*'):
                try:
                    if file_path.is_file() and not self._is_system_file(file_path):
                        category = self._smart_categorize(file_path)
                        file_size = file_path.stat().st_size
                        self.stats[category]['count'] += 1
                        self.stats[category]['size'] += file_size
                except (OSError, PermissionError):
                    continue
                    
            return dict(self.stats)
        except Exception as e:
            return {"error": str(e)}

    def run_stats(self):
        try:
            stats = self.organizer.get_file_stats(self.tools_dir_var.get())
            
            stats_text = "Статистика файлов:\n\n"
            total_files = 0
            total_size = 0
            
            for category, data in stats.items():
                stats_text += f"{category}: {data['count']} файлов, {self.organizer._format_size(data['size'])}\n"
                total_files += data['count']
                total_size += data['size']
            
            stats_text += f"\nИтого: {total_files} файлов, {self.organizer._format_size(total_size)}"
            messagebox.showinfo("Статистика", stats_text)
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось получить статистику: {e}")

    def _format_size(self, size_bytes):
        """Форматирование размера файла (дублирование метода из organizer)"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"

    def _export_stats_csv(self, all_files, export_path):
        with open(export_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['File', 'Size_Bytes', 'Size_MB', 'Category', 'Extension', 'Modified'])
            
            for file_path in all_files:
                size = file_path.stat().st_size
                category = self._smart_categorize(file_path)
                mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                
                writer.writerow([
                    str(file_path),
                    size,
                    size / (1024 * 1024),
                    category,
                    file_path.suffix.lower(),
                    mtime.strftime("%Y-%m-%d %H:%M:%S")
                ])

    def _export_stats_html(self, all_files, export_path):
        html_content = self._generate_html_report(all_files)
        with open(export_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def _generate_html_report(self, all_files):
        total_size = sum(f.stat().st_size for f in all_files)
        total_files = len(all_files)
        
        sorted_categories = sorted(self.stats.items(), 
                                 key=lambda x: x[1]['size'], reverse=True)
        
        html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Отчёт Filer</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #1e1e1e; color: #ffffff; }
        .container { max-width: 1200px; margin: 0 auto; background: #2d2d30; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.3); }
        h1 { color: #0078d4; border-bottom: 2px solid #0078d4; padding-bottom: 10px; }
        .summary { background: #252526; padding: 20px; border-radius: 8px; margin: 20px 0; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
        .stat-card { background: #3e3e42; padding: 15px; border-radius: 8px; border-left: 4px solid #0078d4; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { border: 1px solid #444; padding: 12px; text-align: left; }
        th { background-color: #0078d4; color: white; }
        tr:nth-child(even) { background-color: #2d2d30; }
        .category-badge { padding: 4px 8px; border-radius: 4px; color: white; font-size: 0.8em; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Отчёт о файлах</h1>
        <div class="summary">
            <h3>📈 Сводка</h3>
            <div class="stats-grid">
                <div class="stat-card">
                    <strong>Всего файлов:</strong><br>
                    <span style="font-size: 1.5em; color: #ffffff;">""" + str(total_files) + """</span>
                </div>
                <div class="stat-card">
                    <strong>Общий размер:</strong><br>
                    <span style="font-size: 1.5em; color: #ffffff;">""" + self._format_size(total_size) + """</span>
                </div>
                <div class="stat-card">
                    <strong>Категорий:</strong><br>
                    <span style="font-size: 1.5em; color: #ffffff;">""" + str(len(self.stats)) + """</span>
                </div>
            </div>
        </div>
        
        <h3>📂 Распределение по категориям</h3>
        <table>
            <tr><th>Категория</th><th>Файлов</th><th>Размер</th><th>Процент</th></tr>
"""
        
        for category, data in sorted_categories:
            percentage = (data['size'] / total_size * 100) if total_size > 0 else 0
            html_content += f"""
            <tr>
                <td><span class="category-badge" style="background: #{hash(category) % 0xFFFFFF:06x}">{category}</span></td>
                <td>{data['count']}</td>
                <td>{self._format_size(data['size'])}</td>
                <td>{percentage:.1f}%</td>
            </tr>
"""
        
        html_content += """
        </table>
        
        <h3>📋 Детальная информация</h3>
        <table>
            <tr><th>Файл</th><th>Размер (Байт)</th><th>Размер (МБ)</th><th>Категория</th><th>Расширение</th><th>Изменён</th></tr>
"""
        
        for file_path in sorted(all_files, key=lambda x: x.stat().st_size, reverse=True)[:1000]:
            size = file_path.stat().st_size
            category = self._smart_categorize(file_path)
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            
            html_content += f"""
            <tr>
                <td title="{file_path}">{file_path.name}</td>
                <td>{size}</td>
                <td>{size / (1024 * 1024):.2f}</td>
                <td><span class="category-badge" style="background: #{hash(category) % 0xFFFFFF:06x}">{category}</span></td>
                <td>{file_path.suffix.lower()}</td>
                <td>{mtime.strftime('%Y-%m-%d %H:%M')}</td>
            </tr>
"""
        
        html_content += """
        </table>
    </div>
</body>
</html>"""
        
        return html_content

    def _export_stats_json(self, all_files, export_path):
        stats_data = {
            'summary': {
                'total_files': len(all_files),
                'total_size': sum(f.stat().st_size for f in all_files),
                'categories_count': len(self.stats)
            },
            'categories': dict(self.stats),
            'files': [
                {
                    'path': str(f),
                    'name': f.name,
                    'size': f.stat().st_size,
                    'category': self._smart_categorize(f),
                    'extension': f.suffix.lower(),
                    'modified': f.stat().st_mtime
                } for f in all_files
            ]
        }
        
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, indent=2, ensure_ascii=False)

    def find_files_advanced(self, directory, patterns=None, case_sensitive=False, 
                           recursive=True, progress_callback=None, log_callback=None,
                           content_search=None, file_size_min=0, file_size_max=float('inf'),
                           modified_after=None, modified_before=None, 
                           file_types=None, exclude_patterns=None,
                           content_regex=False, whole_words=False, max_results=1000,
                           encoding_override=None, search_in_binary=False):
        directory = Path(directory)
        matches = []
        
        if recursive:
            walk_iter = list(directory.rglob('*'))
        else:
            walk_iter = list(directory.iterdir())
            
        total_files = len(walk_iter)
        processed = 0

        search_patterns = self._prepare_patterns(patterns, case_sensitive)
        exclude_patterns_list = self._prepare_patterns(exclude_patterns or [], case_sensitive)
        file_types_set = set(ft.lower() for ft in (file_types or []))
        
        for file_path in walk_iter:
            if len(matches) >= max_results:
                if log_callback:
                    log_callback(f"ДОСТИГНУТ ЛИМИТ РЕЗУЛЬТАТОВ: {max_results}")
                break
                
            if file_path.is_file() and not self._is_system_file(file_path):
                filename = file_path.name if case_sensitive else file_path.name.lower()
                
                name_matches = True
                if search_patterns:
                    name_matches = any(fnmatch.fnmatch(filename, pattern) for pattern in search_patterns)
                
                if name_matches and exclude_patterns_list:
                    name_matches = not any(fnmatch.fnmatch(filename, pattern) for pattern in exclude_patterns_list)
                
                if name_matches and file_types_set:
                    file_ext = file_path.suffix.lower()
                    name_matches = file_ext in file_types_set
                
                file_size = file_path.stat().st_size
                size_matches = file_size_min <= file_size <= file_size_max
                
                time_matches = True
                file_mtime = file_path.stat().st_mtime
                if modified_after:
                    time_matches = time_matches and (file_mtime >= modified_after.timestamp())
                if modified_before:
                    time_matches = time_matches and (file_mtime <= modified_before.timestamp())
                
                content_matches = True
                if content_search and name_matches and size_matches and time_matches:
                    content_matches = self._advanced_search_in_file(
                        file_path, content_search, case_sensitive, 
                        content_regex, whole_words, encoding_override,
                        search_in_binary, log_callback
                    )
                
                if name_matches and size_matches and time_matches and content_matches:
                    matches.append(file_path)
                    if log_callback:
                        log_callback(f"НАЙДЕН: {file_path}")
            
            processed += 1
            if progress_callback:
                progress_callback(processed, total_files)
                
        return matches

    def _prepare_patterns(self, patterns, case_sensitive):
        if not patterns:
            return None
        
        if case_sensitive:
            return patterns
        else:
            return [p.lower() for p in patterns]

    def _advanced_search_in_file(self, file_path: Path, search_term: str, 
                               case_sensitive: bool, use_regex: bool, 
                               whole_words: bool, encoding_override: str,
                               search_in_binary: bool, log_callback=None) -> bool:
        try:
            if not search_in_binary and self._is_binary_file(file_path):
                return False
            
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                
            if encoding_override:
                encoding = encoding_override
            else:
                encoding = self._detect_encoding(raw_data, file_path)
            
            try:
                if encoding.lower() == 'binary' and search_in_binary:
                    content = raw_data
                    search_term = search_term.encode() if isinstance(search_term, str) else search_term
                else:
                    content = raw_data.decode(encoding, errors='ignore')
                    
                    if not case_sensitive:
                        content = content.lower()
                        search_term = search_term.lower()
                
                if use_regex:
                    return self._regex_search(content, search_term, whole_words)
                else:
                    return self._text_search(content, search_term, whole_words)
                    
            except UnicodeDecodeError:
                if search_in_binary:
                    content = raw_data
                    search_term_bytes = search_term.encode() if isinstance(search_term, str) else search_term
                    return search_term_bytes in content
                return False
                
        except Exception as e:
            if log_callback:
                log_callback(f"ОШИБКА ЧТЕНИЯ {file_path}: {e}")
            return False

    def _is_binary_file(self, file_path: Path) -> bool:
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(1024)
                return b'\0' in chunk
        except:
            return True

    def _detect_encoding(self, raw_data: bytes, file_path: Path) -> str:
        if not raw_data:
            return 'utf-8'
        
        bom_encodings = {
            b'\xff\xfe': 'utf-16-le',
            b'\xfe\xff': 'utf-16-be', 
            b'\xef\xbb\xbf': 'utf-8-sig',
            b'\xff\xfe\x00\x00': 'utf-32-le',
            b'\x00\x00\xfe\xff': 'utf-32-be'
        }
        
        for bom, encoding in bom_encodings.items():
            if raw_data.startswith(bom):
                return encoding
        
        try:
            result = chardet.detect(raw_data[:4096])
            confidence = result.get('confidence', 0)
            encoding = result.get('encoding', 'utf-8')
            
            if confidence > 0.7:
                return encoding
        except:
            pass
        
        text_extensions = ['.txt', '.md', '.csv', '.json', '.xml', '.html', '.htm', 
                          '.css', '.js', '.py', '.java', '.cpp', '.c', '.h', '.php']
        
        if file_path.suffix.lower() in text_extensions:
            return 'utf-8'
        
        return 'latin-1'

    def _regex_search(self, content: str, pattern: str, whole_words: bool) -> bool:
        try:
            if whole_words:
                pattern = r'\b' + pattern + r'\b'
            
            flags = 0 if whole_words else re.IGNORECASE
            return bool(re.search(pattern, content, flags))
        except re.error:
            return False

    def _text_search(self, content: str, search_term: str, whole_words: bool) -> bool:
        if whole_words:
            words = re.findall(r'\b\w+\b', content)
            return search_term in words
        else:
            return search_term in content

    def find_in_files_batch(self, directory, search_terms, operation='AND', 
                           progress_callback=None, log_callback=None, **kwargs):
        all_matches = []
        
        for i, term in enumerate(search_terms):
            if log_callback:
                log_callback(f"Поиск термина {i+1}/{len(search_terms)}: '{term}'")
            
            matches = self.find_files_advanced(
                directory, 
                content_search=term,
                progress_callback=lambda current, total: progress_callback(
                    current + i * total, len(search_terms) * total
                ) if progress_callback else None,
                log_callback=log_callback,
                **kwargs
            )
            
            if operation.upper() == 'AND':
                if i == 0:
                    all_matches = set(matches)
                else:
                    all_matches = all_matches.intersection(set(matches))
            else:
                all_matches.extend(matches)
        
        return list(all_matches) if operation.upper() == 'AND' else all_matches

    def rename_files(self, directory, pattern, replacement, dry_run=False, 
                    progress_callback=None, log_callback=None, regex=False):
        directory = Path(directory)
        files = [f for f in directory.iterdir() if f.is_file() and not self._is_system_file(f)]
        total_files = len(files)
        processed = 0
        renamed_count = 0

        for file_path in files:
            if file_path.is_file():
                if regex:
                    try:
                        new_name = re.sub(pattern, replacement, file_path.name)
                    except re.error as e:
                        if log_callback:
                            log_callback(f"ОШИБКА РЕГУЛЯРНОГО ВЫРАЖЕНИЯ: {e}")
                        continue
                else:
                    new_name = file_path.name.replace(pattern, replacement)
                
                if new_name != file_path.name:
                    new_path = file_path.parent / new_name
                    
                    if new_path.exists():
                        if log_callback:
                            log_callback(f"КОНФЛИКТ ИМЕН: {new_name} уже существует")
                        continue
                    
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

    def extract_archives(self, directory, delete_after=False, archive_types=None, 
                        progress_callback=None, log_callback=None, password=None):
        directory = Path(directory)
        if archive_types is None:
            archive_types = ['.zip', '.rar', '.tar', '.gz', '.7z']
        
        archives = [f for f in directory.iterdir() 
                   if f.is_file() and f.suffix.lower() in archive_types and not self._is_system_file(f)]
        
        total_archives = len(archives)
        processed = 0
        extracted_count = 0

        for file_path in archives:
            extract_dir = file_path.parent / file_path.stem
            extract_dir.mkdir(exist_ok=True)
            
            try:
                success = False
                
                if file_path.suffix.lower() == '.zip':
                    success = self._extract_zip(file_path, extract_dir, password, log_callback)
                elif file_path.suffix.lower() in ['.tar', '.gz', '.bz2', '.xz']:
                    success = self._extract_tar(file_path, extract_dir, log_callback)
                elif file_path.suffix.lower() == '.7z':
                    success = self._extract_7z(file_path, extract_dir, password, log_callback)
                elif file_path.suffix.lower() == '.rar':
                    success = self._extract_rar(file_path, extract_dir, password, log_callback)
                
                if success:
                    extracted_count += 1
                    if log_callback:
                        log_callback(f"ИЗВЛЕЧЕН: {file_path} -> {extract_dir}")
                    
                    if delete_after:
                        file_path.unlink()
                        if log_callback:
                            log_callback(f"УДАЛЕН АРХИВ: {file_path}")
                
            except Exception as e:
                if log_callback:
                    log_callback(f"ОШИБКА ИЗВЛЕЧЕНИЯ {file_path}: {e}")
            
            processed += 1
            if progress_callback:
                progress_callback(processed, total_archives)
                
        return extracted_count

    def _extract_zip(self, file_path, extract_dir, password, log_callback):
        try:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                if password:
                    zip_ref.setpassword(password.encode())
                zip_ref.extractall(extract_dir)
            return True
        except (zipfile.BadZipFile, RuntimeError) as e:
            if log_callback:
                log_callback(f"ОШИБКА ZIP: {file_path} - {e}")
            return False

    def _extract_tar(self, file_path, extract_dir, log_callback):
        try:
            with tarfile.open(file_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_dir)
            return True
        except (tarfile.ReadError, EOFError) as e:
            if log_callback:
                log_callback(f"ОШИБКА TAR: {file_path} - {e}")
            return False

    def _extract_7z(self, file_path, extract_dir, password, log_callback):
        try:
            if IS_WINDOWS:
                cmd = ['7z', 'x', str(file_path), f'-o{extract_dir}']
                if password:
                    cmd.extend(['-p', password])
                subprocess.run(cmd, check=True, capture_output=True)
                return True
            else:
                cmd = ['7z', 'x', str(file_path), f'-o{extract_dir}']
                if password:
                    cmd.extend(['-p', password])
                subprocess.run(cmd, check=True, capture_output=True)
                return True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            if log_callback:
                log_callback(f"7z не установлен или ошибка: {e}")
            return False

    def _extract_rar(self, file_path, extract_dir, password, log_callback):
        try:
            if IS_WINDOWS:
                cmd = ['unrar', 'x', '-y', str(file_path), str(extract_dir)]
                if password:
                    cmd.extend(['-p', password])
                subprocess.run(cmd, check=True, capture_output=True)
                return True
            else:
                cmd = ['unrar', 'x', '-y', str(file_path), str(extract_dir)]
                if password:
                    cmd.extend(['-p', password])
                subprocess.run(cmd, check=True, capture_output=True)
                return True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            if log_callback:
                log_callback(f"unrar не установлен или ошибка: {e}")
            return False

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
                loaded_config = json.load(f)
                self.extensions.update(loaded_config)
                return loaded_config
        except FileNotFoundError:
            return None
        except json.JSONDecodeError:
            return None

    def bulk_delete(self, directory, patterns, case_sensitive=False, 
                   recursive=True, min_size=0, max_size=float('inf'), 
                   progress_callback=None, log_callback=None):
        directory = Path(directory)
        files_to_delete = []
        
        if recursive:
            walk_iter = list(directory.rglob('*'))
        else:
            walk_iter = list(directory.iterdir())
        
        total_files = len(walk_iter)
        processed = 0

        for file_path in walk_iter:
            if file_path.is_file() and not self._is_system_file(file_path):
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
                if not os.access(file_path, os.W_OK):
                    if log_callback:
                        log_callback(f"НЕТ ПРАВ ДЛЯ УДАЛЕНИЯ: {file_path}")
                    continue
                    
                file_path.unlink()
                deleted_count += 1
                if log_callback:
                    log_callback(f"УДАЛЕН: {file_path}")
            except OSError as e:
                if log_callback:
                    log_callback(f"ОШИБКА УДАЛЕНИЯ {file_path}: {e}")

        return deleted_count

    def create_archives(self, directory, archive_format='zip', compression_level=6,
                       progress_callback=None, log_callback=None):
        directory = Path(directory)
        folders_to_archive = [f for f in directory.iterdir() if f.is_dir()]
        
        total_folders = len(folders_to_archive)
        processed = 0
        archived_count = 0

        for folder in folders_to_archive:
            archive_name = f"{folder.name}.{archive_format}"
            archive_path = directory / archive_name
            
            try:
                if archive_format == 'zip':
                    with zipfile.ZipFile(archive_path, 'w', 
                                      compression=zipfile.ZIP_DEFLATED, 
                                      compresslevel=compression_level) as zipf:
                        for file_path in folder.rglob('*'):
                            if file_path.is_file():
                                arcname = file_path.relative_to(folder)
                                zipf.write(file_path, arcname)
                
                elif archive_format in ['tar', 'gz', 'bz2']:
                    mode = {
                        'tar': 'w',
                        'gz': 'w:gz',
                        'bz2': 'w:bz2'
                    }[archive_format]
                    
                    with tarfile.open(archive_path, mode) as tarf:
                        tarf.add(folder, arcname=folder.name)
                
                archived_count += 1
                if log_callback:
                    log_callback(f"СОЗДАН АРХИВ: {archive_path}")
                    
            except Exception as e:
                if log_callback:
                    log_callback(f"ОШИБКА СОЗДАНИЯ АРХИВА {folder}: {e}")
            
            processed += 1
            if progress_callback:
                progress_callback(processed, total_folders)
                
        return archived_count

    def sync_folders(self, source, destination, delete_extraneous=False,
                    progress_callback=None, log_callback=None):
        source = Path(source)
        destination = Path(destination)
        
        if not source.exists():
            if log_callback:
                log_callback(f"ИСТОЧНИК НЕ СУЩЕСТВУЕТ: {source}")
            return 0
        
        destination.mkdir(parents=True, exist_ok=True)
        
        source_files = {}
        for file_path in source.rglob('*'):
            if file_path.is_file():
                relative_path = file_path.relative_to(source)
                source_files[relative_path] = file_path.stat().st_mtime
        dest_files = {}
        for file_path in destination.rglob('*'):
            if file_path.is_file():
                relative_path = file_path.relative_to(destination)
                dest_files[relative_path] = file_path.stat().st_mtime
        
        copied_count = 0
        for relative_path, mtime in source_files.items():
            dest_path = destination / relative_path
            source_path = source / relative_path
            
            if (relative_path not in dest_files or 
                dest_files[relative_path] < mtime):
                
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, dest_path)
                copied_count += 1
                if log_callback:
                    log_callback(f"СКОПИРОВАН: {relative_path}")
        
        deleted_count = 0
        if delete_extraneous:
            for relative_path in dest_files:
                if relative_path not in source_files:
                    file_path = destination / relative_path
                    file_path.unlink()
                    deleted_count += 1
                    if log_callback:
                        log_callback(f"УДАЛЕН: {relative_path}")
        
        if progress_callback:
            progress_callback(1, 1)
            
        return copied_count + deleted_count

    def get_disk_usage(self, directory):
        directory = Path(directory)
        usage = psutil.disk_usage(str(directory))
        
        return {
            'total': usage.total,
            'used': usage.used,
            'free': usage.free,
            'percent': usage.percent
        }

    def create_archive_from_files(self, files, archive_path, archive_format='zip', compression_level=6):
        try:
            archive_path = Path(archive_path)
            
            if archive_format == 'zip':
                with zipfile.ZipFile(archive_path, 'w', 
                                  compression=zipfile.ZIP_DEFLATED, 
                                  compresslevel=compression_level) as zipf:
                    for file_path in files:
                        if file_path.exists():
                            if file_path.is_file():
                                arcname = file_path.name
                                zipf.write(file_path, arcname)
                            elif file_path.is_dir():
                                for sub_file in file_path.rglob('*'):
                                    if sub_file.is_file():
                                        arcname = sub_file.relative_to(file_path.parent)
                                        zipf.write(sub_file, arcname)
                return True
                
            elif archive_format in ['tar', 'gz', 'bz2']:
                mode = {
                    'tar': 'w',
                    'gz': 'w:gz',
                    'bz2': 'w:bz2'
                }[archive_format]
                
                with tarfile.open(archive_path, mode) as tarf:
                    for file_path in files:
                        if file_path.exists():
                            tarf.add(file_path, arcname=file_path.name)
                return True
                
        except Exception as e:
            print(f"Ошибка создания архива: {e}")
            return False

    def copy_files(self, source_files, destination_dir):
        destination_dir = Path(destination_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)
        
        copied_count = 0
        for source_path in source_files:
            try:
                if source_path.exists():
                    dest_path = destination_dir / source_path.name
                    dest_path = self._resolve_naming_conflict(dest_path)
                    
                    if source_path.is_file():
                        shutil.copy2(source_path, dest_path)
                        copied_count += 1
                    elif source_path.is_dir():
                        shutil.copytree(source_path, dest_path)
                        copied_count += 1
            except Exception as e:
                print(f"Ошибка копирования {source_path}: {e}")
        
        return copied_count

    def move_files(self, source_files, destination_dir):
        destination_dir = Path(destination_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)
        
        moved_count = 0
        for source_path in source_files:
            try:
                if source_path.exists():
                    dest_path = destination_dir / source_path.name
                    dest_path = self._resolve_naming_conflict(dest_path)
                    
                    shutil.move(str(source_path), str(dest_path))
                    moved_count += 1
            except Exception as e:
                print(f"Ошибка перемещения {source_path}: {e}")
        
        return moved_count


class ModernFileOrganizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Filer v2.0-beta")
        self.root.geometry("1400x950")
        self.root.minsize(1400, 850)
        
        self.fm_history = []
        self.fm_history_index = -1
        self.selected_items = []
        self.clipboard_files = []
        self.clipboard_operation = None
        
        self.setup_dark_theme()
        
        self.organizer = EnhancedFileOrganizer()
        
        self.auto_load_var = tk.BooleanVar(value=True)
        self.auto_stats_var = tk.BooleanVar(value=True)
        self.confirm_delete_var = tk.BooleanVar(value=True)
        self.theme_var = tk.StringVar(value="dark")
        self.font_size_var = tk.StringVar(value="normal")
        self.chunk_size_var = tk.StringVar(value="8192")
        self.max_threads_var = tk.IntVar(value=4)
        self.notify_complete_var = tk.BooleanVar(value=True)
        self.notify_errors_var = tk.BooleanVar(value=True)
        self.notify_large_ops_var = tk.BooleanVar(value=True)
        
        self.setup_styles()
        self.setup_ui()
        
        self.load_config()
        
        self.load_current_settings()
        self.apply_theme(self.theme_var.get())
        self.apply_font_size(self.font_size_var.get())
        
        self.start_performance_monitor()
        self.tools_dir_var = tk.StringVar(value=str(Path.cwd()))

    def setup_dark_theme(self):
        self.bg_color = "#1e1e1e"
        self.fg_color = "#ffffff"
        self.accent_color = "#0078d4"
        self.secondary_bg = "#2d2d30"
        self.tertiary_bg = "#3e3e42"
        self.text_color = "#cccccc"
        self.border_color = "#444444"
        
        self.success_color = "#27ae60"
        self.warning_color = "#f39c12"
        self.danger_color = "#e74c3c"
        self.info_color = "#3498db"
        
        self.root.configure(bg=self.bg_color)

    def setup_styles(self):
        self.style = ttk.Style()
        
        self.style.theme_use("clam")
        
        self.style.configure(".", 
                           background=self.bg_color,
                           foreground=self.fg_color,
                           fieldbackground=self.secondary_bg)
        
        self.style.configure("TFrame", background=self.bg_color)
        self.style.configure("TLabel", background=self.bg_color, 
                           foreground=self.fg_color, font=("Segoe UI", 10))
        self.style.configure("TButton", font=("Segoe UI", 10), padding=8,
                           background=self.secondary_bg,
                           foreground=self.fg_color,
                           borderwidth=1,
                           focuscolor=self.secondary_bg)
        self.style.configure("TCheckbutton", background=self.bg_color, 
                           foreground=self.fg_color)
        self.style.configure("TRadiobutton", background=self.bg_color, 
                           foreground=self.fg_color)
        
        self.style.configure("TProgressbar", 
                           thickness=20, 
                           background=self.accent_color,
                           troughcolor=self.secondary_bg)
        
        self.style.configure("TNotebook", background=self.bg_color)
        self.style.configure("TNotebook.Tab", 
                           background=self.secondary_bg,
                           foreground=self.text_color,
                           padding=[15, 5])
        
        self.style.map("TNotebook.Tab", 
                      background=[("selected", self.accent_color),
                                ("active", self.accent_color)],
                      foreground=[("selected", "white"),
                                ("active", "white")])
        
        self.style.configure("TEntry", 
                           fieldbackground=self.secondary_bg,
                           foreground=self.fg_color,
                           insertcolor=self.fg_color)
        
        self.style.configure("TScrollbar", 
                           background=self.secondary_bg,
                           troughcolor=self.bg_color)
        
        self.style.configure("Title.TLabel", 
                           font=("Segoe UI", 12, "bold"),
                           foreground=self.accent_color)
        
        self.style.configure("Treeview",
                           background=self.secondary_bg,
                           foreground=self.fg_color,
                           fieldbackground=self.secondary_bg)
        
        self.style.configure("Treeview.Heading",
                           background=self.tertiary_bg,
                           foreground=self.fg_color)
        
        self.style.map("Treeview",
                      background=[('selected', self.accent_color)])
        
        self.style.configure("Danger.TButton",
                           background=self.danger_color,
                           foreground="white")
        self.style.map("Danger.TButton",
                      background=[('active', '#c0392b')])
        
        self.style.configure("Success.TButton",
                           background=self.success_color,
                           foreground="white")
        self.style.map("Success.TButton",
                      background=[('active', '#219653')])

    def setup_ui(self):
        self.setup_menu()
        
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.setup_file_manager_tab()
        self.setup_organization_tab()
        self.setup_search_tab()
        self.setup_tools_tab()
        self.setup_logs_tab()
        self.setup_settings_tab()
        self.setup_about_tab()
        
        self.setup_status_bar()

    def setup_menu(self):
        menubar = tk.Menu(self.root, bg=self.secondary_bg, fg=self.fg_color)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0, bg=self.secondary_bg, fg=self.fg_color)
        menubar.add_cascade(label="Файл", menu=file_menu)
        file_menu.add_command(label="Новое окно", command=self.new_window)
        file_menu.add_separator()
        file_menu.add_command(label="Выход", command=self.root.quit)
        
        help_menu = tk.Menu(menubar, tearoff=0, bg=self.secondary_bg, fg=self.fg_color)
        menubar.add_cascade(label="Помощь", menu=help_menu)
        help_menu.add_command(label="О программе", command=self.show_about)

    def new_window(self):
        new_root = tk.Toplevel(self.root)
        new_app = ModernFileOrganizerGUI(new_root)

    def show_about(self):
        self.notebook.select(len(self.notebook.tabs())-1)

    def setup_file_manager_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="📂 Файловый менеджер")

        main_container = ttk.Frame(frame)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        top_frame = ttk.Frame(main_container)
        top_frame.pack(fill=tk.X, pady=(0, 10))

        nav_frame = ttk.Frame(top_frame)
        nav_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(nav_frame, text="⬆️ Наверх", command=self.fm_parent).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="← Назад", command=self.fm_back).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="→ Вперёд", command=self.fm_forward).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="🏠 Home", command=self.fm_home).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="/ Корень", command=self.fm_root).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="🔄 Обновить", command=self.fm_refresh).pack(side=tk.LEFT, padx=2)

        path_frame = ttk.Frame(top_frame)
        path_frame.pack(fill=tk.X, pady=5)
        ttk.Label(path_frame, text="Путь:").pack(side=tk.LEFT, padx=5)
        self.fm_dir_var = tk.StringVar(value=str(Path.cwd()))
        self.fm_path_entry = ttk.Entry(path_frame, textvariable=self.fm_dir_var, width=80)
        self.fm_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.fm_path_entry.bind('<Return>', lambda e: self.fm_refresh())
        
        ttk.Button(path_frame, text="Обзор", command=self.fm_browse).pack(side=tk.LEFT, padx=5)

        quick_actions_frame = ttk.LabelFrame(top_frame, text="Быстрые действия")
        quick_actions_frame.pack(fill=tk.X, pady=5)
        
        quick_actions_row1 = ttk.Frame(quick_actions_frame)
        quick_actions_row1.pack(fill=tk.X, pady=2)
        
        ttk.Button(quick_actions_row1, text="📁 Создать папку", command=self.fm_create_dir).pack(side=tk.LEFT, padx=2)
        ttk.Button(quick_actions_row1, text="📄 Создать файл", command=self.fm_create_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(quick_actions_row1, text="🔍 Быстрый поиск", command=self.fm_quick_search).pack(side=tk.LEFT, padx=2)
        ttk.Button(quick_actions_row1, text="📊 Статистика", command=self.fm_stats).pack(side=tk.LEFT, padx=2)
        ttk.Button(quick_actions_row1, text="🧹 Очистить папки", command=self.fm_clean_empty).pack(side=tk.LEFT, padx=2)

        quick_actions_row2 = ttk.Frame(quick_actions_frame)
        quick_actions_row2.pack(fill=tk.X, pady=2)
        
        ttk.Button(quick_actions_row2, text="📦 Архивировать", command=self.fm_archive_selected).pack(side=tk.LEFT, padx=2)
        ttk.Button(quick_actions_row2, text="📤 Извлечь", command=self.fm_extract_selected).pack(side=tk.LEFT, padx=2)
        ttk.Button(quick_actions_row2, text="🔄 Организовать", command=self.fm_organize_current).pack(side=tk.LEFT, padx=2)
        ttk.Button(quick_actions_row2, text="🔍 Дубликаты", command=self.fm_find_duplicates).pack(side=tk.LEFT, padx=2)

        middle_frame = ttk.Frame(main_container)
        middle_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        tree_frame = ttk.LabelFrame(middle_frame, text="Файлы и папки")
        tree_frame.pack(fill=tk.BOTH, expand=True, side=tk.LEFT, padx=(0, 5))

        tree_toolbar = ttk.Frame(tree_frame)
        tree_toolbar.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(tree_toolbar, text="Сортировка:").pack(side=tk.LEFT, padx=5)
        self.sort_var = tk.StringVar(value="name")
        ttk.Combobox(tree_toolbar, textvariable=self.sort_var, 
                    values=["name", "size", "type", "modified"], 
                    state="readonly", width=12).pack(side=tk.LEFT, padx=5)
        self.sort_var.trace('w', lambda *args: self.fm_refresh())
        
        ttk.Label(tree_toolbar, text="Вид:").pack(side=tk.LEFT, padx=5)
        self.view_var = tk.StringVar(value="details")
        ttk.Combobox(tree_toolbar, textvariable=self.view_var,
                    values=["details", "list", "icons"],
                    state="readonly", width=10).pack(side=tk.LEFT, padx=5)
        self.view_var.trace('w', lambda *args: self.fm_refresh())

        self.fm_tree = ttk.Treeview(
            tree_frame, 
            columns=("name", "size", "type", "modified", "permissions", "full_path"), 
            show="headings",
            height=20,
            selectmode='extended'
        )
        
        self.fm_tree.heading("name", text="Имя", command=lambda: self.sort_treeview("name"))
        self.fm_tree.heading("size", text="Размер", command=lambda: self.sort_treeview("size"))
        self.fm_tree.heading("type", text="Тип", command=lambda: self.sort_treeview("type"))
        self.fm_tree.heading("modified", text="Изменён", command=lambda: self.sort_treeview("modified"))
        self.fm_tree.heading("permissions", text="Права")
        self.fm_tree.heading("full_path", text="Полный путь")
        
        self.fm_tree.column("name", width=300, anchor=tk.W)
        self.fm_tree.column("size", width=100, anchor=tk.E)
        self.fm_tree.column("type", width=80, anchor=tk.CENTER)
        self.fm_tree.column("modified", width=120, anchor=tk.CENTER)
        self.fm_tree.column("permissions", width=80, anchor=tk.CENTER)
        self.fm_tree.column("full_path", width=0, stretch=False)

        scrollbar_y = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.fm_tree.yview)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.fm_tree.configure(yscrollcommand=scrollbar_y.set)

        scrollbar_x = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.fm_tree.xview)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.fm_tree.configure(xscrollcommand=scrollbar_x.set)

        self.fm_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.tree_context_menu = tk.Menu(self.fm_tree, tearoff=0, bg=self.secondary_bg, fg=self.fg_color)
        self.tree_context_menu.add_command(label="Открыть", command=self.fm_open_selected)
        self.tree_context_menu.add_command(label="Открыть в проводнике", command=self.fm_open_in_explorer)
        self.tree_context_menu.add_separator()
        self.tree_context_menu.add_command(label="Копировать", command=self.fm_copy_selected)
        self.tree_context_menu.add_command(label="Вырезать", command=self.fm_cut_selected)
        self.tree_context_menu.add_command(label="Вставить", command=self.fm_paste_selected)
        self.tree_context_menu.add_separator()
        self.tree_context_menu.add_command(label="Переименовать", command=self.fm_rename_selected)
        self.tree_context_menu.add_command(label="Удалить", command=self.fm_delete_selected)
        self.tree_context_menu.add_separator()
        self.tree_context_menu.add_command(label="Свойства", command=self.fm_properties)

        self.fm_tree.bind('<Double-1>', lambda e: self.fm_open_selected())
        self.fm_tree.bind('<Return>', lambda e: self.fm_open_selected())
        self.fm_tree.bind('<<TreeviewSelect>>', lambda e: self.fm_update_info())
        self.fm_tree.bind('<Button-3>', self.show_tree_context_menu)

        info_frame = ttk.LabelFrame(middle_frame, text="Информация и предпросмотр")
        info_frame.pack(fill=tk.BOTH, expand=True, side=tk.RIGHT, padx=(5, 0))
        
        quick_info_frame = ttk.Frame(info_frame)
        quick_info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.quick_info_text = tk.Text(quick_info_frame, height=4, wrap=tk.WORD, 
                                     bg=self.secondary_bg, fg=self.fg_color,
                                     font=("Consolas", 9), state=tk.DISABLED)
        self.quick_info_text.pack(fill=tk.X)
        
        preview_frame = ttk.LabelFrame(info_frame, text="Предпросмотр")
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.preview_text = scrolledtext.ScrolledText(
            preview_frame, 
            wrap=tk.WORD, 
            bg=self.secondary_bg, 
            fg=self.fg_color, 
            font=("Consolas", 9), 
            state=tk.DISABLED
        )
        self.preview_text.pack(fill=tk.BOTH, expand=True)

        bottom_frame = ttk.Frame(main_container)
        bottom_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(bottom_frame, text="📂 Открыть", command=self.fm_open_selected).pack(side=tk.LEFT, padx=2)
        ttk.Button(bottom_frame, text="✏️ Переименовать", command=self.fm_rename_selected).pack(side=tk.LEFT, padx=2)
        ttk.Button(bottom_frame, text="🗑️ Удалить", command=self.fm_delete_selected, style="Danger.TButton").pack(side=tk.LEFT, padx=2)
        ttk.Button(bottom_frame, text="📋 Копировать", command=self.fm_copy_selected).pack(side=tk.LEFT, padx=2)
        ttk.Button(bottom_frame, text="✂️ Вырезать", command=self.fm_cut_selected).pack(side=tk.LEFT, padx=2)
        ttk.Button(bottom_frame, text="📝 Вставить", command=self.fm_paste_selected).pack(side=tk.LEFT, padx=2)
        ttk.Button(bottom_frame, text="➡️ Переместить в...", command=self.fm_move_selected).pack(side=tk.LEFT, padx=2)

        console_frame = ttk.LabelFrame(main_container, text="Журнал операций")
        console_frame.pack(fill=tk.BOTH, expand=True)
        self.fm_console = self.create_console(console_frame, height=6)

        self.fm_refresh()

    def setup_organization_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="🔄 Организация")

        main_container = ttk.Frame(frame)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(main_container, text="Организация файлов", style="Title.TLabel").pack(anchor=tk.W, pady=(0, 20))

        settings_frame = ttk.LabelFrame(main_container, text="Настройки организации")
        settings_frame.pack(fill=tk.X, pady=10)

        dir_frame = ttk.Frame(settings_frame)
        dir_frame.pack(fill=tk.X, pady=5)
        ttk.Label(dir_frame, text="Директория:").pack(side=tk.LEFT, padx=5)
        self.org_dir_var = tk.StringVar(value=str(Path.cwd()))
        ttk.Entry(dir_frame, textvariable=self.org_dir_var, width=60).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(dir_frame, text="Обзор", command=lambda: self.browse_dir_or_file(self.org_dir_var)).pack(side=tk.LEFT, padx=5)

        options_frame = ttk.Frame(settings_frame)
        options_frame.pack(fill=tk.X, pady=5)
        
        self.org_by_date_var = tk.BooleanVar()
        ttk.Checkbutton(options_frame, text="По дате изменения", variable=self.org_by_date_var).pack(side=tk.LEFT, padx=10)
        
        self.org_dry_run_var = tk.BooleanVar()
        ttk.Checkbutton(options_frame, text="Пробный запуск", variable=self.org_dry_run_var).pack(side=tk.LEFT, padx=10)
        
        self.org_backup_var = tk.BooleanVar()
        ttk.Checkbutton(options_frame, text="Создать резервные копии", variable=self.org_backup_var).pack(side=tk.LEFT, padx=10)

        action_frame = ttk.Frame(main_container)
        action_frame.pack(fill=tk.X, pady=10)
        ttk.Button(action_frame, text="🔄 Запустить организацию", command=self.run_org, style="Success.TButton").pack(side=tk.LEFT, padx=5)
        self.org_progress = ttk.Progressbar(action_frame, mode='determinate')
        self.org_progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        console_frame = ttk.Frame(main_container)
        console_frame.pack(fill=tk.BOTH, expand=True)
        self.org_console = self.create_console(console_frame)

    def setup_search_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="🔍 Поиск")

        main_container = ttk.Frame(frame)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(main_container, text="Расширенный поиск файлов", style="Title.TLabel").pack(anchor=tk.W, pady=(0, 20))

        search_frame = ttk.LabelFrame(main_container, text="Параметры поиска")
        search_frame.pack(fill=tk.X, pady=10)

        dir_frame = ttk.Frame(search_frame)
        dir_frame.pack(fill=tk.X, pady=5)
        ttk.Label(dir_frame, text="Директория:").pack(side=tk.LEFT, padx=5)
        self.search_dir_var = tk.StringVar(value=str(Path.cwd()))
        ttk.Entry(dir_frame, textvariable=self.search_dir_var, width=60).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(dir_frame, text="Обзор", command=lambda: self.browse_dir_or_file(self.search_dir_var)).pack(side=tk.LEFT, padx=5)

        name_frame = ttk.Frame(search_frame)
        name_frame.pack(fill=tk.X, pady=5)
        ttk.Label(name_frame, text="Имя файла (шаблоны):").pack(side=tk.LEFT, padx=5)
        self.search_name_var = tk.StringVar()
        ttk.Entry(name_frame, textvariable=self.search_name_var, width=60).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        content_frame = ttk.Frame(search_frame)
        content_frame.pack(fill=tk.X, pady=5)
        ttk.Label(content_frame, text="Содержимое:").pack(side=tk.LEFT, padx=5)
        self.search_content_var = tk.StringVar()
        ttk.Entry(content_frame, textvariable=self.search_content_var, width=60).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        options_frame = ttk.Frame(search_frame)
        options_frame.pack(fill=tk.X, pady=5)
        
        self.search_case_var = tk.BooleanVar()
        ttk.Checkbutton(options_frame, text="Чувствительность к регистру", variable=self.search_case_var).pack(side=tk.LEFT, padx=10)
        
        self.search_regex_var = tk.BooleanVar()
        ttk.Checkbutton(options_frame, text="Регулярные выражения", variable=self.search_regex_var).pack(side=tk.LEFT, padx=10)
        
        self.search_recursive_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Рекурсивный поиск", variable=self.search_recursive_var).pack(side=tk.LEFT, padx=10)

        action_frame = ttk.Frame(main_container)
        action_frame.pack(fill=tk.X, pady=10)
        ttk.Button(action_frame, text="🔍 Начать поиск", command=self.run_search, style="Success.TButton").pack(side=tk.LEFT, padx=5)
        self.search_progress = ttk.Progressbar(action_frame, mode='determinate')
        self.search_progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        results_frame = ttk.LabelFrame(main_container, text="Результаты поиска")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.search_results_tree = ttk.Treeview(
            results_frame, 
            columns=("name", "path", "size", "modified"), 
            show="headings",
            height=10
        )
        
        self.search_results_tree.heading("name", text="Имя файла")
        self.search_results_tree.heading("path", text="Путь")
        self.search_results_tree.heading("size", text="Размер")
        self.search_results_tree.heading("modified", text="Изменён")
        
        self.search_results_tree.column("name", width=200)
        self.search_results_tree.column("path", width=400)
        self.search_results_tree.column("size", width=100)
        self.search_results_tree.column("modified", width=150)

        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.search_results_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.search_results_tree.configure(yscrollcommand=scrollbar.set)

        self.search_results_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.search_results_tree.bind('<Double-1>', lambda e: self.open_search_result())

    def setup_tools_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="🛠️ Инструменты")

        main_container = ttk.Frame(frame)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(main_container, text="Дополнительные инструменты", style="Title.TLabel").pack(anchor=tk.W, pady=(0, 20))

        dir_frame = ttk.LabelFrame(main_container, text="Рабочая директория")
        dir_frame.pack(fill=tk.X, pady=10)
        
        tools_dir_row = ttk.Frame(dir_frame)
        tools_dir_row.pack(fill=tk.X, pady=5)
        ttk.Label(tools_dir_row, text="Директория:").pack(side=tk.LEFT, padx=5)
        self.tools_dir_var = tk.StringVar(value=str(Path.cwd()))
        ttk.Entry(tools_dir_row, textvariable=self.tools_dir_var, width=60).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(tools_dir_row, text="Обзор", command=lambda: self.browse_dir_or_file(self.tools_dir_var)).pack(side=tk.LEFT, padx=5)

        tools_frame = ttk.Frame(main_container)
        tools_frame.pack(fill=tk.BOTH, expand=True)

        col1 = ttk.Frame(tools_frame)
        col1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        dup_frame = ttk.LabelFrame(col1, text="🔍 Поиск дубликатов")
        dup_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(dup_frame, text="Найти дубликаты", command=self.run_dup_find,
                style="Success.TButton").pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(dup_frame, text="Удалить дубликаты", command=self.run_dup_delete,
                style="Danger.TButton").pack(fill=tk.X, padx=5, pady=5)

        clean_frame = ttk.LabelFrame(col1, text="🧹 Очистка")
        clean_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(clean_frame, text="Очистить пустые папки", command=self.run_clean).pack(fill=tk.X, padx=5, pady=5)

        col2 = ttk.Frame(tools_frame)
        col2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        archive_frame = ttk.LabelFrame(col2, text="📦 Архивы")
        archive_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(archive_frame, text="Создать архивы из папок", command=self.run_create_archives).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(archive_frame, text="Извлечь все архивы", command=self.run_extract_all).pack(fill=tk.X, padx=5, pady=2)

        rename_frame = ttk.LabelFrame(col2, text="✏️ Переименование")
        rename_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(rename_frame, text="Пакетное переименование", command=self.run_batch_rename).pack(fill=tk.X, padx=5, pady=5)

        col3 = ttk.Frame(tools_frame)
        col3.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        sync_frame = ttk.LabelFrame(col3, text="🔄 Синхронизация")
        sync_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(sync_frame, text="Синхронизировать папки", command=self.run_sync).pack(fill=tk.X, padx=5, pady=5)

        stats_frame = ttk.LabelFrame(col3, text="📊 Статистика")
        stats_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(stats_frame, text="Получить статистику", command=self.run_stats).pack(fill=tk.X, padx=5, pady=5)

        console_frame = ttk.LabelFrame(main_container, text="Журнал операций")
        console_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        self.tools_console = self.create_console(console_frame)

    def setup_logs_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="📋 Логи")

        main_frame = ttk.Frame(frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(main_frame, text="Журнал операций", style="Title.TLabel").pack(anchor=tk.W, pady=(0, 10))

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=5)

        ttk.Button(button_frame, text="🔄 Обновить", command=self.update_logs).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="🧹 Очистить", command=self.clear_logs).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="💾 Сохранить", command=self.save_logs).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="📧 Экспорт отчёта", command=self.export_log_report).pack(side=tk.LEFT, padx=5)

        self.log_text = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, height=25, font=("Consolas", 10),
                                                bg=self.secondary_bg, fg=self.fg_color)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        self.update_logs()

    def setup_settings_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="⚙️ Настройки")
        
        main_container = ttk.Frame(frame)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(main_container, text="Настройки программы", style="Title.TLabel").pack(anchor=tk.W, pady=(0, 20))
        
        general_frame = ttk.LabelFrame(main_container, text="Основные настройки")
        general_frame.pack(fill=tk.X, pady=10)
        
        self.auto_load_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(general_frame, text="Автозагрузка конфигурации при старте", 
                    variable=self.auto_load_var).pack(anchor=tk.W, pady=2)
        
        self.auto_stats_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(general_frame, text="Автообновление статистики системы", 
                    variable=self.auto_stats_var).pack(anchor=tk.W, pady=2)
        
        self.confirm_delete_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(general_frame, text="Запрашивать подтверждение при удалении", 
                    variable=self.confirm_delete_var).pack(anchor=tk.W, pady=2)
        
        ui_frame = ttk.LabelFrame(main_container, text="Настройки интерфейса")
        ui_frame.pack(fill=tk.X, pady=10)
        
        ui_row1 = ttk.Frame(ui_frame)
        ui_row1.pack(fill=tk.X, pady=5)
        ttk.Label(ui_row1, text="Тема:").pack(side=tk.LEFT, padx=5)
        self.theme_var = tk.StringVar(value="dark")
        ttk.Combobox(ui_row1, textvariable=self.theme_var, 
                    values=["dark", "light", "blue", "green"], state="readonly", width=15).pack(side=tk.LEFT, padx=5)
        
        ui_row2 = ttk.Frame(ui_frame)
        ui_row2.pack(fill=tk.X, pady=5)
        ttk.Label(ui_row2, text="Размер шрифта:").pack(side=tk.LEFT, padx=5)
        self.font_size_var = tk.StringVar(value="normal")
        ttk.Combobox(ui_row2, textvariable=self.font_size_var, 
                    values=["small", "normal", "large", "x-large"], state="readonly", width=15).pack(side=tk.LEFT, padx=5)
        
        notify_frame = ttk.LabelFrame(main_container, text="Уведомления")
        notify_frame.pack(fill=tk.X, pady=10)
        
        self.notify_complete_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(notify_frame, text="Уведомлять о завершении операций", 
                    variable=self.notify_complete_var).pack(anchor=tk.W, pady=2)
        
        self.notify_errors_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(notify_frame, text="Уведомлять об ошибках", 
                    variable=self.notify_errors_var).pack(anchor=tk.W, pady=2)
        
        actions_frame = ttk.Frame(main_container)
        actions_frame.pack(fill=tk.X, pady=20)
        
        ttk.Button(actions_frame, text="💾 Применить настройки", 
                command=self.apply_settings, style="Success.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(actions_frame, text="🔄 Сбросить настройки", 
                command=self.reset_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(actions_frame, text="📁 Открыть папку конфига", 
                command=self.open_config_folder).pack(side=tk.LEFT, padx=5)
        
        self.load_current_settings()

    def setup_about_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="ℹ️ О программе")

        main_frame = ttk.Frame(frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        ttk.Label(main_frame, text="Filer v2.0", 
                 style="Title.TLabel").pack(pady=10)

        about_notebook = ttk.Notebook(main_frame)
        about_notebook.pack(fill=tk.BOTH, expand=True, pady=10)

        info_frame = ttk.Frame(about_notebook)
        about_notebook.add(info_frame, text="Основная информация")

        info_text = scrolledtext.ScrolledText(
            info_frame, 
            wrap=tk.WORD, 
            height=15, 
            font=("Consolas", 10),
            bg=self.secondary_bg, 
            fg=self.fg_color,
            state=tk.DISABLED
        )
        info_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        sys_frame = ttk.Frame(about_notebook)
        about_notebook.add(sys_frame, text="Системная информация")

        sys_text = scrolledtext.ScrolledText(
            sys_frame, 
            wrap=tk.WORD, 
            height=15, 
            font=("Consolas", 10),
            bg=self.secondary_bg, 
            fg=self.fg_color,
            state=tk.DISABLED
        )
        sys_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        load_frame = ttk.Frame(about_notebook)
        about_notebook.add(load_frame, text="Производительность")

        self.load_text = scrolledtext.ScrolledText(
            load_frame, 
            wrap=tk.WORD, 
            height=15, 
            font=("Consolas", 10),
            bg=self.secondary_bg, 
            fg=self.fg_color,
            state=tk.DISABLED
        )
        self.load_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.update_about_info(info_text, sys_text, self.load_text)
        self.start_performance_monitor()

    def start_performance_monitor(self):
        def update_performance():
            performance_info = self.get_performance_info()
            self.load_text.config(state=tk.NORMAL)
            self.load_text.delete(1.0, tk.END)
            self.load_text.insert(tk.END, performance_info)
            self.load_text.config(state=tk.DISABLED)
            self.root.after(1000, update_performance)
        
        update_performance()

    def get_system_info(self):
        info = []
        
        info.append("Информация о Python")
        info.append(f"Версия Python: {sys.version}")
        info.append(f"Реализация: {platform.python_implementation()}")
        info.append(f"Компилятор: {platform.python_compiler()}")
        info.append(f"Сборка: {platform.python_build()}")
        info.append(f"Путь к Python: {sys.executable}")
        info.append(f"Кодировка по умолчанию: {sys.getdefaultencoding()}")
        info.append(f"Файловая системная кодировка: {sys.getfilesystemencoding()}")
        info.append("")
        
        info.append("Информация о системе")
        info.append(f"ОС: {platform.system()} {platform.release()}")
        info.append(f"Версия ОС: {platform.version()}")
        info.append(f"Архитектура: {platform.architecture()[0]}")
        info.append(f"Процессор: {platform.processor()}")
        info.append(f"Имя машины: {platform.node()}")
        info.append(f"Платформа: {platform.platform()}")
        info.append("")
        
        info.append("Информация о сборке")
        if hasattr(sys, 'getwindowsversion'):
            win_ver = sys.getwindowsversion()
            info.append(f"Версия Windows: {win_ver.major}.{win_ver.minor}.{win_ver.build}")
            info.append(f"Пакет обновления: {win_ver.service_pack}")
        info.append("")
        
        info.append("Пути")
        info.append(f"Текущая рабочая директория: {os.getcwd()}")
        info.append(f"Директория скрипта: {os.path.dirname(os.path.abspath(__file__))}")
        info.append(f"Путь к исполняемому файлу: {sys.executable}")
        info.append("")
        
        info.append("Окружение")
        info.append(f"Количество CPU: {os.cpu_count()}")
        info.append(f"Кодировка stdout: {sys.stdout.encoding}")
        info.append(f"Кодировка stderr: {sys.stderr.encoding}")
        
        return "\n".join(info)

    def get_performance_info(self):
        left_col = []
        right_col = []
        
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            left_col.append("Загрузка CPU")
            left_col.append(f"Загрузка: {cpu_percent}%")
            left_col.append(f"Ядра: {cpu_count}")
            if cpu_freq:
                left_col.append(f"Текущая: {cpu_freq.current:.0f} MHz")
                left_col.append(f"Максимальная: {cpu_freq.max:.0f} MHz")
            left_col.append("")
            
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            left_col.append("Оперативная память")
            left_col.append(f"Всего: {self.organizer._format_size(memory.total)}")
            left_col.append(f"Доступно: {self.organizer._format_size(memory.available)}")
            left_col.append(f"Использовано: {self.organizer._format_size(memory.used)}")
            left_col.append(f"Процент: {memory.percent}%")
            left_col.append("")
            
            left_col.append("SWAP память")
            left_col.append(f"Всего: {self.organizer._format_size(swap.total)}")
            left_col.append(f"Использовано: {self.organizer._format_size(swap.used)}")
            left_col.append(f"Процент: {swap.percent}%")
            left_col.append("")
            
            left_col.append("Система")
            boot_time = datetime.fromtimestamp(psutil.boot_time())
            left_col.append(f"Загрузка: {boot_time.strftime('%d.%m %H:%M')}")
            uptime = datetime.now() - boot_time
            days = uptime.days
            hours = uptime.seconds // 3600
            minutes = (uptime.seconds % 3600) // 60
            left_col.append(f"Время работы: {days}д {hours}ч {minutes}м")
            
            right_col.append("Дисковое пространство")
            partitions = psutil.disk_partitions()
            
            for partition in partitions:
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    device_name = partition.device.replace('\\', '').replace('/', '')
                    right_col.append(f"Диск {device_name}:")
                    right_col.append(f"  {partition.mountpoint}")
                    right_col.append(f"  {self.organizer._format_size(usage.used)} / {self.organizer._format_size(usage.total)}")
                    right_col.append(f"  Свободно: {usage.percent}%")
                    right_col.append("")
                except (PermissionError, OSError):
                    continue
            
        except Exception as e:
            left_col.append(f"Ошибка: {e}")
        
        return self._format_columns(left_col, right_col)

    def _format_columns(self, left_col, right_col):
        """Форматирует две колонки с выравниванием"""
        max_left_width = max(len(line) for line in left_col) if left_col else 0
        col_spacing = 4
        
        result = []
        max_lines = max(len(left_col), len(right_col))
        
        for i in range(max_lines):
            left_line = left_col[i] if i < len(left_col) else ""
            right_line = right_col[i] if i < len(right_col) else ""
            
            formatted_line = f"{left_line:<{max_left_width + col_spacing}}{right_line}"
            result.append(formatted_line)
        
        return "\n".join(result)

    def update_about_info(self, info_text, sys_text, load_text):
        basic_info = """GitHub: https://github.com/QUIK1001/Event-Horizon
Telegram: https://t.me/Event_Horizon_Shell

Filer v2.0 

Внимание: Будьте осторожны при удалении файлов!
Все операции выполняются на ваш страх и риск.
"""
        
        system_info = self.get_system_info()
        
        performance_info = self.get_performance_info()
        
        for text_widget, content in [
            (info_text, basic_info),
            (sys_text, system_info),
            (load_text, performance_info)
        ]:
            text_widget.config(state=tk.NORMAL)
            text_widget.delete(1.0, tk.END)
            text_widget.insert(tk.END, content)
            text_widget.config(state=tk.DISABLED)

    def setup_status_bar(self):
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_var = tk.StringVar(value="Готов")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, 
                               relief=tk.SUNKEN, anchor=tk.W)
        status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=2)
        
        sys_info_frame = ttk.Frame(status_frame)
        sys_info_frame.pack(side=tk.RIGHT, padx=5)
        
        self.sys_info_var = tk.StringVar(value="Загрузка...")
        sys_info_label = ttk.Label(sys_info_frame, textvariable=self.sys_info_var, font=("Segoe UI", 8))
        sys_info_label.pack(side=tk.RIGHT)
        
        self.update_system_info()

    def update_system_info(self):
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used = self.organizer._format_size(memory.used)
            memory_total = self.organizer._format_size(memory.total)
            
            disk_usage = self.organizer.get_disk_usage(Path.cwd())
            disk_free = self.organizer._format_size(disk_usage['free'])
            disk_total = self.organizer._format_size(disk_usage['total'])
            disk_percent = disk_usage['percent']
            
            sys_info = f"CPU: {cpu_percent:.1f}% | RAM: {memory_percent:.1f}% ({memory_used}/{memory_total}) | Диск: {disk_percent:.1f}% ({disk_free} свободно)"
            self.sys_info_var.set(sys_info)
            
        except Exception as e:
            self.sys_info_var.set("Системная информация недоступна")
        
        if self.auto_stats_var.get():
            self.root.after(2000, self.update_system_info)

    def fm_parent(self):
        current_path = Path(self.fm_dir_var.get())
        parent_path = current_path.parent
        if parent_path.exists() and str(parent_path) != str(current_path):
            self.fm_dir_var.set(str(parent_path))
            self.fm_add_to_history(str(parent_path))
            self.fm_refresh()
            self.log_to_console(self.fm_console, f"Переход в родительскую папку: {parent_path}")

    def fm_back(self):
        if self.fm_history_index > 0:
            self.fm_history_index -= 1
            self.fm_dir_var.set(self.fm_history[self.fm_history_index])
            self.fm_refresh()
            self.log_to_console(self.fm_console, f"Назад: {self.fm_dir_var.get()}")

    def fm_forward(self):
        if self.fm_history_index < len(self.fm_history) - 1:
            self.fm_history_index += 1
            self.fm_dir_var.set(self.fm_history[self.fm_history_index])
            self.fm_refresh()
            self.log_to_console(self.fm_console, f"Вперёд: {self.fm_dir_var.get()}")

    def fm_home(self):
        home_dir = str(Path.home())
        self.fm_dir_var.set(home_dir)
        self.fm_add_to_history(home_dir)
        self.fm_refresh()
        self.log_to_console(self.fm_console, f"Home: {home_dir}")

    def fm_root(self):
        root_dir = str(Path(self.fm_dir_var.get()).root)
        self.fm_dir_var.set(root_dir)
        self.fm_add_to_history(root_dir)
        self.fm_refresh()
        self.log_to_console(self.fm_console, f"Корень: {root_dir}")

    def fm_add_to_history(self, path):
        if not self.fm_history or self.fm_history[-1] != path:
            self.fm_history.append(path)
            self.fm_history_index = len(self.fm_history) - 1
            if len(self.fm_history) > 50:
                self.fm_history.pop(0)
                self.fm_history_index -= 1

    def fm_browse(self):
        directory = filedialog.askdirectory(initialdir=self.fm_dir_var.get())
        if directory:
            self.fm_dir_var.set(directory)
            self.fm_add_to_history(directory)
            self.fm_refresh()
            self.log_to_console(self.fm_console, f"Переход в директорию: {directory}")

    def sort_treeview(self, column):
        items = [(self.fm_tree.set(item, column), item) for item in self.fm_tree.get_children('')]
        
        try:
            if column == "size":
                def parse_size(size_str):
                    if size_str == "": return 0
                    units = {'B': 1, 'KB': 1024, 'MB': 1024**2, 'GB': 1024**3, 'TB': 1024**4}
                    for unit, multiplier in units.items():
                        if size_str.endswith(unit):
                            num = float(size_str[:-len(unit)].strip())
                            return num * multiplier
                    return 0
                
                items.sort(key=lambda x: parse_size(x[0]))
            elif column == "modified":
                items.sort(key=lambda x: x[0], reverse=True)
            else:
                items.sort(key=lambda x: x[0].lower())
        except:
            items.sort(key=lambda x: x[0].lower())
        
        for index, (_, item) in enumerate(items):
            self.fm_tree.move(item, '', index)

    def fm_refresh(self):
        for item in self.fm_tree.get_children():
            self.fm_tree.delete(item)

        path = Path(self.fm_dir_var.get())
        if not path.exists():
            self.log_to_console(self.fm_console, f"ОШИБКА: Директория не существует: {path}")
            return

        try:
            dirs = []
            files = []
            
            for item in path.iterdir():
                try:
                    if item.is_dir():
                        dirs.append(item)
                    elif item.is_file():
                        files.append(item)
                except (PermissionError, OSError) as e:
                    continue

            sort_by = self.sort_var.get()
            if sort_by == "name":
                dirs.sort(key=lambda x: x.name.lower())
                files.sort(key=lambda x: x.name.lower())
            elif sort_by == "size":
                dirs.sort(key=lambda x: x.stat().st_size if x.is_file() else 0)
                files.sort(key=lambda x: x.stat().st_size)
            elif sort_by == "type":
                dirs.sort(key=lambda x: x.name.lower())
                files.sort(key=lambda x: x.suffix.lower())
            elif sort_by == "modified":
                dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            for directory in dirs:
                try:
                    size = directory.stat().st_size
                    mtime = datetime.fromtimestamp(directory.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                    permissions = self.get_permissions(directory)
                    full_path = str(directory.resolve())
                    
                    self.fm_tree.insert(
                        "", 
                        tk.END, 
                        values=(f"📁 {directory.name}", self.organizer._format_size(size), "Папка", mtime, permissions, full_path),
                        tags=('directory',)
                    )
                except (OSError, PermissionError) as e:
                    continue

            for file in files:
                try:
                    size = file.stat().st_size
                    mtime = datetime.fromtimestamp(file.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                    permissions = self.get_permissions(file)
                    file_icon = self.get_file_icon(file.suffix.lower())
                    full_path = str(file.resolve())
                    
                    self.fm_tree.insert(
                        "", 
                        tk.END, 
                        values=(f"{file_icon} {file.name}", self.organizer._format_size(size), file.suffix or "Файл", mtime, permissions, full_path),
                        tags=('file',)
                    )
                except (OSError, PermissionError) as e:
                    continue
            
            self.fm_tree.tag_configure('directory', foreground='#3498db')
            self.fm_tree.tag_configure('file', foreground=self.fg_color)
                    
        except OSError as e:
            self.log_to_console(self.fm_console, f"ОШИБКА чтения директории: {e}")

    def get_permissions(self, path):
        try:
            if IS_WINDOWS:
                return "rw-rw-rw-" if os.access(path, os.W_OK) else "r--r--r--"
            else:
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
            '.pdf': '📄', '.doc': '📄', '.docx': '📄', '.txt': '📄', '.rtf': '📄',
            '.jpg': '🖼️', '.jpeg': '🖼️', '.png': '🖼️', '.gif': '🖼️', '.bmp': '🖼️', '.svg': '🖼️',
            '.mp3': '🎵', '.wav': '🎵', '.flac': '🎵', '.aac': '🎵', '.ogg': '🎵',
            '.mp4': '🎬', '.avi': '🎬', '.mkv': '🎬', '.mov': '🎬', '.wmv': '🎬',
            '.zip': '📦', '.rar': '📦', '.7z': '📦', '.tar': '📦', '.gz': '📦',
            '.exe': '⚙️', '.msi': '⚙️', '.dmg': '⚙️',
            '.py': '🐍', '.js': '📜', '.html': '🌐', '.css': '🎨', '.java': '☕',
            '.xls': '📊', '.xlsx': '📊', '.csv': '📊'
        }
        return icons.get(extension, '📄')

    def show_tree_context_menu(self, event):
        try:
            self.tree_context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.tree_context_menu.grab_release()

    def fm_update_info(self):
        selected = self.fm_tree.selection()
        if not selected:
            self.quick_info_text.config(state=tk.NORMAL)
            self.quick_info_text.delete(1.0, tk.END)
            self.quick_info_text.insert(tk.END, "Выберите файл или папку для просмотра информации")
            self.quick_info_text.config(state=tk.DISABLED)
            
            self.preview_text.config(state=tk.NORMAL)
            self.preview_text.delete(1.0, tk.END)
            self.preview_text.config(state=tk.DISABLED)
            return

        total_size = 0
        file_count = 0
        dir_count = 0
        
        for item_id in selected:
            values = self.fm_tree.item(item_id, 'values')
            path_str = values[5] if len(values) > 5 else ""
            try:
                path = Path(path_str)
                if path.is_file():
                    file_count += 1
                    total_size += path.stat().st_size
                else:
                    dir_count += 1
            except:
                pass

        info_text = f"Выбрано: {len(selected)} элементов\n"
        info_text += f"Файлов: {file_count}, Папок: {dir_count}\n"
        info_text += f"Общий размер: {self.organizer._format_size(total_size)}\n"
        
        if len(selected) == 1:
            values = self.fm_tree.item(selected[0], 'values')
            path_str = values[5]
            path = Path(path_str)
            try:
                stat = path.stat()
                info_text += f"Создан: {datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M')}\n"
                info_text += f"Изменён: {datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')}\n"
                info_text += f"Права: {self.get_permissions(path)}"
            except:
                pass

        self.quick_info_text.config(state=tk.NORMAL)
        self.quick_info_text.delete(1.0, tk.END)
        self.quick_info_text.insert(tk.END, info_text)
        self.quick_info_text.config(state=tk.DISABLED)

        if len(selected) == 1:
            values = self.fm_tree.item(selected[0], 'values')
            path_str = values[5]
            path = Path(path_str)
            
            if path.is_file():
                self.preview_file_content(path)
            else:
                self.preview_text.config(state=tk.NORMAL)
                self.preview_text.delete(1.0, tk.END)
                self.preview_text.insert(tk.END, f"Папка: {path.name}\n\nСодержит файлы и подпапки")
                self.preview_text.config(state=tk.DISABLED)
        else:
            self.preview_text.config(state=tk.NORMAL)
            self.preview_text.delete(1.0, tk.END)
            self.preview_text.insert(tk.END, f"Выбрано {len(selected)} элементов")
            self.preview_text.config(state=tk.DISABLED)

    def preview_file_content(self, file_path):
        self.preview_text.config(state=tk.NORMAL)
        self.preview_text.delete(1.0, tk.END)
        
        try:
            if file_path.stat().st_size > 1000000:
                self.preview_text.insert(tk.END, f"Файл слишком большой для предпросмотра ({self.organizer._format_size(file_path.stat().st_size)})")
                return
            
            image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
            if file_path.suffix.lower() in image_extensions:
                try:
                    img = Image.open(file_path)
                    img.thumbnail((300, 300))
                    photo = ImageTk.PhotoImage(img)
                    
                    preview_window = tk.Toplevel(self.root)
                    preview_window.title(f"Предпросмотр: {file_path.name}")
                    preview_window.geometry("400x400")
                    
                    label = tk.Label(preview_window, image=photo)
                    label.image = photo
                    label.pack(padx=10, pady=10)
                    
                    info_label = tk.Label(preview_window, 
                                        text=f"Размер: {img.size[0]}x{img.size[1]}\n"
                                             f"Формат: {img.format}\n"
                                             f"Размер файла: {self.organizer._format_size(file_path.stat().st_size)}")
                    info_label.pack(pady=5)
                    
                    return
                except Exception as e:
                    self.preview_text.insert(tk.END, f"Ошибка открытия изображения: {e}")
                    return
            
            if self.organizer._is_binary_file(file_path):
                self.preview_text.insert(tk.END, f"Бинарный файл: {file_path.name}\n\nПредпросмотр недоступен для бинарных файлов")
                return
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(5000)
                self.preview_text.insert(tk.END, content)
                if len(content) == 5000:
                    self.preview_text.insert(tk.END, "\n\n... (файл обрезан для предпросмотра)")
                    
        except Exception as e:
            self.preview_text.insert(tk.END, f"Ошибка чтения файла: {e}")
        
        self.preview_text.config(state=tk.DISABLED)

    def fm_open_selected(self):
        selected = self.fm_tree.selection()
        if selected:
            for item_id in selected:
                values = self.fm_tree.item(item_id, 'values')
                path_str = values[5] if len(values) > 5 else ""
                try:
                    path = Path(path_str)
                    if not path.exists():
                        messagebox.showerror("Ошибка", f"Файл/папка не существует: {path_str}")
                        return
                        
                    if path.is_file():
                        try:
                            if IS_WINDOWS:
                                os.startfile(path)
                            elif IS_MAC:
                                subprocess.run(['open', path])
                            else:
                                subprocess.run(['xdg-open', path])
                            self.log_message(f"Открыт файл: {path}")
                        except Exception as e:
                            messagebox.showerror("Ошибка", f"Не удалось открыть файл: {e}")
                    else:
                        self.fm_dir_var.set(path_str)
                        self.fm_add_to_history(path_str)
                        self.fm_refresh()
                        self.log_message(f"Открыта папка: {path}")
                        break
                except Exception as e:
                    messagebox.showerror("Ошибка", f"Неверный путь: {path_str}\nОшибка: {e}")

    def fm_open_in_explorer(self):
        selected = self.fm_tree.selection()
        if selected:
            item_id = selected[0]
            values = self.fm_tree.item(item_id, 'values')
            path_str = values[5] if len(values) > 5 else ""
            try:
                path = Path(path_str)
                if path.exists():
                    if IS_WINDOWS:
                        subprocess.run(['explorer', '/select,', str(path)])
                    elif IS_MAC:
                        subprocess.run(['open', '-R', str(path)])
                    else:
                        subprocess.run(['xdg-open', str(path.parent)])
                    self.log_message(f"Открыто в проводнике: {path}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось открыть в проводнике: {e}")

    def fm_delete_selected(self):
        selected = self.fm_tree.selection()
        if selected:
            if not self.confirm_delete_var.get() or messagebox.askyesno(
                "Удалить", 
                f"Удалить {len(selected)} выбранных элементов?\nЭто действие нельзя отменить!", 
                icon='warning'
            ):
                deleted_count = 0
                for item_id in selected:
                    values = self.fm_tree.item(item_id, 'values')
                    path_str = values[5] if len(values) > 5 else ""
                    try:
                        path = Path(path_str)
                        if not path.exists():
                            continue
                            
                        try:
                            if path.is_file():
                                path.unlink()
                                self.log_message(f"УДАЛЕН ФАЙЛ: {path}")
                            else:
                                shutil.rmtree(path)
                                self.log_message(f"УДАЛЕНА ПАПКА: {path}")
                            deleted_count += 1
                        except OSError as e:
                            messagebox.showerror("Ошибка", f"Не удалось удалить {path.name}: {e}")
                    except Exception as e:
                        messagebox.showerror("Ошибка", f"Неверный путь: {path_str}\nОшибка: {e}")
                
                self.fm_refresh()
                if self.notify_complete_var.get():
                    messagebox.showinfo("Успех", f"Удалено элементов: {deleted_count}")

    def fm_rename_selected(self):
        selected = self.fm_tree.selection()
        if selected and len(selected) == 1:
            item_id = selected[0]
            values = self.fm_tree.item(item_id, 'values')
            path_str = values[5] if len(values) > 5 else ""
            try:
                path = Path(path_str)
                if not path.exists():
                    messagebox.showerror("Ошибка", f"Файл/папка не существует: {path_str}")
                    return
                    
                new_name = simpledialog.askstring("Переименовать", "Введите новое имя:", initialvalue=path.name)
                if new_name and new_name != path.name:
                    try:
                        new_path = path.parent / new_name
                        path.rename(new_path)
                        self.log_message(f"ПЕРЕИМЕНОВАНО: {path.name} -> {new_name}")
                        self.fm_refresh()
                    except OSError as e:
                        messagebox.showerror("Ошибка", f"Не удалось переименовать: {e}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Неверный путь: {path_str}\nОшибка: {e}")
        elif len(selected) > 1:
            messagebox.showinfo("Информация", "Для переименования выберите только один элемент")

    def fm_create_dir(self):
        dir_name = simpledialog.askstring("Создать папку", "Введите имя новой папки:")
        if dir_name:
            try:
                new_dir = Path(self.fm_dir_var.get()) / dir_name
                new_dir.mkdir(exist_ok=True)
                self.log_message(f"СОЗДАНА ПАПКА: {new_dir}")
                self.fm_refresh()
            except OSError as e:
                messagebox.showerror("Ошибка", f"Не удалось создать папку: {e}")

    def fm_create_file(self):
        file_name = simpledialog.askstring("Создать файл", "Введите имя нового файла:")
        if file_name:
            try:
                new_file = Path(self.fm_dir_var.get()) / file_name
                new_file.touch()
                self.log_message(f"СОЗДАН ФАЙЛ: {new_file}")
                self.fm_refresh()
            except OSError as e:
                messagebox.showerror("Ошибка", f"Не удалось создать файл: {e}")

    def fm_copy_selected(self):
        selected = self.fm_tree.selection()
        if selected:
            self.clipboard_files = []
            for item_id in selected:
                values = self.fm_tree.item(item_id, 'values')
                path_str = values[5] if len(values) > 5 else ""
                path = Path(path_str)
                if path.exists():
                    self.clipboard_files.append(path)
            
            self.clipboard_operation = 'copy'
            self.log_message(f"Скопировано в буфер: {len(self.clipboard_files)} элементов")
            if self.notify_complete_var.get():
                messagebox.showinfo("Успех", f"Скопировано в буфер: {len(self.clipboard_files)} элементов")

    def fm_cut_selected(self):
        selected = self.fm_tree.selection()
        if selected:
            self.clipboard_files = []
            for item_id in selected:
                values = self.fm_tree.item(item_id, 'values')
                path_str = values[5] if len(values) > 5 else ""
                path = Path(path_str)
                if path.exists():
                    self.clipboard_files.append(path)
            
            self.clipboard_operation = 'move'
            self.log_message(f"Вырезано в буфер: {len(self.clipboard_files)} элементов")
            if self.notify_complete_var.get():
                messagebox.showinfo("Успех", f"Вырезано в буфер: {len(self.clipboard_files)} элементов")

    def fm_paste_selected(self):
        if self.clipboard_files:
            destination = Path(self.fm_dir_var.get())
            try:
                if self.clipboard_operation == 'copy':
                    copied_count = self.organizer.copy_files(self.clipboard_files, destination)
                    self.log_message(f"СКОПИРОВАНО: {copied_count} элементов -> {destination}")
                    operation_text = "скопировано"
                else:
                    moved_count = self.organizer.move_files(self.clipboard_files, destination)
                    self.log_message(f"ПЕРЕМЕЩЕНО: {moved_count} элементов -> {destination}")
                    operation_text = "перемещено"
                    self.clipboard_files = []
                
                self.fm_refresh()
                if self.notify_complete_var.get():
                    messagebox.showinfo("Успех", f"{operation_text.capitalize()} элементов: {copied_count if self.clipboard_operation == 'copy' else moved_count}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось вставить файлы: {e}")

    def fm_move_selected(self):
        selected = self.fm_tree.selection()
        if selected:
            destination = filedialog.askdirectory(initialdir=self.fm_dir_var.get())
            if destination:
                files_to_move = []
                for item_id in selected:
                    values = self.fm_tree.item(item_id, 'values')
                    path_str = values[5] if len(values) > 5 else ""
                    path = Path(path_str)
                    if path.exists():
                        files_to_move.append(path)
                
                if files_to_move:
                    try:
                        moved_count = self.organizer.move_files(files_to_move, destination)
                        self.log_message(f"ПЕРЕМЕЩЕНО: {moved_count} элементов -> {destination}")
                        self.fm_refresh()
                        if self.notify_complete_var.get():
                            messagebox.showinfo("Успех", f"Перемещено элементов: {moved_count}")
                    except Exception as e:
                        messagebox.showerror("Ошибка", f"Не удалось переместить файлы: {e}")

    def fm_archive_selected(self):
        selected = self.fm_tree.selection()
        if selected:
            files_to_archive = []
            for item_id in selected:
                values = self.fm_tree.item(item_id, 'values')
                path_str = values[5] if len(values) > 5 else ""
                path = Path(path_str)
                if path.exists():
                    files_to_archive.append(path)
            
            if files_to_archive:
                archive_name = simpledialog.askstring("Архивировать", "Введите имя архива:", 
                                                    initialvalue=f"archive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")
                if archive_name:
                    archive_path = Path(self.fm_dir_var.get()) / archive_name
                    try:
                        success = self.organizer.create_archive_from_files(files_to_archive, archive_path)
                        if success:
                            self.log_message(f"АРХИВИРОВАНО: {len(files_to_archive)} элементов -> {archive_path}")
                            self.fm_refresh()
                            if self.notify_complete_var.get():
                                messagebox.showinfo("Успех", f"Архив создан: {archive_name}")
                        else:
                            messagebox.showerror("Ошибка", "Не удалось создать архив")
                    except Exception as e:
                        messagebox.showerror("Ошибка", f"Не удалось создать архив: {e}")

    def fm_extract_selected(self):
        selected = self.fm_tree.selection()
        if selected:
            for item_id in selected:
                values = self.fm_tree.item(item_id, 'values')
                path_str = values[5] if len(values) > 5 else ""
                path = Path(path_str)
                
                if path.exists() and path.is_file() and path.suffix.lower() in ['.zip', '.rar', '.tar', '.gz', '.7z']:
                    try:
                        self.organizer.extract_archives(str(path.parent), archive_types=[path.suffix.lower()])
                        self.log_message(f"ИЗВЛЕЧЕН: {path}")
                        self.fm_refresh()
                        if self.notify_complete_var.get():
                            messagebox.showinfo("Успех", f"Архив извлечен: {path.name}")
                    except Exception as e:
                        messagebox.showerror("Ошибка", f"Не удалось извлечь архив: {e}")

    def fm_quick_search(self):
        search_term = simpledialog.askstring("Быстрый поиск", "Введите текст для поиска:")
        
        if search_term is None:
            return
        if not search_term.strip():
            messagebox.showwarning("Предупреждение", "Введите текст для поиска")
            return
        
        try:
            from pathlib import Path
            
            matches = []
            search_lower = search_term.strip().lower()
            search_dir = Path(self.fm_dir_var.get())
            
            for item_path in search_dir.rglob('*'):
                if search_lower in item_path.name.lower():
                    matches.append(item_path)
                if len(matches) >= 50:
                    break
            
            if matches:
                result = f"Найдено объектов: {len(matches)}\n\n"
                for match in matches:
                    item_type = "[ФАЙЛ]" if match.is_file() else "[ПАПКА]"
                    result += f"{item_type} {match.name}\n"
                messagebox.showinfo("Результаты поиска", result)
                self.log_message(f"Поиск '{search_term}': найдено {len(matches)} объектов")
            else:
                messagebox.showinfo("Результаты поиска", "Файлы и папки не найдены")
                self.log_message(f"Поиск '{search_term}': объекты не найдены")
                
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка поиска: {e}")

    def fm_organize_current(self):
        if messagebox.askyesno("Организовать", "Организовать файлы в текущей папке по категориям?"):
            try:
                self.organizer.org_files(
                    self.fm_dir_var.get(),
                    progress_callback=lambda current, total: self.status_var.set(f"Организация... {current}/{total}"),
                    log_callback=self.log_message
                )
                self.fm_refresh()
                if self.notify_complete_var.get():
                    messagebox.showinfo("Успех", "Файлы организованы успешно!")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка организации: {e}")

    def fm_find_duplicates(self):
        if messagebox.askyesno("Поиск дубликатов", "Найти дубликаты файлов в текущей папке и подпапках?"):
            try:
                duplicates_count = self.organizer.find_dup(
                    self.fm_dir_var.get(),
                    progress_callback=lambda current, total: self.status_var.set(f"Поиск дубликатов... {current}/{total}"),
                    log_callback=self.log_message
                )
                if self.notify_complete_var.get():
                    messagebox.showinfo("Результат", f"Найдено дубликатов: {duplicates_count}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка поиска дубликатов: {e}")

    def fm_clean_empty(self):
        if messagebox.askyesno("Очистка", "Удалить все пустые папки в текущей директории и подпапках?"):
            try:
                cleaned_count = self.organizer.clean_empty_dirs(
                    self.fm_dir_var.get(),
                    progress_callback=lambda current, total: self.status_var.set(f"Очистка... {current}/{total}"),
                    log_callback=self.log_message
                )
                self.fm_refresh()
                if self.notify_complete_var.get():
                    messagebox.showinfo("Успех", f"Очищено пустых папок: {cleaned_count}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка очистки: {e}")

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
            self.log_message(f"Статистика для {path}: {file_count} файлов, {dir_count} папок, {self.organizer._format_size(total_size)}")
        except OSError as e:
            messagebox.showerror("Ошибка", f"Не удалось собрать статистику: {e}")

    def fm_properties(self):
        selected = self.fm_tree.selection()
        if selected:
            if len(selected) == 1:
                item_id = selected[0]
                values = self.fm_tree.item(item_id, 'values')
                path_str = values[5] if len(values) > 5 else ""
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
                    info += f"Группа: {stat.st_gid}"
                    
                    messagebox.showinfo("Свойства", info)
                except OSError as e:
                    messagebox.showerror("Ошибка", f"Не удалось получить свойства: {e}")
            else:
                total_size = 0
                for item_id in selected:
                    values = self.fm_tree.item(item_id, 'values')
                    path_str = values[5] if len(values) > 5 else ""
                    path = Path(path_str)
                    try:
                        if path.is_file():
                            total_size += path.stat().st_size
                    except:
                        pass
                
                info = f"Выбрано элементов: {len(selected)}\n"
                info += f"Общий размер: {self.organizer._format_size(total_size)}"
                messagebox.showinfo("Свойства", info)

    def run_in_thread(self, func, *args, **kwargs):
        progress_callback = kwargs.pop('progress_callback', None)
        log_callback = kwargs.pop('log_callback', None)
        completion_callback = kwargs.pop('completion_callback', None)
        console_callback = kwargs.pop('console_callback', None)

        def worker():
            try:
                try:
                    root_exists = self.root.winfo_exists()
                except (tk.TclError, RuntimeError):
                    root_exists = False
                    
                if root_exists:
                    self.root.after(0, lambda: self.status_var.set("Выполняется..."))
                    
                result = func(*args, progress_callback=progress_callback, 
                            log_callback=log_callback, **kwargs)
                        
                try:
                    root_exists = self.root.winfo_exists()
                except (tk.TclError, RuntimeError):
                    root_exists = False
                    
                if root_exists:
                    self.root.after(0, lambda: self.status_var.set("Готов"))
                    if completion_callback:
                        self.root.after(0, lambda: completion_callback(result))
                    if self.notify_complete_var.get() and root_exists:
                        self.root.after(0, lambda: messagebox.showinfo("Успех", "Операция завершена"))
                        
            except Exception as e:
                error_msg = str(e)
                
                try:
                    root_exists = self.root.winfo_exists()
                except (tk.TclError, RuntimeError):
                    root_exists = False
                    
                if root_exists:
                    self.root.after(0, lambda: self.status_var.set("Ошибка"))
                    if self.notify_errors_var.get():
                        self.root.after(0, lambda: messagebox.showerror("Ошибка", error_msg))
                    self.root.after(0, lambda: self.log_message(f"ОШИБКА: {error_msg}"))
                    if console_callback:
                        self.root.after(0, lambda: self.log_to_console(console_callback, f"ОШИБКА: {error_msg}"))
                else:
                    print(f"ОШИБКА (окно закрыто): {error_msg}")

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
            backup_dir=self.org_dir_var.get() + "_backup" if self.org_backup_var.get() else None,
            progress_callback=lambda current, total: self.org_progress.config(value=(current / total) * 100) if total > 0 else None,
            log_callback=log_callback,
            console_callback=self.org_console
        )

    def run_search(self):
        self.search_progress['value'] = 0
        for item in self.search_results_tree.get_children():
            self.search_results_tree.delete(item)
        
        def log_callback(message):
            self.log_message(message)
        
        def completion_callback(results):
            for file_path in results:
                try:
                    size = file_path.stat().st_size
                    mtime = datetime.fromtimestamp(file_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                    self.search_results_tree.insert(
                        "", 
                        tk.END, 
                        values=(file_path.name, str(file_path.parent), self.organizer._format_size(size), mtime)
                    )
                except:
                    pass
            
            self.log_to_console(self.fm_console, f"Поиск завершен. Найдено: {len(results)} файлов")
        
        patterns = self.search_name_var.get().split() if self.search_name_var.get() else None
        content_search = self.search_content_var.get() if self.search_content_var.get() else None
        
        self.run_in_thread(
            self.organizer.find_files_advanced,
            self.search_dir_var.get(),
            patterns=patterns,
            content_search=content_search,
            case_sensitive=self.search_case_var.get(),
            content_regex=self.search_regex_var.get(),
            recursive=self.search_recursive_var.get(),
            progress_callback=lambda current, total: self.search_progress.config(value=(current / total) * 100) if total > 0 else None,
            log_callback=log_callback,
            completion_callback=completion_callback
        )

    def open_search_result(self):
        selected = self.search_results_tree.selection()
        if selected:
            item = selected[0]
            values = self.search_results_tree.item(item, 'values')
            file_path = Path(values[1]) / values[0]
            try:
                if file_path.exists():
                    if file_path.is_file():
                        if IS_WINDOWS:
                            os.startfile(file_path)
                        elif IS_MAC:
                            subprocess.run(['open', file_path])
                        else:
                            subprocess.run(['xdg-open', file_path])
                    else:
                        self.fm_dir_var.set(str(file_path))
                        self.fm_add_to_history(str(file_path))
                        self.fm_refresh()
                        self.notebook.select(0)
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось открыть файл: {e}")

    def run_dup_find(self):
        def completion_callback(result):
            messagebox.showinfo("Результат", f"Найдено дубликатов: {result}")
            
        self.run_in_thread(
            self.organizer.find_dup,
            self.tools_dir_var.get(),
            progress_callback=lambda current, total: self.status_var.set(f"Поиск дубликатов... {current}/{total}"),
            log_callback=self.log_message,
            completion_callback=completion_callback,
            console_callback=self.tools_console
        )

    def run_dup_delete(self):
        if messagebox.askyesno("Удаление дубликатов", "Удалить все найденные дубликаты файлов?\nЭто действие нельзя отменить!"):
            def completion_callback(result):
                messagebox.showinfo("Результат", f"Удалено дубликатов: {result}")
                self.fm_refresh()
                
            self.run_in_thread(
                self.organizer.find_dup,
                self.tools_dir_var.get(),
                delete=True,
                progress_callback=lambda current, total: self.status_var.set(f"Удаление дубликатов... {current}/{total}"),
                log_callback=self.log_message,
                completion_callback=completion_callback,
                console_callback=self.tools_console
            )

    def run_clean(self):
        def completion_callback(result):
            messagebox.showinfo("Результат", f"Очищено пустых папок: {result}")
            self.fm_refresh()
            
        self.run_in_thread(
            self.organizer.clean_empty_dirs,
            self.tools_dir_var.get(),
            progress_callback=lambda current, total: self.status_var.set(f"Очистка... {current}/{total}"),
            log_callback=self.log_message,
            completion_callback=completion_callback,
            console_callback=self.tools_console
        )

    def run_create_archives(self):
        def completion_callback(result):
            messagebox.showinfo("Результат", f"Создано архивов: {result}")
            self.fm_refresh()
            
        self.run_in_thread(
            self.organizer.create_archives,
            self.tools_dir_var.get(),
            progress_callback=lambda current, total: self.status_var.set(f"Создание архивов... {current}/{total}"),
            log_callback=self.log_message,
            completion_callback=completion_callback,
            console_callback=self.tools_console
        )

    def run_extract_all(self):
        def completion_callback(result):
            messagebox.showinfo("Результат", f"Извлечено архивов: {result}")
            self.fm_refresh()
            
        self.run_in_thread(
            self.organizer.extract_archives,
            self.tools_dir_var.get(),
            progress_callback=lambda current, total: self.status_var.set(f"Извлечение архивов... {current}/{total}"),
            log_callback=self.log_message,
            completion_callback=completion_callback,
            console_callback=self.tools_console
        )

    def run_batch_rename(self):
        pattern = simpledialog.askstring("Переименование", "Введите шаблон для поиска:")
        if pattern:
            replacement = simpledialog.askstring("Переименование", "Введите замену:")
            if replacement is not None:
                def completion_callback(result):
                    messagebox.showinfo("Результат", f"Переименовано файлов: {result}")
                    self.fm_refresh()
                    
                self.run_in_thread(
                    self.organizer.rename_files,
                    self.tools_dir_var.get(),
                    pattern,
                    replacement,
                    progress_callback=lambda current, total: self.status_var.set(f"Переименование... {current}/{total}"),
                    log_callback=self.log_message,
                    completion_callback=completion_callback,
                    console_callback=self.tools_console
                )

    def run_sync(self):
        source = filedialog.askdirectory(title="Выберите исходную папку")
        if source:
            destination = filedialog.askdirectory(title="Выберите целевую папку")
            if destination:
                if messagebox.askyesno("Синхронизация", f"Синхронизировать {source} с {destination}?"):
                    def completion_callback(result):
                        messagebox.showinfo("Результат", f"Синхронизировано файлов: {result}")
                        
                    self.run_in_thread(
                        self.organizer.sync_folders,
                        source,
                        destination,
                        progress_callback=lambda current, total: self.status_var.set(f"Синхронизация... {current}/{total}"),
                        log_callback=self.log_message,
                        completion_callback=completion_callback,
                        console_callback=self.tools_console
                    )

    def run_stats(self):
        try:
            stats = self.organizer.get_file_stats(self.tools_dir_var.get())
            
            stats_text = "Статистика файлов:\n\n"
            total_files = 0
            total_size = 0
            
            for category, data in stats.items():
                stats_text += f"{category}: {data['count']} файлов, {self.organizer._format_size(data['size'])}\n"
                total_files += data['count']
                total_size += data['size']
            
            stats_text += f"\nИтого: {total_files} файлов, {self.organizer._format_size(total_size)}"
            messagebox.showinfo("Статистика", stats_text)
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось получить статистику: {e}")

    def browse_dir_or_file(self, var):
        path = filedialog.askdirectory(initialdir=var.get())
        if path:
            var.set(path)

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

    def export_log_report(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".html",
            filetypes=[("HTML files", "*.html"), ("All files", "*.*")]
        )
        if file_path:
            try:
                html_content = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Отчет Filer - {datetime.now().strftime('%Y-%m-%d %H:%M')}</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        .log-entry {{ margin: 5px 0; padding: 5px; border-left: 3px solid #0078d4; }}
                        .timestamp {{ color: #666; }}
                        .error {{ border-left-color: #e74c3c; background: #fee; }}
                    </style>
                </head>
                <body>
                    <h1>Отчет Filer</h1>
                    <p>Сгенерирован: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <div class="logs">
                """
                
                for log in self.organizer.get_logs():
                    css_class = "error" if "ОШИБКА" in log else ""
                    html_content += f'<div class="log-entry {css_class}">{log}</div>'
                
                html_content += """
                    </div>
                </body>
                </html>
                """
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                messagebox.showinfo("Успех", f"Отчет сохранен: {file_path}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить отчет: {e}")

    def create_console(self, parent, height=8):
        console_frame = ttk.Frame(parent)
        console_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        console = scrolledtext.ScrolledText(console_frame, height=height, wrap=tk.WORD, 
                                          bg=self.secondary_bg, fg=self.fg_color, 
                                          font=("Consolas", 9), state=tk.DISABLED)
        console.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        return console

    def log_to_console(self, console, message):
        console.config(state=tk.NORMAL)
        console.insert(tk.END, f"{message}\n")
        console.see(tk.END)
        console.config(state=tk.DISABLED)

    def clear_console(self, console):
        console.config(state=tk.NORMAL)
        console.delete(1.0, tk.END)
        console.config(state=tk.DISABLED)

    def load_current_settings(self):
        try:
            config_path = Path.home() / ".filer_settings.json"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                    
                self.auto_load_var.set(settings.get('auto_load', True))
                self.auto_stats_var.set(settings.get('auto_stats', True))
                self.confirm_delete_var.set(settings.get('confirm_delete', True))
                self.theme_var.set(settings.get('theme', 'dark'))
                self.font_size_var.set(settings.get('font_size', 'normal'))
                self.chunk_size_var.set(settings.get('chunk_size', '8192'))
                self.max_threads_var.set(settings.get('max_threads', 4))
                self.notify_complete_var.set(settings.get('notify_complete', True))
                self.notify_errors_var.set(settings.get('notify_errors', True))
                self.notify_large_ops_var.set(settings.get('notify_large_ops', True))
                
        except Exception as e:
            print(f"Ошибка загрузки настроек: {e}")

    def apply_settings(self):
        try:
            settings = {
                'auto_load': self.auto_load_var.get(),
                'auto_stats': self.auto_stats_var.get(),
                'confirm_delete': self.confirm_delete_var.get(),
                'theme': self.theme_var.get(),
                'font_size': self.font_size_var.get(),
                'chunk_size': self.chunk_size_var.get(),
                'max_threads': self.max_threads_var.get(),
                'notify_complete': self.notify_complete_var.get(),
                'notify_errors': self.notify_errors_var.get(),
                'notify_large_ops': self.notify_large_ops_var.get()
            }
            
            config_path = Path.home() / ".filer_settings.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
            
            self.apply_theme(settings['theme'])
            self.apply_font_size(settings['font_size'])
            
            messagebox.showinfo("Успех", "Настройки применены и сохранены!")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить настройки: {e}")

    def apply_theme(self, theme):
        if theme == "light":
            self.bg_color = "#ffffff"
            self.fg_color = "#000000"
            self.secondary_bg = "#f0f0f0"
            self.tertiary_bg = "#e0e0e0"
            self.text_color = "#333333"
            self.border_color = "#cccccc"
        elif theme == "dark":
            self.bg_color = "#1e1e1e"
            self.fg_color = "#ffffff"
            self.secondary_bg = "#2d2d30"
            self.tertiary_bg = "#3e3e42"
            self.text_color = "#cccccc"
            self.border_color = "#444444"
        elif theme == "blue":
            self.bg_color = "#1a1a2e"
            self.fg_color = "#e6e6e6"
            self.secondary_bg = "#16213e"
            self.tertiary_bg = "#0f3460"
            self.text_color = "#b8b8b8"
            self.border_color = "#2d4059"
        elif theme == "green":
            self.bg_color = "#1a2f1a"
            self.fg_color = "#e6f2e6"
            self.secondary_bg = "#2d4a2d"
            self.tertiary_bg = "#3d5c3d"
            self.text_color = "#b8d4b8"
            self.border_color = "#4a6b4a"
        
        self.setup_styles()
        self.refresh_ui_colors()

    def refresh_ui_colors(self):
        for widget in self.root.winfo_children():
            self.update_widget_colors(widget)

    def update_widget_colors(self, widget):
        try:
            if isinstance(widget, (tk.Text, tk.ScrolledText)):
                widget.configure(bg=self.secondary_bg, fg=self.fg_color)
            elif isinstance(widget, tk.Frame):
                widget.configure(bg=self.bg_color)
        except:
            pass
        
        for child in widget.winfo_children():
            self.update_widget_colors(child)

    def apply_font_size(self, size):
        sizes = {"small": 9, "normal": 10, "large": 12, "x-large": 14}
        new_size = sizes.get(size, 10)
        
        self.style.configure("TLabel", font=("Segoe UI", new_size))
        self.style.configure("TButton", font=("Segoe UI", new_size))

    def reset_settings(self):
        if messagebox.askyesno("Сброс настроек", "Вы уверены что хотите сбросить все настройки к значениям по умолчанию?"):
            self.auto_load_var.set(True)
            self.auto_stats_var.set(True)
            self.confirm_delete_var.set(True)
            self.theme_var.set("dark")
            self.font_size_var.set("normal")
            self.chunk_size_var.set("8192")
            self.max_threads_var.set(4)
            self.notify_complete_var.set(True)
            self.notify_errors_var.set(True)
            self.notify_large_ops_var.set(True)
            
            messagebox.showinfo("Успех", "Настройки сброшены к значениям по умолчанию")

    def open_config_folder(self):
        config_dir = Path.home()
        try:
            if IS_WINDOWS:
                os.startfile(config_dir)
            elif IS_MAC:
                subprocess.run(['open', config_dir])
            else:
                subprocess.run(['xdg-open', config_dir])
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось открыть папку: {e}")

    def load_config(self):
        config_paths = [
            Path.home() / ".filer_config.json",
            Path.cwd() / "filer_config.json"
        ]
        
        for config_path in config_paths:
            if config_path.exists():
                loaded = self.organizer.load_cfg(config_path)
                if loaded:
                    self.log_message(f"Загружена конфигурация из {config_path}")
                    break

    def start_performance_monitor(self):
        def update_performance():
            try:
                self.update_system_info()
                self.root.after(5000, update_performance)
            except Exception:
                self.root.after(5000, update_performance)
        
        update_performance()


def main():
    try:
        root = tk.Tk()
        app = ModernFileOrganizerGUI(root)
        
        root.update_idletasks()
        x = (root.winfo_screenwidth() - root.winfo_reqwidth()) // 2
        y = (root.winfo_screenheight() - root.winfo_reqheight()) // 2
        root.geometry(f"+{x}+{y}")
        
        def on_closing():
            try:
                root.quit()
                root.destroy()
            except:
                pass
                
        root.protocol("WM_DELETE_WINDOW", on_closing)
        
        root.mainloop()
    except Exception as e:
        print(f"Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        if IS_WINDOWS:
            input("Нажмите Enter для выхода...")
        else:
            print("Программа завершена.")

if __name__ == "__main__":
    main()
