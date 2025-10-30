import sys
import subprocess
import os
import importlib

print(
    "Сейчас будет много ошибок. Не волнуйтесь, плагин будет работать корректно."
)


def ensure_packages():
    """Проверяет и устанавливает необходимые пакеты"""
    packages = [
        'numpy', 'scipy', 'matplotlib', 'streamlit', 'plotly', 'pandas',
        'sympy'
    ]
    missing = []
    print("Проверка зависимостей...")
    for package in packages:
        try:
            importlib.import_module(package)
            print(f"{package} - установлен")
        except ImportError:
            print(f"{package} - отсутствует")
            missing.append(package)
    if missing:
        print(f"\nУстановка недостающих пакетов: {', '.join(missing)}")
        for package in missing:
            try:
                print(f"Установка {package}...")
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", package],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL)
                print(f"{package} успешно установлен")
            except subprocess.CalledProcessError:
                print(f" Не удалось установить {package}")
                return False
    else:
        print("Все зависимости установлены!")
    return True


if not ensure_packages():
    print(
        "Не удалось установить все зависимости. Приложение может работать некорректно."
    )
    sys.exit(1)


def is_streamlit_script(file_path):
    """Проверяет, является ли файл Streamlit скриптом"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            streamlit_indicators = [
                'import streamlit', 'from streamlit', 'st.', 'streamlit.',
                'st.set_page_config', 'st.title(', 'st.write('
            ]
            return any(indicator in content
                       for indicator in streamlit_indicators)
    except:
        return False


def run_streamlit_app(script_path):
    """Запускает Streamlit приложение"""
    try:
        print(f"Запуск Streamlit приложения: {os.path.basename(script_path)}")
        print("Откройте браузер и перейдите по адресу: http://localhost:8501")
        print("Для остановки нажмите Ctrl+C в этом окне")
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", script_path,
            "--server.port", "8501", "--browser.serverAddress", "localhost"
        ])
        process.wait()
        return True
    except KeyboardInterrupt:
        print("\nОстановка приложения...")
        if 'process' in locals():
            process.terminate()
        return True
    except Exception as e:
        print(f"Ошибка запуска Streamlit: {e}")
        return False


def main():
    """Главная функция для плагинной системы"""
    current_file = os.path.abspath(__file__)
    if is_streamlit_script(current_file):
        success = run_streamlit_app(current_file)
        if not success:
            input("Press Enter to continue...")
        return


import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import optimize, integrate, linalg, fft
from scipy.interpolate import make_interp_spline
import json
from datetime import datetime
import base64
import math
import functools
import re
import cmath
from fractions import Fraction
import plotly.figure_factory as ff
from math import gcd as math_gcd
import time
import streamlit as st
import sympy as sp

plotly_template = "plotly_dark"
if 'welcome_shown' not in st.session_state:
    welcome_container = st.empty()
    with welcome_container.container():
        st.markdown("""
        <style>
        .welcome-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 400px;
            text-align: center;
            background: linear-gradient(180deg, #0a0e1a 0%, #1e3a8a 100%);
            color: white;
            padding: 2rem;
            border: 2px solid #3b82f6;
            border-radius: 20px;
            margin: 10px;
            box-shadow: 0 10px 30px rgba(59, 130, 246, 0.3); 
        }
        .welcome-title {
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 1rem;
            animation: fadeIn 1s ease-in;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            color: transparent;
        }
        .welcome-logo {
            font-size: 3rem;
            margin-bottom: 1rem;
            display: block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            color: transparent;
        }
        .welcome-subtitle {
            font-size: 1.2rem;
            margin-bottom: 1.5rem;
            animation: fadeIn 1.5s ease-in;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            color: transparent;
            opacity: 0.9;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        </style>
        <div class="welcome-container">
            <span class="welcome-logo">🧮</span>
            <div class="welcome-title">Математический калькулятор</div>
            <div class="welcome-subtitle">by quik</div>
        </div>
        """,
                    unsafe_allow_html=True)

    progress_bar = st.progress(0)
    status_text = st.empty()
    note_text = st.empty()

    modules = [
        "Загрузка математического парсера...",
        "Инициализация графического движка...",
        "Подготовка матричных операций...",
        "Загрузка символьных вычислений...",
        "Подготовка рабочего пространства..."
    ]
    for i, module in enumerate(modules):
        progress_bar.progress((i + 1) * 20)
        status_text.text(f"{module}")
        time.sleep(0.5)

    progress_bar.progress(100)
    status_text.text("Загрузка завершена.")
    time.sleep(0.5)

    welcome_container.empty()
    progress_bar.empty()
    status_text.empty()

    st.session_state.welcome_shown = True
    st.rerun()


def beta_func(a, b):
    """Бета-функция через гамма-функции"""
    return math.gamma(a) * math.gamma(b) / math.gamma(a + b)


def combination(n, k):
    """Число сочетаний C(n, k)"""
    if k > n or k < 0:
        return 0
    return math.comb(int(n), int(k))


def permutation(n, k):
    """Число перестановок P(n, k)"""
    if k > n or k < 0:
        return 0
    return math.perm(int(n), int(k))


class AdvancedMathParser:

    def __init__(self):
        self.operators = {'+', '-', '*', '/', '**', '//', '%'}
        self.functions = {
            'sin':
            math.sin,
            'cos':
            math.cos,
            'tan':
            math.tan,
            'cot':
            lambda x: 1 / math.tan(x),
            'arcsin':
            math.asin,
            'arccos':
            math.acos,
            'arctan':
            math.atan,
            'arccot':
            lambda x: math.pi / 2 - math.atan(x),
            'sinh':
            math.sinh,
            'cosh':
            math.cosh,
            'tanh':
            math.tanh,
            'arcsinh':
            math.asinh,
            'arccosh':
            math.acosh,
            'arctanh':
            math.atanh,
            'exp':
            math.exp,
            'log':
            math.log,
            'log10':
            math.log10,
            'log2':
            math.log2,
            'ln':
            math.log,
            'lg':
            math.log10,
            'lb':
            math.log2,
            'sqrt':
            math.sqrt,
            'cbrt':
            lambda x: math.copysign(abs(x)**(1 / 3), x),
            'ceil':
            math.ceil,
            'floor':
            math.floor,
            'round':
            round,
            'abs':
            abs,
            'sign':
            lambda x: 1 if x > 0 else (-1 if x < 0 else 0),
            'factorial':
            math.factorial,
            'gamma':
            math.gamma,
            'erf':
            math.erf,
            'erfc':
            math.erfc,
            'beta':
            lambda x: math.gamma(x) * math.gamma(1 - x)
            if x > 0 else float('nan'),
            'heaviside':
            lambda x: 1 if x >= 0 else 0,
            'step':
            lambda x: 1 if x >= 0 else 0,
            'C':
            combination,
            'P':
            permutation,
            'comb':
            combination,
            'perm':
            permutation,
            'real':
            lambda x: x.real,
            'imag':
            lambda x: x.imag,
            'conj':
            lambda x: x.conjugate(),
            'arg':
            lambda x: cmath.phase(x),
            're':
            lambda x: x.real,
            'im':
            lambda x: x.imag,
            'mod':
            abs,
        }
        self.constants = {
            'pi': math.pi,
            'e': math.e,
            'tau': 2 * math.pi,
            'phi': (1 + math.sqrt(5)) / 2,
            'inf': float('inf'),
            'i': 1j,
            'j': 1j,
        }
        self.cache = {}
        self.variables = {}

    def tokenize(self, expression):
        """Токенизация выражения"""
        pattern = r'(\d+\.?\d*[jJ]|\d+/\d+|\d+\.?\d*([eE][+-]?\d+)?|[a-zA-Z_][a-zA-Z0-9_]*|\*\*|//|<=|>=|==|!=|[+\-*/()^%<>,])'
        tokens = re.findall(pattern, expression)
        return [t[0] for t in tokens]

    def handle_unary_operators(self, tokens):
        """Обработка унарных операторов"""
        result = []
        for i, token in enumerate(tokens):
            if token in ['-', '+']:
                is_unary = (i == 0 or result[-1]
                            in ['(', '+', '-', '*', '/', '//', '%', '**', '^'])

                if is_unary:
                    if token == '-':
                        result.append('unary_minus')
                    else:
                        result.append('unary_plus')
                else:
                    result.append(token)
            else:
                result.append(token)
        return result

    def parse_range(self, range_str: str) -> np.ndarray:
        """Парсит диапазон вида 'start:end:step' или 'start:end'"""
        parts = range_str.strip().split(':')
        if len(parts) == 2:
            start, end = map(float, parts)
            step = 1.0
        elif len(parts) == 3:
            start, end, step = map(float, parts)
        else:
            raise ValueError(
                "Неверный формат диапазона. Используйте 'start:end' или 'start:end:step'"
            )
        if step == 0:
            raise ValueError("Шаг не может быть нулём")
        if (end - start) * step < 0:
            return np.array([])
        return np.arange(start, end, step)

    def shunting_yard(self, tokens):
        """Алгоритм сортировочной станции"""
        output = []
        stack = []
        precedence = {
            '+': 1,
            '-': 1,
            '*': 2,
            '/': 2,
            '//': 2,
            '%': 2,
            '**': 3,
            '^': 3,
            'unary_minus': 4,
            'unary_plus': 4,
        }
        right_associative = {'**', '^', 'unary_minus', 'unary_plus'}

        for token in tokens:
            if self.is_number(token) or self.is_fraction(
                    token) or token in self.constants:
                output.append(token)
            elif token in self.functions:
                stack.append(token)
            elif token == '(':
                stack.append(token)
            elif token == ')':
                while stack and stack[-1] != '(':
                    output.append(stack.pop())
                if not stack:
                    raise ValueError("Несогласованные скобки")
                stack.pop()
                if stack and stack[-1] in self.functions:
                    output.append(stack.pop())
            elif token in precedence:
                while stack and stack[-1] != '(' and stack[-1] in precedence:
                    top_op = stack[-1]
                    if (token not in right_associative
                            and precedence[top_op] >= precedence[token]):
                        output.append(stack.pop())
                    elif (token in right_associative
                          and precedence[top_op] > precedence[token]):
                        output.append(stack.pop())
                    else:
                        break
                stack.append(token)
            else:
                output.append(token)

        while stack:
            if stack[-1] in '()':
                raise ValueError("Несогласованные скобки")
            output.append(stack.pop())

        return output

    def is_number(self, token):
        """Проверка, является ли токен числом"""
        try:
            complex(token)
            return True
        except ValueError:
            return False

    def is_fraction(self, token):
        """Проверка, является ли токен дробью"""
        return re.match(r'^\d+/\d+$', token) is not None

    def evaluate_rpn(self, rpn, variables=None):
        """Вычисление RPN выражения"""
        if variables is None:
            variables = {}
        all_vars = {**self.variables, **variables}
        stack = []

        two_arg_functions = {'C', 'P', 'comb', 'perm'}

        for token in rpn:
            if token == 'unary_minus':
                if not stack:
                    raise ValueError(
                        "Недостаточно операндов для унарного минуса")
                stack.append(-stack.pop())
            elif token == 'unary_plus':
                if not stack:
                    raise ValueError(
                        "Недостаточно операндов для унарного плюса")
                pass
            elif self.is_fraction(token):
                num, den = map(int, token.split('/'))
                if den == 0:
                    raise ValueError("Деление на ноль в дроби")
                stack.append(num / den)
            elif self.is_number(token):
                stack.append(complex(token))
            elif token in self.constants:
                stack.append(complex(self.constants[token]))
            elif token in all_vars:
                stack.append(complex(all_vars[token]))
            elif token in self.functions:
                if token in two_arg_functions:
                    if len(stack) < 2:
                        raise ValueError(
                            f"Недостаточно аргументов для {token}")
                    k = stack.pop()
                    n = stack.pop()
                    try:
                        n_int = int(round(n.real))
                        k_int = int(round(k.real))
                        result = self.functions[token](n_int, k_int)
                        stack.append(complex(result))
                    except Exception as e:
                        raise ValueError(f"Ошибка в функции {token}: {e}")
                    continue

                if not stack:
                    raise ValueError(f"Недостаточно аргументов для {token}")
                arg = stack.pop()
                try:
                    complex_functions = {
                        'real', 'imag', 'conj', 'arg', 're', 'im', 'mod'
                    }
                    integer_functions = {'factorial'}

                    if token in complex_functions:
                        if token in ('real', 're'):
                            result = arg.real
                        elif token in ('imag', 'im'):
                            result = arg.imag
                        elif token == 'conj':
                            result = arg.conjugate()
                        elif token == 'arg':
                            result = cmath.phase(arg)
                        elif token == 'mod':
                            result = abs(arg)
                        result = complex(result)

                    elif token in integer_functions:
                        arg_real = arg.real
                        if abs(arg_real - round(arg_real)) > 1e-10:
                            raise ValueError(
                                f"Функция {token} требует целочисленный аргумент, получено: {arg_real}"
                            )
                        arg_int = int(round(arg_real))
                        if arg_int < 0:
                            raise ValueError(
                                f"Функция {token} не определена для отрицательных чисел"
                            )
                        result = self.functions[token](arg_int)
                        result = complex(result)

                    else:
                        arg_real = arg.real
                        result = self.functions[token](arg_real)
                        result = complex(result)

                    stack.append(result)
                except Exception as e:
                    raise ValueError(f"Ошибка в функции {token}: {e}")
            elif token in self.operators:
                if len(stack) < 2:
                    raise ValueError(f"Недостаточно операндов для {token}")
                b = stack.pop()
                a = stack.pop()

                try:
                    if token == '+':
                        stack.append(a + b)
                    elif token == '-':
                        stack.append(a - b)
                    elif token == '*':
                        stack.append(a * b)
                    elif token == '/':
                        if b == 0:
                            raise ValueError("Деление на ноль")
                        stack.append(a / b)
                    elif token == '//':
                        if b == 0:
                            raise ValueError("Целочисленное деление на ноль")
                        stack.append(a // b)
                    elif token == '%':
                        if b == 0:
                            raise ValueError("Остаток от деления на ноль")
                        stack.append(a % b)
                    elif token in ['**', '^']:
                        stack.append(a**b)
                except Exception as e:
                    raise ValueError(
                        f"Ошибка при выполнении операции {token}: {e}")
            elif token in ['i', 'j']:
                stack.append(1j)
            else:
                if token in all_vars:
                    stack.append(complex(all_vars[token]))
                else:
                    raise ValueError(f"Неизвестный токен: {token}")

        if len(stack) != 1:
            raise ValueError("Некорректное выражение")

        return stack[0]

    def evaluate_expression(self, expression, variables=None):
        """Основной метод вычисления выражения"""
        try:
            expr = expression.replace(' ', '').replace('^', '**')

            tokens = self.tokenize(expr)
            tokens = self.handle_unary_operators(tokens)

            rpn = self.shunting_yard(tokens)

            result = self.evaluate_rpn(rpn, variables)
            return result

        except Exception as e:
            raise ValueError(f"Ошибка вычисления '{expression}': {str(e)}")

    def set_variable(self, name, value):
        self.variables[name] = value

    def get_variables(self):
        return self.variables.copy()

    def clear_variables(self):
        self.variables.clear()

    def parse_cached(self, expression: str):
        if expression in self.cache:
            return self.cache[expression]
        result = self.evaluate_expression(expression)
        self.cache[expression] = result
        return result

    def compile_function(self, expression):

        def func(x):
            return self.evaluate_expression(expression, {'x': x})

        return func


class FractionEngine:

    @staticmethod
    def to_fraction(decimal: float, max_denominator: int = 10000) -> str:
        try:
            if abs(decimal - round(decimal)) < 1e-10:
                return str(int(round(decimal)))
            frac = Fraction(decimal).limit_denominator(max_denominator)
            if frac.denominator == 1:
                return str(frac.numerator)
            return f"{frac.numerator}/{frac.denominator}"
        except:
            return f"{decimal:.10g}"

    @staticmethod
    def evaluate_fraction(expression: str) -> str:
        try:
            expr_decimal = expression
            fraction_pattern = r'(\d+)/(\d+)'
            matches = re.findall(fraction_pattern, expression)
            for num, den in matches:
                decimal_val = float(num) / float(den)
                expr_decimal = expr_decimal.replace(f"{num}/{den}",
                                                    str(decimal_val))
            result = parser.evaluate_expression(expr_decimal)
            if abs(result.imag) < 1e-12:
                return FractionEngine.to_fraction(result.real)
            else:
                real_frac = FractionEngine.to_fraction(result.real)
                imag_frac = FractionEngine.to_fraction(result.imag)
                return f"{real_frac} + {imag_frac}i"
        except Exception as e:
            return f"Ошибка: {str(e)}"

    @staticmethod
    def mixed_fraction(decimal: float) -> str:
        try:
            if abs(decimal) < 1e-10:
                return "0"
            whole = int(decimal)
            fractional = decimal - whole
            if abs(fractional) < 1e-10:
                return str(whole)
            frac = Fraction(fractional).limit_denominator(1000)
            if whole == 0:
                return f"{frac.numerator}/{frac.denominator}"
            elif decimal > 0:
                return f"{whole} {frac.numerator}/{frac.denominator}"
            else:
                return f"-{abs(whole)} {frac.numerator}/{frac.denominator}"
        except:
            return str(decimal)

class InteractiveConsole:

    def __init__(self):
        self.history = []
        if 'console_variables' not in st.session_state:
            st.session_state.console_variables = {}
    
    @property
    def variables(self):
        return st.session_state.console_variables
    
    def execute(self, command: str) -> str:
        try:
            command = command.strip()
    
            if not command:
                return ""
    
            if command.lower() in ['quit', 'exit', 'q']:
                return "Выход из консоли"
    
            if '=' in command:
                var, expr = command.split('=', 1)
                var = var.strip()
                if not var.isidentifier():
                    return f"Недопустимое имя переменной: {var}"
    
                try:
                    result = parser.evaluate_expression(expr.strip(), self.variables)
                    self.variables[var] = result
                    parser.set_variable(var, result)
    
                    if abs(result.imag) < 1e-12:
                        return f"{var} = {result.real}"
                    else:
                        return f"{var} = {result.real} + {result.imag}j"
                except Exception as e:
                    return f"Ошибка вычисления: {str(e)}"
    
            elif command == 'clear':
                self.history.clear()
                self.variables.clear()
                parser.clear_variables()
                return "История и переменные очищены"
    
            elif command == 'vars':
                if not self.variables:
                    return "Переменные не определены"
                vars_list = []
                for k, v in self.variables.items():
                    if abs(v.imag) < 1e-12:
                        vars_list.append(f"{k} = {v.real}")
                    else:
                        vars_list.append(f"{k} = {v.real} + {v.imag}j")
                return "\n".join(vars_list)
    
            elif command == 'history':
                if not self.history:
                    return "История команд пуста"
                history_list = []
                for i, cmd in enumerate(self.history[-15:], 1):
                    history_list.append(f"{i:2d}: {cmd}")
                return "\n".join(history_list)
    
            elif command == 'help':
                return ("Доступные команды:\n"
                        "  x = выражение    - присвоение переменной\n"
                        "  vars            - список переменных\n"
                        "  history         - история команд\n"
                        "  clear           - очистка истории и переменных\n"
                        "  start:end:step  - создание диапазона\n"
                        "  help            - эта справка\n\n"
                        "Примеры:\n"
                        "  x = 2*pi + sin(0.5)\n"
                        "  y = 1:10:0.5\n"
                        "  z = x**2 + y")
    
            elif ':' in command and '=' not in command:
                try:
                    range_array = parser.parse_range(command)
                    if len(range_array) == 0:
                        parts = command.split(':')
                        if len(parts) == 2:
                            start, end = map(float, parts)
                            if start > end:
                                return f"Пустой диапазон: start={start} > end={end} (увеличьте end)"
                            else:
                                return f"Пустой диапазон: start={start}, end={end}, step=1.0"
                        else:
                            start, end, step = map(float, parts)
                            if (end - start) * step <= 0:
                                return f"Пустой диапазон: start={start}, end={end}, step={step} (неверное направление)"
                            else:
                                return f"Пустой диапазон: start={start}, end={end}, step={step}"
    
                    preview = list(range_array[:8])
                    preview_str = ", ".join(f"{x:.2f}" for x in preview)
                    if len(range_array) > 8:
                        preview_str += f", ... (всего {len(range_array)} элементов)"
                    return f"Диапазон: [{preview_str}]"
                except Exception as e:
                    return f"Ошибка диапазона: {str(e)}"
    
            else:
                result = parser.evaluate_expression(command, self.variables)
                self.history.append(command)
                if abs(result.imag) < 1e-12:
                    return f"{result.real}"
                else:
                    return f"{result.real} + {result.imag}j"
    
        except Exception as e:
            return f"Ошибка: {str(e)}"

parser = AdvancedMathParser()
fraction_engine = FractionEngine()
console = InteractiveConsole()

st.set_page_config(
    page_title="Math Calculator",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.caption("by quik")
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #0a0e1a 0%, #1e3a8a 100%) !important;
        background-attachment: fixed !important;
    }
    .main { 
        padding: 1rem;
        background: transparent !important;
    }
    .stTabs [data-baseweb="tab-list"] { 
        gap: 4px; 
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border-radius: 8px 8px 0 0;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] { 
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%); 
        border-radius: 8px 8px 0 0;
        padding: 10px 16px;
        font-weight: 500;
        border: 1px solid #3b82f6;
        margin: 0 2px;
        color: white !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { 
        background: linear-gradient(135deg, #3b82f6 0%, #60a5fa 100%) !important; 
        color: white !important;
        border-color: #60a5fa;
        box-shadow: 0 4px 6px rgba(59, 130, 246, 0.3);
    }
    .stTabs [data-baseweb="tab"]:hover { 
        background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%); 
        border-color: #60a5fa;
    }
    .console {
        background-color: #1E1E1E;
        color: #00FF00;
        padding: 15px;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        height: 400px;
        overflow-y: auto;
        border: 1px solid #444;
        font-size: 14px;
        line-height: 1.4;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    .header-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    [data-baseweb="select"] > div {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%) !important;
        border-radius: 8px !important;
        border: 1px solid #2563eb !important;
        color: white !important;
    }
    [data-baseweb="select"] > div:hover {
        border-color: #60a5fa !important;
        box-shadow: 0 2px 4px rgba(59, 130, 246, 0.3);
    }
    [data-baseweb="select"] svg {
        fill: white !important;
    }
    [data-baseweb="select"] input {
        color: white !important;
    }
    [data-baseweb="select"] [role="option"] {
        color: white !important;
    }
    [data-baseweb="select"] span {
        color: white !important;
    }
    [data-baseweb="popover"] {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%) !important;
        border-radius: 8px !important;
    }
    [data-baseweb="popover"] ul {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%) !important;
    }
    [data-baseweb="popover"] li {
        color: white !important;
    }
    [data-baseweb="popover"] li:hover {
        background-color: rgba(59, 130, 246, 0.5) !important;
    }
</style>
""",
            unsafe_allow_html=True)

if 'calc_history' not in st.session_state:
    st.session_state.calc_history = []
if 'console_history' not in st.session_state:
    st.session_state.console_history = []
if 'user_variables' not in st.session_state:
    st.session_state.user_variables = {}
if 'saved_graphs' not in st.session_state:
    st.session_state.saved_graphs = []


def save_history(operation, result, details=""):
    if 'calc_history' not in st.session_state:
        st.session_state.calc_history = []

    st.session_state.calc_history.append({
        'time':
        datetime.now().strftime("%H:%M:%S"),
        'operation':
        operation,
        'result':
        str(result),
        'details':
        details
    })

    if len(st.session_state.calc_history) > 50:
        st.session_state.calc_history = st.session_state.calc_history[-50:]


def export_data():
    session_data = {
        'timestamp': datetime.now().isoformat(),
        'history': st.session_state.calc_history,
        'variables': st.session_state.user_variables,
        'graphs': len(st.session_state.saved_graphs)
    }
    return json.dumps(session_data, indent=2, ensure_ascii=False)


def parse_matrix_input(matrix_str, rows, cols):
    try:
        values = [
            float(x.strip()) for x in matrix_str.replace(',', ' ').split()
        ]
        if len(values) != rows * cols:
            raise ValueError(f"Ожидается {rows*cols} элементов")
        return np.array(values).reshape(rows, cols)
    except ValueError as e:
        raise ValueError(f"Ошибка ввода матрицы: {str(e)}")


def parse_vector_input(vector_str):
    try:
        values = [
            float(x.strip()) for x in vector_str.replace(',', ' ').split()
        ]
        if len(values) == 0:
            raise ValueError("Вектор не может быть пустым")
        return np.array(values)
    except ValueError as e:
        raise ValueError(f"Ошибка ввода вектора: {str(e)}")


st.markdown('<div class="header-title">🧮 Math Calculator</div>',
            unsafe_allow_html=True)

with st.sidebar:
    st.subheader("📊 Управление данными")
    if st.button("Очистить всю историю", width='stretch'):
        st.session_state.calc_history = []
        st.session_state.console_history = []
        st.session_state.user_variables = {}
        parser.clear_variables()
        console.history.clear()
        console.variables.clear()
        st.rerun()
    if st.button("Экспорт всех данных", width='stretch'):
        session_json = export_data()
        st.download_button(
            label="Скачать данные",
            data=session_json,
            file_name=f"math_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            width='stretch')
    st.markdown("---")
    st.subheader("📈 История вычислений")
    if st.session_state.calc_history:
        with st.expander(
                f"Последние операции ({len(st.session_state.calc_history)})"):
            for i, item in enumerate(
                    reversed(st.session_state.calc_history[-8:])):
                with st.container():
                    st.caption(f"{item['time']}")
                    st.write(f"**{item['operation'][:30]}...**")
                    st.write(f"`{item['result']}`")
                    if i < len(st.session_state.calc_history[-8:]) - 1:
                        st.markdown("---")
    else:
        st.info("История вычислений пуста")
    st.markdown("---")
    st.header("Помощь с темами")
    st.info(
        "Проверьте что у вас тёмная тема в настройках streamlit\n(Три точки в правом углу экрана.)"
    )

tabs = st.tabs([
    "🔢 Калькулятор", "Консоль", "📈 Графики", "Матрицы", "🔺 Векторы", "🌐 3D",
    "🔧 Системы", "♾️ Символы", "⚡ Оптимизация", "📊 Анализ", "🧮 Дроби",
    "🎲 Генерация случайных данных",
    "📈 Дифференциальные уравнения и численные методы", "Комплексный анализ"
])

st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        flex-wrap: wrap;
        overflow-x: auto;
        white-space: nowrap;
    }
    .stTabs [data-baseweb="tab"] {
        display: inline-block;
    }
</style>
""",
            unsafe_allow_html=True)

with tabs[0]:
    st.header("🔢 Расширенный калькулятор")
    col1, col2 = st.columns([3, 1])
    with col1:
        expression = st.text_area(
            "Введите математическое выражение:",
            height=120,
            placeholder=
            "Примеры:\n• 1/2 + 1/3 + 1/6 (дроби)\n• (3+4j) * (2-1j) (комплексные)\n• factorial(5) (факториал)\n• cbrt(27) + erf(1) (спец. функции)\n• x = 1:10:0.5 (диапазоны)\n• sin(pi/2) * exp(1) (стандартные)",
            help=
            "Поддерживаются дроби, комплексные числа, специальные функции, факториал, диапазоны"
        )
    with col2:
        mode = st.radio(
            "Режим вычислений:",
            ["Авто", "Точные дроби", "Комплексные", "Быстрый парсер"],
            index=0)
        precision = st.slider("Точность:", 2, 16, 8)
        st.markdown("---")
        if st.button("Очистить кэш", width='stretch'):
            parser.cache.clear()
            st.success("Кэш очищен!")
    if st.button("Вычислить выражение", type="primary",
                 width='stretch') and expression:
        with st.spinner("Вычисление..."):
            start_time = time.time()
            try:
                if mode == "Точные дроби":
                    result = fraction_engine.evaluate_fraction(expression)
                    exec_time = time.time() - start_time
                    st.markdown(
                        f'<div class="success-box"><h4>Результат (дроби):</h4><h3>{result}</h3></div>',
                        unsafe_allow_html=True)
                    try:
                        decimal_result = parser.evaluate_expression(expression)
                        if abs(decimal_result.imag) < 1e-12:
                            st.write(
                                f"**Десятичное:** {decimal_result.real:.{precision}f}"
                            )
                        else:
                            st.write(
                                f"**Десятичное:** {decimal_result.real:.{precision}f} + {decimal_result.imag:.{precision}f}i"
                            )
                    except:
                        pass
                elif mode == "Быстрый парсер":
                    result = parser.parse_cached(expression)
                    exec_time = time.time() - start_time
                    if abs(result.imag) < 1e-12:
                        display_result = result.real
                        st.markdown(
                            f'<div class="success-box"><h4>⚡ Результат (парсер):</h4><h3>{display_result:.{precision}f}</h3></div>',
                            unsafe_allow_html=True)
                        frac_result = fraction_engine.mixed_fraction(
                            display_result)
                        if '/' in frac_result:
                            st.markdown(
                                f'<div class="info-box"><strong>Смешанная дробь:</strong> {frac_result}</div>',
                                unsafe_allow_html=True)
                    else:
                        st.markdown(
                            f'<div class="success-box"><h4>⚡ Результат (комплексный):</h4><h3>{result.real:.{precision}f} + {result.imag:.{precision}f}i</h3></div>',
                            unsafe_allow_html=True)
                    st.info(
                        f"Время выполнения: {exec_time:.4f} секунд (с кэшированием)"
                    )
                else:
                    result = parser.evaluate_expression(expression)
                    exec_time = time.time() - start_time
                    if abs(result.imag) < 1e-12:
                        display_result = result.real
                        st.markdown(
                            f'<div class="success-box"><h4>Результат:</h4><h3>{display_result:.{precision}f}</h3></div>',
                            unsafe_allow_html=True)
                        if mode == "Авто" and (display_result
                                               != int(display_result)
                                               and abs(display_result) < 1000):
                            frac_result = fraction_engine.to_fraction(
                                display_result)
                            if '/' in frac_result:
                                st.markdown(
                                    f'<div class="info-box"><strong>Обычная дробь:</strong> {frac_result}</div>',
                                    unsafe_allow_html=True)
                            mixed_result = fraction_engine.mixed_fraction(
                                display_result)
                            if ' ' in mixed_result:
                                st.markdown(
                                    f'<div class="info-box"><strong>Смешанная дробь:</strong> {mixed_result}</div>',
                                    unsafe_allow_html=True)
                    else:
                        st.markdown(
                            f'<div class="success-box"><h4>Результат (комплексный):</h4><h3>{result.real:.{precision}f} + {result.imag:.{precision}f}i</h3></div>',
                            unsafe_allow_html=True)
                    st.info(f"Время выполнения: {exec_time:.4f} секунд")
                col1, col2, col3 = st.columns(3)
                with col1:
                    if abs(result.imag) < 1e-12:
                        st.metric("Научная запись",
                                  f"{result.real:.{precision}e}")
                    else:
                        st.metric("Модуль", f"{abs(result):.{precision}f}")
                with col2:
                    if abs(result.imag) < 1e-12:
                        st.metric("Округление",
                                  f"{round(result.real, precision)}")
                    else:
                        st.metric("Аргумент",
                                  f"{cmath.phase(result):.{precision}f}")
                with col3:
                    st.metric("Режим", mode)
                save_history(expression, str(result),
                             f"Режим: {mode}, Время: {exec_time:.4f}с")
            except Exception as e:
                st.error(f"❌ Ошибка вычисления: {str(e)}")
                st.info(
                    "💡 **Подсказка:** Проверьте синтаксис выражения и используйте поддерживаемые функции"
                )
    st.markdown("---")
    st.subheader("Быстрые вычисления")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🔢 Специальные функции**")
        spec_func = st.text_input("Выражение со спец. функциями:",
                                  "cbrt(64) + gamma(5) + erf(0.5)",
                                  key="spec_func_col1")
        if st.button("Вычислить спец. функции", key="btn_spec_col1"):
            try:
                result = parser.evaluate_expression(spec_func)
                if abs(result.imag) < 1e-12:
                    st.success(f"**= {result.real:.8f}**")
                else:
                    st.success(f"**= {result.real:.8f} + {result.imag:.8f}i**")
            except Exception as e:
                st.error(f"Ошибка: {str(e)}")
    with col2:
        st.markdown("**Комбинаторика**")
        comb_expr = st.text_input("Комбинаторное выражение:",
                                  "factorial(5) + 120",
                                  key="comb_expr_col2")
        if st.button("Вычислить комбинаторику", key="btn_comb_col2"):
            try:
                result = parser.evaluate_expression(comb_expr)
                st.success(f"**= {result.real:.0f}**")
            except Exception as e:
                st.error(f"Ошибка: {str(e)}")
    with col3:
        st.markdown("**Диапазоны**")
        range_expr = st.text_input("Диапазон (start:end:step):",
                                   "0:10:0.5",
                                   key="range_expr_col3")
        if st.button("Создать диапазон", key="btn_range_col3"):
            try:
                arr = parser.parse_range(range_expr)
                preview = list(arr[:6])
                preview_str = ", ".join(f"{x:.1f}" for x in preview)
                if len(arr) > 6:
                    preview_str += f", ... (всего {len(arr)} элементов)"
                st.success(f"**Создан массив:** [{preview_str}]")
                st.info(
                    f"**Характеристики:** мин={arr.min():.2f}, макс={arr.max():.2f}, шаг={arr[1]-arr[0]:.2f}"
                )
            except Exception as e:
                st.error(f"Ошибка: {str(e)}")

with tabs[1]:
    st.header("Интерактивная консоль")

    st.markdown("""
    **Справка по командам:**
    - `переменная = выражение` - создание переменной
    - `диапазон` - создание массива (формат: start:end:step)
    - `vars` - список всех переменных
    - `history` - история команд
    - `clear` - очистка истории и переменных
    - `help` - показать эту справку

    **Примеры:**
    - `x = 2*pi + sin(0.5)`
    - `arr = 0:10:0.5` 
    - `y = x**2 + 1`
    """)

    if st.session_state.console_variables:
        st.subheader("Текущие переменные:")
        for var_name, var_value in st.session_state.console_variables.items():
            if abs(var_value.imag) < 1e-12:
                st.write(f"`{var_name} = {var_value.real:.6f}`")
            else:
                st.write(f"`{var_name} = {var_value.real:.6f} + {var_value.imag:.6f}j`")

    console_output = st.container()
    with console_output:
        st.markdown("**Вывод консоли:**")
        console_content = "\n".join(
            [f"> {line}" for line in st.session_state.console_history[-20:]])
        st.markdown(f'<div class="console">{console_content}</div>',
                    unsafe_allow_html=True)

    with st.form("console_form", clear_on_submit=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            cmd = st.text_input("Введите команду:",
                                key="console_input",
                                placeholder="Например: x = 2*pi + sin(0.5)")
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("⏎ Выполнить", width='stretch')

        if submitted and cmd:
            result = console.execute(cmd)
            st.session_state.console_history.append(f"{cmd}")
            st.session_state.console_history.append(f"  → {result}")
            st.rerun()

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("История", width='stretch'):
            result = console.execute("history")
            st.session_state.console_history.append(f"history")
            st.session_state.console_history.append(f"  → {result}")
            st.rerun()

    with col2:
        if st.button("Переменные", width='stretch'):
            result = console.execute("vars")
            st.session_state.console_history.append(f"vars")
            st.session_state.console_history.append(f"  → {result}")
            st.rerun()

    with col3:
        if st.button("Справка", width='stretch'):
            result = console.execute("help")
            st.session_state.console_history.append(f"help")
            st.session_state.console_history.append(f"  → {result}")
            st.rerun()

    if st.button("Очистить консоль", width='stretch', type="secondary"):
        result = console.execute("clear")
        st.session_state.console_history.append(f"clear")
        st.session_state.console_history.append(f"  → {result}")
        st.rerun()

with tabs[2]:
    st.header("📈 Графики функций")

    col1, col2 = st.columns([2, 1])

    with col1:
        func_input = st.text_input("Функция f(x) =",
                                   "sin(x) + cos(2*x)",
                                   key="func_2d")
        st.caption("💡 Используйте x как переменную.")

    with col2:
        plot_type = st.selectbox("Тип графика", [
            "Функция", "Функция + Производная", "Функция + Интеграл",
            "Все вместе"
        ],
                                 key="plot_type_2d")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        x_min = st.number_input("x минимум", -50.0, 50.0, -10.0, key="xmin_2d")
    with col2:
        x_max = st.number_input("x максимум", -50.0, 50.0, 10.0, key="xmax_2d")
    with col3:
        points = st.slider("Количество точек",
                           100,
                           5000,
                           1000,
                           key="points_2d_graph")
    with col4:
        line_width = st.slider("Толщина линии", 1, 5, 2, key="line_width_2d")

    if st.button("📊 Построить график", type="primary", width='stretch'):
        try:
            x_sym = sp.Symbol('x')
            expr = sp.sympify(func_input)
            f_lambda = sp.lambdify(x_sym, expr, 'numpy')

            x_vals = np.linspace(x_min, x_max, points)
            y_vals = f_lambda(x_vals)

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(x=x_vals,
                           y=y_vals,
                           mode='lines',
                           name=f'f(x) = {func_input}',
                           line=dict(width=line_width, color='blue')))

            if "Производная" in plot_type or "Все" in plot_type:
                try:
                    derivative = sp.diff(expr, x_sym)
                    df_lambda = sp.lambdify(x_sym, derivative, 'numpy')
                    dy_vals = df_lambda(x_vals)

                    fig.add_trace(
                        go.Scatter(x=x_vals,
                                   y=dy_vals,
                                   mode='lines',
                                   name=f"f'(x) = {sp.latex(derivative)}",
                                   line=dict(width=line_width - 1,
                                             color='red',
                                             dash='dash')))
                except Exception as e:
                    st.warning(f"Не удалось вычислить производную: {e}")

            if "Интеграл" in plot_type or "Все" in plot_type:
                try:
                    integral = sp.integrate(expr, x_sym)
                    int_lambda = sp.lambdify(x_sym, integral, 'numpy')
                    int_vals = int_lambda(x_vals)

                    fig.add_trace(
                        go.Scatter(x=x_vals,
                                   y=int_vals,
                                   mode='lines',
                                   name=f"∫f(x)dx = {sp.latex(integral)}",
                                   line=dict(width=line_width - 1,
                                             color='green',
                                             dash='dot')))
                except Exception as e:
                    st.warning(f"Не удалось вычислить интеграл: {e}")

            fig.update_layout(title=f"График функции: {func_input}",
                              xaxis_title="x",
                              yaxis_title="y",
                              template=plotly_template,
                              height=600,
                              showlegend=True,
                              hovermode='x unified')

            st.plotly_chart(fig,
                            use_container_width=True,
                            config={'displayModeBar': True})

            st.subheader("Анализ функции")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Производная:**")
                try:
                    derivative = sp.diff(expr, x_sym)
                    st.latex(f"f'(x) = {sp.latex(derivative)}")
                except:
                    st.write("Не вычисляется")

            with col2:
                st.markdown("**Интеграл:**")
                try:
                    integral = sp.integrate(expr, x_sym)
                    st.latex(f"\\int f(x) dx = {sp.latex(integral)} + C")
                except:
                    st.write("Не вычисляется")

            with col3:
                st.markdown("**Пределы:**")
                try:
                    limit_inf = sp.limit(expr, x_sym, sp.oo)
                    st.latex(
                        f"\\lim_{{x \\to \\infty}} f(x) = {sp.latex(limit_inf)}"
                    )
                except:
                    st.write("Не вычисляется")

            save_history(f"График: {func_input}", "Построен",
                         f"Диапазон: [{x_min}, {x_max}], Точек: {points}")

        except Exception as e:
            st.error(f"❌ Ошибка построения графика: {str(e)}")

    st.markdown("---")
    st.subheader("Поиск корней уравнения")

    col1, col2 = st.columns(2)

    with col1:
        root_eq = st.text_input("Уравнение f(x) = 0",
                                "x**2 - 4",
                                key="root_eq")

    with col2:
        root_range = st.slider("Диапазон поиска корней",
                               -20.0,
                               20.0, (-5.0, 5.0),
                               key="root_range")

    if st.button("🔍 Найти корни", width='stretch'):
        try:
            x_sym = sp.Symbol('x')
            expr = sp.sympify(root_eq)
            f_lambda = sp.lambdify(x_sym, expr, 'numpy')

            roots = []
            for x0 in np.linspace(root_range[0], root_range[1], 50):
                try:
                    root = optimize.root_scalar(f_lambda,
                                                x0=x0,
                                                method='newton')
                    if root.converged and root_range[
                            0] <= root.root <= root_range[1]:
                        if not any(abs(root.root - r) < 0.001 for r in roots):
                            roots.append(root.root)
                except:
                    continue

            if roots:
                st.success(f"Найдено корней: {len(roots)}")
                for i, root in enumerate(sorted(roots), 1):
                    st.write(f"**x_{i}** = {root:.8f}")
                    st.write(f"  f(x_{i}) = {f_lambda(root):.2e}")
            else:
                st.warning("Корни не найдены в указанном диапазоне")

        except Exception as e:
            st.error(f"Ошибка поиска корней: {str(e)}")

with tabs[3]:
    st.header("🎲 Матричные операции")

    operation = st.selectbox("Выберите операцию", [
        "Сложение/Вычитание", "Умножение", "Поэлементное умножение",
        "Определитель", "Обратная матрица", "Возведение в степень",
        "Собственные значения", "LU-разложение", "QR-разложение", "SVD",
        "Ранг", "След", "Норма"
    ],
                             key="matrix_op")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Матрица A")
        rows_a = st.number_input("Строк A", 2, 8, 3, key="rows_a")
        cols_a = st.number_input("Столбцов A", 2, 8, 3, key="cols_a")

        st.write("Введите элементы построчно (через пробел):")
        a_data = []
        for i in range(rows_a):
            default_row = " ".join(
                ["1" if j == i else "0" for j in range(cols_a)])
            row = st.text_input(f"Строка {i+1} A", default_row, key=f"a_{i}")
            try:
                a_data.append([float(x) for x in row.split()])
            except ValueError:
                st.error(f"Ошибка в строке {i+1}")
                a_data = None
                break

        if a_data and all(len(row) == cols_a for row in a_data):
            A = np.array(a_data)
            st.write("**Матрица A:**")
            st.dataframe(pd.DataFrame(A), width='stretch')
        else:
            A = None

    with col2:
        if operation in [
                "Сложение/Вычитание", "Умножение", "Поэлементное умножение"
        ]:
            st.subheader("Матрица B")
            rows_b = st.number_input("Строк B", 2, 8, 3, key="rows_b")
            cols_b = st.number_input("Столбцов B", 2, 8, 3, key="cols_b")

            st.write("Введите элементы построчно (через пробел):")
            b_data = []
            for i in range(rows_b):
                default_row = " ".join(
                    ["2" if j == i else "0" for j in range(cols_b)])
                row = st.text_input(f"Строка {i+1} B",
                                    default_row,
                                    key=f"b_{i}")
                try:
                    b_data.append([float(x) for x in row.split()])
                except ValueError:
                    st.error(f"Ошибка в строке {i+1}")
                    b_data = None
                    break

            if b_data and all(len(row) == cols_b for row in b_data):
                B = np.array(b_data)
                st.write("**Матрица B:**")
                st.dataframe(pd.DataFrame(B), width='stretch')
            else:
                B = None
        else:
            B = None

    if operation == "Возведение в степень":
        power = st.number_input("Степень", 1, 10, 2, key="matrix_power")

    if operation == "Норма":
        norm_type = st.selectbox(
            "Тип нормы",
            ["Фробениуса", "L1", "L2", "Бесконечность"],
            key="norm_type")

    if operation == "Сложение/Вычитание":
        op_type = st.radio("Операция", ["Сложение", "Вычитание"],
                           horizontal=True,
                           key="add_sub")

    if st.button("Вычислить", type="primary", width='stretch'):
        if A is None:
            st.error("Ошибка в матрице A")
        elif operation in [
                "Сложение/Вычитание", "Умножение", "Поэлементное умножение"
        ] and B is None:
            st.error("Ошибка в матрице B")
        else:
            try:
                if operation == "Сложение/Вычитание":
                    if A.shape == B.shape:
                        result = A + B if op_type == "Сложение" else A - B
                        st.success(f"Результат ({op_type}):")
                        st.dataframe(pd.DataFrame(result), width='stretch')
                    else:
                        st.error("Размеры матриц должны совпадать")

                elif operation == "Умножение":
                    if A.shape[1] == B.shape[0]:
                        result = A @ B
                        st.success("Результат (A × B):")
                        st.dataframe(pd.DataFrame(result), width='stretch')
                    else:
                        st.error(
                            "Число столбцов A должно равняться числу строк B")

                elif operation == "Поэлементное умножение":
                    if A.shape == B.shape:
                        result = A * B
                        st.success("Результат (поэлементное умножение):")
                        st.dataframe(pd.DataFrame(result), width='stretch')
                    else:
                        st.error("Размеры матриц должны совпадать")

                elif operation == "Определитель":
                    if A.shape[0] == A.shape[1]:
                        det = np.linalg.det(A)
                        st.success(f"**det(A) = {det:.8f}**")
                        save_history("Определитель матрицы", f"{det:.8f}")
                    else:
                        st.error("Матрица должна быть квадратной")

                elif operation == "Обратная матрица":
                    if A.shape[0] == A.shape[1]:
                        try:
                            inv = np.linalg.inv(A)
                            st.success("**Обратная матрица A⁻¹:**")
                            st.dataframe(pd.DataFrame(inv), width='stretch')

                            st.info("**Проверка (A × A⁻¹):**")
                            identity_check = A @ inv
                            st.dataframe(pd.DataFrame(identity_check),
                                         width='stretch')
                        except np.linalg.LinAlgError:
                            st.error("Матрица вырождена (определитель = 0)")
                    else:
                        st.error("Матрица должна быть квадратной")

                elif operation == "Возведение в степень":
                    if A.shape[0] == A.shape[1]:
                        result = np.linalg.matrix_power(A, power)
                        st.success(f"**Результат A^{power}:**")
                        st.dataframe(pd.DataFrame(result), width='stretch')
                    else:
                        st.error("Матрица должна быть квадратной")

                elif operation == "Собственные значения":
                    if A.shape[0] == A.shape[1]:
                        eigenvalues, eigenvectors = np.linalg.eig(A)
                        st.success("**Собственные значения:**")
                        for i, val in enumerate(eigenvalues):
                            st.write(f"**λ_{i+1}** = {val:.8f}")

                        st.info("**Собственные векторы:**")
                        st.dataframe(pd.DataFrame(eigenvectors),
                                     width='stretch')
                    else:
                        st.error("Матрица должна быть квадратной")

                elif operation == "LU-разложение":
                    if A.shape[0] == A.shape[1]:
                        try:
                            P, L, U = linalg.lu(A)
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.write("**Матрица P:**")
                                st.dataframe(pd.DataFrame(P), width='stretch')
                            with col2:
                                st.write("**Матрица L:**")
                                st.dataframe(pd.DataFrame(L), width='stretch')
                            with col3:
                                st.write("**Матрица U:**")
                                st.dataframe(pd.DataFrame(U), width='stretch')
                        except:
                            st.error("Ошибка LU-разложения")
                    else:
                        st.error("Матрица должна быть квадратной")

                elif operation == "QR-разложение":
                    Q, R = np.linalg.qr(A)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Матрица Q:**")
                        st.dataframe(pd.DataFrame(Q), width='stretch')
                    with col2:
                        st.write("**Матрица R:**")
                        st.dataframe(pd.DataFrame(R), width='stretch')

                elif operation == "SVD":
                    U, S, Vt = np.linalg.svd(A)
                    st.success("**Сингулярные значения:**")
                    st.write(S)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write("**Матрица U:**")
                        st.dataframe(pd.DataFrame(U), width='stretch')
                    with col2:
                        st.write("**Матрица Σ:**")
                        st.dataframe(pd.DataFrame(np.diag(S)), width='stretch')
                    with col3:
                        st.write("**Матрица Vᵀ:**")
                        st.dataframe(pd.DataFrame(Vt), width='stretch')

                elif operation == "Ранг":
                    rank = np.linalg.matrix_rank(A)
                    st.success(f"**Ранг матрицы: {rank}**")

                elif operation == "След":
                    if A.shape[0] == A.shape[1]:
                        trace = np.trace(A)
                        st.success(f"**След матрицы: {trace:.8f}**")
                    else:
                        st.error("Матрица должна быть квадратной")

                elif operation == "Норма":
                    if norm_type == "Фробениуса":
                        norm_val = np.linalg.norm(A, 'fro')
                    elif norm_type == "L1":
                        norm_val = np.linalg.norm(A, 1)
                    elif norm_type == "L2":
                        norm_val = np.linalg.norm(A, 2)
                    else:
                        norm_val = np.linalg.norm(A, np.inf)
                    st.success(f"**Норма ({norm_type}): {norm_val:.8f}**")

                save_history(f"Матричная операция: {operation}",
                             "Успешно выполнена")

            except Exception as e:
                st.error(f"❌ Ошибка вычисления: {str(e)}")

with tabs[4]:
    st.header("🔺 Векторные операции")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Вектор A")
        dim = st.number_input("Размерность", 2, 10, 3, key="vec_dim")
        vec_a_input = st.text_input("Координаты A (через пробел)",
                                    "1 2 3",
                                    key="vec_a")
        try:
            vec_a = parse_vector_input(vec_a_input)
            if len(vec_a) != dim:
                st.error(f"Ожидается {dim} координат, получено {len(vec_a)}")
                vec_a = None
            else:
                st.write("**Вектор A:**", vec_a)
        except Exception as e:
            st.error(str(e))
            vec_a = None

    with col2:
        st.subheader("Вектор B")
        vec_b_input = st.text_input("Координаты B (через пробел)",
                                    "4 5 6",
                                    key="vec_b")
        try:
            vec_b = parse_vector_input(vec_b_input)
            if len(vec_b) != dim:
                st.error(f"Ожидается {dim} координат, получено {len(vec_b)}")
                vec_b = None
            else:
                st.write("**Вектор B:**", vec_b)
        except Exception as e:
            st.error(str(e))
            vec_b = None

    if st.button("Вычислить векторные операции",
                 type="primary",
                 width='stretch'):
        if vec_a is None or vec_b is None:
            st.error("Ошибка в векторах")
        elif len(vec_a) != len(vec_b):
            st.error("Векторы должны быть одинаковой размерности")
        else:
            try:
                st.subheader("📊 Результаты")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Скалярное произведение",
                              f"{np.dot(vec_a, vec_b):.8f}")
                    st.metric("Длина вектора A",
                              f"{np.linalg.norm(vec_a):.8f}")
                    st.metric("Длина вектора B",
                              f"{np.linalg.norm(vec_b):.8f}")

                with col2:
                    if np.linalg.norm(vec_a) > 1e-10 and np.linalg.norm(
                            vec_b) > 1e-10:
                        cos_angle = np.dot(vec_a, vec_b) / (
                            np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
                        cos_angle = np.clip(cos_angle, -1, 1)
                        angle_rad = np.arccos(cos_angle)
                        angle_deg = np.degrees(angle_rad)

                        st.metric("Косинус угла", f"{cos_angle:.8f}")
                        st.metric("Угол (радианы)", f"{angle_rad:.8f}")
                        st.metric("Угол (градусы)", f"{angle_deg:.8f}")
                    else:
                        st.warning("Один из векторов нулевой")

                with col3:
                    st.write("**A + B:**", vec_a + vec_b)
                    st.write("**A - B:**", vec_a - vec_b)

                    if np.linalg.norm(vec_b) > 1e-10:
                        projection = (np.dot(vec_a, vec_b) /
                                      np.dot(vec_b, vec_b)) * vec_b
                        st.write("**Проекция A на B:**", projection)

                    if len(vec_a) == 3 and len(vec_b) == 3:
                        cross_product = np.cross(vec_a, vec_b)
                        st.write("**Векторное произведение A × B:**",
                                 cross_product)

                if len(vec_a) == 3 and len(vec_b) == 3:
                    st.subheader("3D Визуализация векторов")

                    fig = go.Figure()

                    origin = [0, 0, 0]

                    fig.add_trace(
                        go.Scatter3d(x=[origin[0], vec_a[0]],
                                     y=[origin[1], vec_a[1]],
                                     z=[origin[2], vec_a[2]],
                                     mode='lines+markers+text',
                                     name='Вектор A',
                                     line=dict(color='blue', width=8),
                                     marker=dict(size=6, color='blue'),
                                     text=['', 'A'],
                                     textposition="top center"))

                    fig.add_trace(
                        go.Scatter3d(x=[origin[0], vec_b[0]],
                                     y=[origin[1], vec_b[1]],
                                     z=[origin[2], vec_b[2]],
                                     mode='lines+markers+text',
                                     name='Вектор B',
                                     line=dict(color='red', width=8),
                                     marker=dict(size=6, color='red'),
                                     text=['', 'B'],
                                     textposition="top center"))

                    if np.linalg.norm(np.cross(vec_a, vec_b)) > 1e-10:
                        cross_product = np.cross(vec_a, vec_b)
                        fig.add_trace(
                            go.Scatter3d(x=[origin[0], cross_product[0]],
                                         y=[origin[1], cross_product[1]],
                                         z=[origin[2], cross_product[2]],
                                         mode='lines+markers+text',
                                         name='A × B',
                                         line=dict(color='green',
                                                   width=8,
                                                   dash='dash'),
                                         marker=dict(size=6, color='green'),
                                         text=['', 'A×B'],
                                         textposition="top center"))

                    max_val = max(np.max(np.abs(vec_a)), np.max(
                        np.abs(vec_b))) * 1.2
                    fig.update_layout(
                        title="3D Визуализация векторов",
                        scene=dict(xaxis_title='X',
                                   yaxis_title='Y',
                                   zaxis_title='Z',
                                   xaxis=dict(range=[-max_val, max_val]),
                                   yaxis=dict(range=[-max_val, max_val]),
                                   zaxis=dict(range=[-max_val, max_val])),
                        template=plotly_template,
                        height=600)

                    st.plotly_chart(fig,
                                    use_container_width=True,
                                    config={'displayModeBar': True})

                save_history("Векторные операции", "Успешно выполнены")

            except Exception as e:
                st.error(f"❌ Ошибка вычисления: {str(e)}")

with tabs[5]:
    st.header("🌐 3D Графики и поверхности")

    plot_3d_type = st.selectbox(
        "Тип 3D графика",
        ["Поверхность z=f(x,y)", "Параметрическая кривая", "Векторное поле"],
        key="3d_type")

    if plot_3d_type == "Поверхность z=f(x,y)":
        func_3d = st.text_input("Функция z = f(x,y)",
                                "sin(sqrt(x**2 + y**2))",
                                key="func_3d")
        st.caption("💡 Используйте x и y как переменные")

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            x_min_3d = st.number_input("x min",
                                       -10.0,
                                       10.0,
                                       -5.0,
                                       key="xmin_3d")
        with col2:
            x_max_3d = st.number_input("x max",
                                       -10.0,
                                       10.0,
                                       5.0,
                                       key="xmax_3d")
        with col3:
            y_min_3d = st.number_input("y min",
                                       -10.0,
                                       10.0,
                                       -5.0,
                                       key="ymin_3d")
        with col4:
            y_max_3d = st.number_input("y max",
                                       -10.0,
                                       10.0,
                                       5.0,
                                       key="ymax_3d")
        with col5:
            resolution = st.slider("Разрешение",
                                   20,
                                   200,
                                   50,
                                   key="res_3d_surface")

        if st.button("🌐 Построить 3D поверхность",
                     type="primary",
                     width='stretch'):
            try:
                x_sym, y_sym = sp.symbols('x y')
                expr = sp.sympify(func_3d)
                f_lambda = sp.lambdify((x_sym, y_sym), expr, 'numpy')

                x = np.linspace(x_min_3d, x_max_3d, resolution)
                y = np.linspace(y_min_3d, y_max_3d, resolution)
                X, Y = np.meshgrid(x, y)
                Z = f_lambda(X, Y)

                fig = go.Figure(data=[
                    go.Surface(x=X,
                               y=Y,
                               z=Z,
                               colorscale='Viridis',
                               opacity=0.9,
                               showscale=True)
                ])

                fig.update_layout(
                    title=f"3D Поверхность: z = {func_3d}",
                    scene=dict(xaxis_title='X',
                               yaxis_title='Y',
                               zaxis_title='Z',
                               camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))),
                    template=plotly_template,
                    height=700)

                st.plotly_chart(fig,
                                use_container_width=True,
                                config={'displayModeBar': True})

                st.info(f"**Характеристики поверхности:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Минимум z", f"{np.min(Z):.4f}")
                with col2:
                    st.metric("Максимум z", f"{np.max(Z):.4f}")
                with col3:
                    st.metric("Среднее z", f"{np.mean(Z):.4f}")

                save_history(
                    f"3D поверхность: {func_3d}", "Построена",
                    f"Область: x∈[{x_min_3d},{x_max_3d}], y∈[{y_min_3d},{y_max_3d}]"
                )

            except Exception as e:
                st.error(f"❌ Ошибка построения: {str(e)}")

    elif plot_3d_type == "Параметрическая кривая":
        st.subheader("Параметрическая кривая: r(t) = (x(t), y(t), z(t))")

        col1, col2, col3 = st.columns(3)
        with col1:
            x_param = st.text_input("x(t)", "cos(3*t)", key="x_param")
        with col2:
            y_param = st.text_input("y(t)", "sin(3*t)", key="y_param")
        with col3:
            z_param = st.text_input("z(t)", "t", key="z_param")

        col1, col2, col3 = st.columns(3)
        with col1:
            t_min = st.number_input("t min", 0.0, 100.0, 0.0, key="tmin_3d")
        with col2:
            t_max = st.number_input("t max", 0.0, 100.0, 10.0, key="tmax_3d")
        with col3:
            t_points = st.slider("Количество точек",
                                 50,
                                 2000,
                                 500,
                                 key="tpoints_3d_curve")

        if st.button("Построить параметрическую кривую",
                     type="primary",
                     width='stretch'):
            try:
                t_sym = sp.Symbol('t')
                x_expr = sp.sympify(x_param)
                y_expr = sp.sympify(y_param)
                z_expr = sp.sympify(z_param)

                x_lambda = sp.lambdify(t_sym, x_expr, 'numpy')
                y_lambda = sp.lambdify(t_sym, y_expr, 'numpy')
                z_lambda = sp.lambdify(t_sym, z_expr, 'numpy')

                t = np.linspace(t_min, t_max, t_points)
                x_vals = x_lambda(t)
                y_vals = y_lambda(t)
                z_vals = z_lambda(t)

                fig = go.Figure(data=[
                    go.Scatter3d(x=x_vals,
                                 y=y_vals,
                                 z=z_vals,
                                 mode='lines',
                                 line=dict(
                                     color=t, colorscale='Rainbow', width=6),
                                 name='Кривая')
                ])

                fig.update_layout(
                    title=
                    f"Параметрическая кривая: x={x_param}, y={y_param}, z={z_param}",
                    scene=dict(xaxis_title='X',
                               yaxis_title='Y',
                               zaxis_title='Z'),
                    template=plotly_template,
                    height=700)

                st.plotly_chart(fig,
                                use_container_width=True,
                                config={'displayModeBar': True})

                save_history(
                    f"Параметрическая кривая", "Построена",
                    f"x={x_param}, y={y_param}, z={z_param}, t∈[{t_min},{t_max}]"
                )

            except Exception as e:
                st.error(f"❌ Ошибка построения: {str(e)}")

    elif plot_3d_type == "Векторное поле":
        st.subheader("Векторное поле F(x,y) = [P(x,y), Q(x,y)]")

        col1, col2 = st.columns(2)
        with col1:
            p_func = st.text_input("P(x,y) - x компонента", "-y", key="p_func")
        with col2:
            q_func = st.text_input("Q(x,y) - y компонента", "x", key="q_func")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            vf_x_min = st.number_input("x min", -5.0, 5.0, -2.0, key="vf_xmin")
        with col2:
            vf_x_max = st.number_input("x max", -5.0, 5.0, 2.0, key="vf_xmax")
        with col3:
            vf_y_min = st.number_input("y min", -5.0, 5.0, -2.0, key="vf_ymin")
        with col4:
            vf_y_max = st.number_input("y max", -5.0, 5.0, 2.0, key="vf_ymax")

        grid_density = st.slider("Плотность сетки",
                                 5,
                                 30,
                                 15,
                                 key="grid_density_field")
        arrow_scale = st.slider("Масштаб стрелок",
                                0.1,
                                2.0,
                                0.3,
                                key="arrow_scale_field")

        if st.button("Построить векторное поле",
                     type="primary",
                     width='stretch'):
            try:
                x_sym, y_sym = sp.symbols('x y')

                p_expr = sp.sympify(p_func)
                q_expr = sp.sympify(q_func)

                p_lambda = sp.lambdify((x_sym, y_sym), p_expr, 'numpy')
                q_lambda = sp.lambdify((x_sym, y_sym), q_expr, 'numpy')

                x = np.linspace(vf_x_min, vf_x_max, grid_density)
                y = np.linspace(vf_y_min, vf_y_max, grid_density)
                X, Y = np.meshgrid(x, y)

                U = p_lambda(X, Y)
                V = q_lambda(X, Y)

                fig = ff.create_quiver(X,
                                       Y,
                                       U,
                                       V,
                                       scale=arrow_scale,
                                       name='Векторное поле')

                fig.update_layout(
                    title=f"Векторное поле: F(x,y) = [{p_func}, {q_func}]",
                    xaxis_title='X',
                    yaxis_title='Y',
                    template=plotly_template,
                    height=600)

                st.plotly_chart(fig,
                                use_container_width=True,
                                config={'displayModeBar': True})

                st.info("**Анализ векторного поля:**")
                col1, col2 = st.columns(2)
                with col1:
                    divergence = sp.diff(p_expr, x_sym) + sp.diff(
                        q_expr, y_sym)
                    st.write(f"**Дивергенция:** ∇·F = {sp.latex(divergence)}")
                with col2:
                    curl = sp.diff(q_expr, x_sym) - sp.diff(p_expr, y_sym)
                    st.write(f"**Ротор (2D):** ∇×F = {sp.latex(curl)}")

                save_history(f"Векторное поле", "Построено",
                             f"F(x,y)=[{p_func}, {q_func}]")

            except Exception as e:
                st.error(f"❌ Ошибка построения: {str(e)}")

with tabs[6]:
    st.header("🔧 Системы линейных уравнений")

    st.markdown("Решение системы вида **Ax = b**")

    n_equations = st.number_input("Количество уравнений/неизвестных",
                                  2,
                                  10,
                                  3,
                                  key="n_eq")

    st.subheader("Матрица коэффициентов A")
    st.write("Введите коэффициенты уравнений построчно (через пробел):")

    A_data = []
    for i in range(n_equations):
        default_row = " ".join(
            [str(j + 1) if j == i else "1" for j in range(n_equations)])
        row = st.text_input(f"Уравнение {i+1} коэффициенты",
                            default_row,
                            key=f"eq_{i}")
        try:
            A_data.append([float(x) for x in row.split()])
        except ValueError:
            st.error(f"Ошибка в уравнении {i+1}")
            A_data = None
            break

    if A_data and all(len(row) == n_equations for row in A_data):
        A = np.array(A_data)
        st.write("**Матрица A:**")
        st.dataframe(pd.DataFrame(A), width='stretch')
    else:
        A = None

    st.subheader("Вектор свободных членов b")
    b_input = st.text_input("Свободные члены (через пробел)",
                            " ".join(["1"] * n_equations),
                            key="b_vec")
    try:
        b = np.array([float(x) for x in b_input.split()])
        if len(b) != n_equations:
            st.error(f"Ожидается {n_equations} элементов, получено {len(b)}")
            b = None
        else:
            st.write("**Вектор b:**", b)
    except ValueError:
        st.error("Ошибка в векторе b")
        b = None

    if st.button("Решить систему", type="primary", width='stretch'):
        if A is None or b is None:
            st.error("Ошибка в данных системы")
        else:
            try:
                st.write("**Матрица A:**")
                st.dataframe(pd.DataFrame(A), width='stretch')
                st.write("**Вектор b:**", b)

                det = np.linalg.det(A)
                st.info(f"**Определитель матрицы A:** {det:.8f}")

                if abs(det) > 1e-10:
                    x = np.linalg.solve(A, b)

                    st.success("**Решение системы:**")
                    for i, val in enumerate(x):
                        st.write(f"**x_{i+1}** = {val:.8f}")

                    st.info("**Проверка решения (Ax):**")
                    verification = A @ x
                    check_data = {
                        'Ожидалось (b)': [f"{val:.8f}" for val in b],
                        'Вычислено (A×x)':
                        [f"{val:.8f}" for val in verification],
                        'Невязка':
                        [f"{val:.2e}" for val in np.abs(verification - b)]
                    }
                    st.dataframe(pd.DataFrame(check_data), width='stretch')
                    max_error = np.max(np.abs(verification - b))
                    st.write(f"**Максимальная невязка:** {max_error:.2e}")

                    save_history(f"Система {n_equations} уравнений",
                                 f"Решение: {x}", f"Определитель: {det:.8f}")
                else:
                    st.warning(
                        "**Система вырождена или имеет бесконечно много решений**"
                    )

                    st.write(
                        "**Попытка найти решение методом наименьших квадратов:**"
                    )
                    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

                    st.write("**Приближенное решение:**")
                    for i, val in enumerate(x):
                        st.write(f"**x_{i+1}** = {val:.8f}")

                    st.write(
                        f"**Невязка:** {residuals[0] if len(residuals) > 0 else 'N/A':.8f}"
                    )
                    st.write(f"**Ранг матрицы:** {rank}")

            except Exception as e:
                st.error(f"❌ Ошибка решения системы: {str(e)}")

with tabs[7]:
    st.header("♾️ Символьные вычисления")

    sym_operation = st.selectbox("Выберите операцию", [
        "Упрощение", "Разложение на множители", "Раскрытие скобок",
        "Производная", "Интеграл", "Предел", "Ряд Тейлора",
        "Решение уравнения", "Дифференциальные уравнения"
    ],
                                 key="sym_op")

    expr_input = st.text_input("Введите выражение",
                               value="(x**2 - 1)/(x - 1)" if sym_operation
                               == "Упрощение" else "x**2 + 2*x + 1",
                               key="sym_expr")

    if sym_operation == "Производная":
        derivative_order = st.number_input("Порядок производной",
                                           1,
                                           5,
                                           1,
                                           key="deriv_order")

    if sym_operation == "Интеграл":
        integral_type = st.radio("Тип интеграла",
                                 ["Неопределенный", "Определенный"],
                                 key="integral_type",
                                 horizontal=True)
        if integral_type == "Определенный":
            col1, col2 = st.columns(2)
            with col1:
                a_int = st.number_input("Нижний предел",
                                        value=0.0,
                                        key="a_int")
            with col2:
                b_int = st.number_input("Верхний предел",
                                        value=1.0,
                                        key="b_int")

    if sym_operation == "Предел":
        col1, col2 = st.columns(2)
        with col1:
            limit_point = st.text_input("Точка", "0", key="limit_point")
        with col2:
            limit_dir = st.selectbox("Направление", ["±", "+", "-"],
                                     key="limit_dir")

    if sym_operation == "Ряд Тейлора":
        col1, col2 = st.columns(2)
        with col1:
            taylor_point = st.number_input("Точка разложения",
                                           value=0.0,
                                           key="taylor_point")
        with col2:
            taylor_order = st.number_input("Порядок",
                                           1,
                                           10,
                                           5,
                                           key="taylor_order")

    if st.button("🔍 Выполнить символьное вычисление",
                 type="primary",
                 width='stretch'):
        try:
            x = sp.Symbol('x')
            expr = sp.sympify(expr_input)

            if sym_operation == "Упрощение":
                result = sp.simplify(expr)
                st.success("**Упрощенное выражение:**")
                st.latex(sp.latex(result))
                st.code(f"Python: {result}")

            elif sym_operation == "Разложение на множители":
                result = sp.factor(expr)
                st.success("**Разложение на множители:**")
                st.latex(sp.latex(result))
                st.code(f"Python: {result}")

            elif sym_operation == "Раскрытие скобок":
                result = sp.expand(expr)
                st.success("**Раскрытие скобок:**")
                st.latex(sp.latex(result))
                st.code(f"Python: {result}")

            elif sym_operation == "Производная":
                result = sp.diff(expr, x, derivative_order)
                st.success(f"**Производная {derivative_order}-го порядка:**")
                st.latex(sp.latex(result))
                st.code(f"Python: {result}")

            elif sym_operation == "Интеграл":
                if integral_type == "Определенный":
                    result = sp.integrate(expr, (x, a_int, b_int))
                    st.success(
                        f"**Определенный интеграл от {a_int} до {b_int}:**")
                    st.latex(sp.latex(result))
                    numeric_result = float(result.evalf())
                    st.write(f"**Численное значение:** {numeric_result:.8f}")
                else:
                    result = sp.integrate(expr, x)
                    st.success("**Неопределенный интеграл:**")
                    st.latex(sp.latex(result))
                st.code(f"Python: {result}")

            elif sym_operation == "Предел":
                point_val = sp.oo if limit_point == "oo" else (
                    -sp.oo if limit_point == "-oo" else float(limit_point))

                if limit_dir == "±":
                    result = sp.limit(expr, x, point_val)
                else:
                    result = sp.limit(expr, x, point_val, limit_dir)

                st.success(f"**Предел при x → {limit_point} ({limit_dir}):**")
                st.latex(sp.latex(result))
                st.code(f"Python: {result}")

            elif sym_operation == "Ряд Тейлора":
                result = sp.series(expr, x, taylor_point, taylor_order)
                st.success(
                    f"**Ряд Тейлора в точке {taylor_point} до порядка {taylor_order}:**"
                )
                st.latex(sp.latex(result))
                st.code(f"Python: {result}")

            elif sym_operation == "Решение уравнения":
                st.info("**Решаем уравнение: выражение = 0**")
                solutions = sp.solve(expr, x)
                result = solutions
                st.success(f"**Найдено решений: {len(solutions)}**")
                for i, sol in enumerate(solutions, 1):
                    st.write(f"**x_{i}** =")
                    st.latex(sp.latex(sol))
                    numeric_sol = complex(sol.evalf())
                    if abs(numeric_sol.imag) < 1e-10:
                        st.write(f"Численное значение: {numeric_sol.real:.8f}")
                    else:
                        st.write(
                            f"Численное значение: {numeric_sol.real:.8f} + {numeric_sol.imag:.8f}i"
                        )
                    st.code(f"Python: {sol}")

            elif sym_operation == "Дифференциальные уравнения":
                st.info(
                    "**Решение обыкновенного дифференциального уравнения**")
                st.caption(
                    "Используйте y(x) как функцию и Derivative(y(x), x) для производной"
                )
                st.caption("Пример: Derivative(y(x), x) - y(x)")
                try:
                    y = sp.Function('y')
                    eq = sp.sympify(expr_input)

                    if y(x) in eq.free_symbols or any(
                            isinstance(arg, sp.Derivative)
                            for arg in eq.atoms()):
                        solution = sp.dsolve(eq, y(x))
                        result = solution
                        st.success("**Решение ДУ:**")
                        st.latex(sp.latex(solution))
                        st.code(f"Python: {solution}")
                    else:
                        result = "Не является дифференциальным уравнением"
                        st.error(
                            "❌ Выражение не содержит дифференциальное уравнение"
                        )
                        st.info(
                            "💡 Используйте формат: `Derivative(y(x), x) + y(x)` или `y(x).diff(x) - y(x)`"
                        )
                except Exception as e:
                    result = f"Ошибка: {str(e)}"
                    st.error(f"❌ Не удалось решить: {str(e)}")
                    st.info(
                        "💡 Примеры:\n- `Derivative(y(x), x) - y(x)` (y' = y)\n- `Derivative(y(x), x, x) + y(x)` (y'' + y = 0)"
                    )

            save_history(f"Символьное: {sym_operation}", f"{expr_input}",
                         f"Результат: {str(result)[:100]}")

        except Exception as e:
            st.error(f"❌ Ошибка символьного вычисления: {str(e)}")

with tabs[8]:
    st.header("⚡ Численная оптимизация")

    opt_type = st.selectbox("Тип оптимизации", [
        "Минимум одномерной функции", "Минимум многомерной функции",
        "Подбор параметров кривой"
    ],
                            key="opt_type")

    if opt_type == "Минимум одномерной функции":
        func_opt = st.text_input("Функция f(x)",
                                 "x**4 - 3*x**3 + 2",
                                 key="func_1d_opt")

        col1, col2 = st.columns(2)
        with col1:
            x_start = st.number_input("Начальная точка",
                                      -10.0,
                                      10.0,
                                      0.0,
                                      key="x0_1d")
        with col2:
            method_1d = st.selectbox("Метод",
                                     ["BFGS", "Nelder-Mead", "Powell", "CG"],
                                     key="method_1d")

        if st.button("Найти минимум", type="primary", width='stretch'):
            try:
                x_sym = sp.Symbol('x')
                expr = sp.sympify(func_opt)
                f_lambda = sp.lambdify(x_sym, expr, 'numpy')

                result = optimize.minimize(f_lambda, x_start, method=method_1d)

                st.success("**Результат оптимизации:**")
                st.write(f"**x_min** = {result.x[0]:.8f}")
                st.write(f"**f(x_min)** = {result.fun:.8f}")
                st.write(
                    f"**Итераций:** {result.nit if hasattr(result, 'nit') else 'N/A'}"
                )
                st.write(f"**Успех:** {result.success}")

                if hasattr(result, 'jac') and result.jac is not None:
                    st.write(f"**Градиент в минимуме:** {result.jac[0]:.2e}")

                x_plot = np.linspace(result.x[0] - 3, result.x[0] + 3, 300)
                y_plot = f_lambda(x_plot)

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(x=x_plot, y=y_plot, mode='lines', name='f(x)'))
                fig.add_trace(
                    go.Scatter(x=[result.x[0]],
                               y=[result.fun],
                               mode='markers',
                               marker=dict(size=15, color='red'),
                               name='Минимум'))
                fig.update_layout(
                    title=f"График функции с найденным минимумом: {func_opt}",
                    xaxis_title="x",
                    yaxis_title="f(x)",
                    template=plotly_template)
                st.plotly_chart(fig,
                                use_container_width=True,
                                config={'displayModeBar': True})

                save_history(f"Оптимизация 1D: {func_opt}",
                             f"x_min={result.x[0]:.8f}",
                             f"f_min={result.fun:.8f}")

            except Exception as e:
                st.error(f"❌ Ошибка оптимизации: {str(e)}")

    elif opt_type == "Минимум многомерной функции":
        st.subheader("Минимизация функции f(x₁, x₂, ..., xₙ)")

        func_multi = st.text_input("Функция f(x)",
                                   "x[0]**2 + x[1]**2 + x[0]*x[1]",
                                   key="func_multi")
        n_vars = st.number_input("Количество переменных",
                                 2,
                                 5,
                                 2,
                                 key="n_vars")

        st.write("Начальное приближение (через запятую):")
        initial_guess = st.text_input("Начальная точка",
                                      "1.0, 1.0",
                                      key="x0_multi")

        method_multi = st.selectbox(
            "Метод оптимизации",
            ["BFGS", "Nelder-Mead", "Powell", "CG", "L-BFGS-B"],
            key="method_multi")

        if st.button("Найти многомерный минимум",
                     type="primary",
                     width='stretch'):
            try:

                def objective_function(x):
                    return eval(func_multi, {"x": x, "np": np, "math": math})

                x0 = np.array(
                    [float(v.strip()) for v in initial_guess.split(',')])
                if len(x0) != n_vars:
                    st.error(
                        f"Ожидается {n_vars} начальных значений, получено {len(x0)}"
                    )
                else:
                    result = optimize.minimize(objective_function,
                                               x0,
                                               method=method_multi)

                    st.success("**Результат оптимизации:**")
                    st.write(
                        f"**x_min** = [{', '.join(f'{v:.8f}' for v in result.x)}]"
                    )
                    st.write(f"**f(x_min)** = {result.fun:.8f}")
                    st.write(f"**Успех:** {result.success}")

                    if n_vars == 2:
                        st.subheader("Визуализация функции двух переменных")

                        x1 = np.linspace(result.x[0] - 2, result.x[0] + 2, 100)
                        x2 = np.linspace(result.x[1] - 2, result.x[1] + 2, 100)
                        X1, X2 = np.meshgrid(x1, x2)

                        Z = np.zeros_like(X1)
                        for i in range(X1.shape[0]):
                            for j in range(X1.shape[1]):
                                Z[i,
                                  j] = objective_function([X1[i, j], X2[i, j]])

                        fig = go.Figure(data=[
                            go.Contour(x=x1,
                                       y=x2,
                                       z=Z,
                                       colorscale='Viridis',
                                       contours=dict(
                                           coloring='heatmap',
                                           showlabels=True,
                                       ),
                                       colorbar=dict(title='f(x₁,x₂)'))
                        ])

                        fig.add_trace(
                            go.Scatter(x=[result.x[0]],
                                       y=[result.x[1]],
                                       mode='markers',
                                       marker=dict(size=12,
                                                   color='red',
                                                   line=dict(width=2,
                                                             color='white')),
                                       name='Минимум'))

                        fig.update_layout(title="Контурный график функции",
                                          xaxis_title='x₁',
                                          yaxis_title='x₂',
                                          template=plotly_template)

                        st.plotly_chart(fig,
                                        use_container_width=True,
                                        config={'displayModeBar': True})

                    save_history(f"Многомерная оптимизация",
                                 f"x_min={result.x}",
                                 f"f_min={result.fun:.8f}")

            except Exception as e:
                st.error(f"❌ Ошибка оптимизации: {str(e)}")

    elif opt_type == "Подбор параметров кривой":
        st.subheader("Аппроксимация данных полиномом")

        data_points = st.text_area(
            "Введите точки (x,y) по одной в строке, разделенные запятой:",
            "0,1\n1,2\n2,5\n3,10\n4,17\n5,26",
            height=150,
            key="fit_data")

        degree = st.slider("Степень полинома", 1, 10, 2, key="poly_degree")

        if st.button("📈 Подобрать кривую", type="primary", width='stretch'):
            try:
                points = [
                    line.split(',') for line in data_points.strip().split('\n')
                    if line.strip()
                ]
                x_data = np.array([float(p[0].strip()) for p in points])
                y_data = np.array([float(p[1].strip()) for p in points])

                coeffs = np.polyfit(x_data, y_data, degree)
                poly = np.poly1d(coeffs)

                x_smooth = np.linspace(x_data.min(), x_data.max(), 300)
                y_smooth = poly(x_smooth)

                fig = go.Figure()

                fig.add_trace(
                    go.Scatter(x=x_data,
                               y=y_data,
                               mode='markers',
                               marker=dict(size=8, color='blue'),
                               name='Исходные данные'))

                fig.add_trace(
                    go.Scatter(x=x_smooth,
                               y=y_smooth,
                               mode='lines',
                               line=dict(color='red', width=3),
                               name=f'Полином {degree} степени'))

                fig.update_layout(title="Аппроксимация данных полиномом",
                                  xaxis_title="x",
                                  yaxis_title="y",
                                  template=plotly_template)

                st.plotly_chart(fig,
                                use_container_width=True,
                                config={'displayModeBar': True})

                st.success("**Коэффициенты полинома:**")
                st.code(str(poly))

                mse = np.mean((y_data - poly(x_data))**2)
                r_squared = 1 - np.sum((y_data - poly(x_data))**2) / np.sum(
                    (y_data - np.mean(y_data))**2)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Среднеквадратичная ошибка", f"{mse:.6f}")
                with col2:
                    st.metric("R² коэффициент", f"{r_squared:.6f}")
                with col3:
                    st.metric("Макс. отклонение",
                              f"{np.max(np.abs(y_data - poly(x_data))):.6f}")

                save_history(f"Аппроксимация полиномом степени {degree}",
                             f"MSE={mse:.6f}", f"R²={r_squared:.6f}")

            except Exception as e:
                st.error(f"❌ Ошибка аппроксимации: {str(e)}")

with tabs[9]:
    st.header("📊 Анализ данных и сигналов")

    analysis_type = st.selectbox("Тип анализа", [
        "Преобразование Фурье",
        "Корреляционный анализ",
    ],
                                 key="analysis_type")

    if analysis_type == "Преобразование Фурье":
        st.subheader("Анализ Фурье - Быстрое преобразование Фурье (FFT)")

        signal_type = st.selectbox("Тип сигнала", [
            "Синусоидальный", "Сумма синусоид", "Прямоугольный",
            "Пилообразный", "Случайный"
        ],
                                   key="signal_type")

        if signal_type == "Синусоидальный":
            col1, col2 = st.columns(2)
            with col1:
                freq = st.slider("Основная частота (Гц)",
                                 1.0,
                                 100.0,
                                 10.0,
                                 key="freq_sine")
            with col2:
                amplitude = st.slider("Амплитуда",
                                      0.1,
                                      10.0,
                                      1.0,
                                      key="amp_sine")

            t = np.linspace(0, 1, 1000)
            signal = amplitude * np.sin(2 * np.pi * freq * t)

        elif signal_type == "Сумма синусоид":
            col1, col2, col3 = st.columns(3)
            with col1:
                f1 = st.slider("Частота 1 (Гц)", 1.0, 50.0, 5.0, key="f1")
                a1 = st.slider("Амплитуда 1", 0.1, 5.0, 1.0, key="a1")
            with col2:
                f2 = st.slider("Частота 2 (Гц)", 1.0, 50.0, 20.0, key="f2")
                a2 = st.slider("Амплитуда 2", 0.1, 5.0, 0.5, key="a2")
            with col3:
                f3 = st.slider("Частота 3 (Гц)", 1.0, 50.0, 35.0, key="f3")
                a3 = st.slider("Амплитуда 3", 0.1, 5.0, 0.3, key="a3")

            t = np.linspace(0, 1, 1000)
            signal = (a1 * np.sin(2 * np.pi * f1 * t) +
                      a2 * np.sin(2 * np.pi * f2 * t) +
                      a3 * np.sin(2 * np.pi * f3 * t))

        elif signal_type == "Прямоугольный":
            freq = st.slider("Частота (Гц)", 1.0, 50.0, 5.0, key="freq_square")
            t = np.linspace(0, 1, 1000)
            signal = 0.5 * (1 + np.sign(np.sin(2 * np.pi * freq * t)))

        elif signal_type == "Пилообразный":
            freq = st.slider("Частота (Гц)", 1.0, 50.0, 5.0, key="freq_saw")
            t = np.linspace(0, 1, 1000)
            signal = 2 * (t * freq - np.floor(0.5 + t * freq))

        else:
            noise_level = st.slider("Уровень шума",
                                    0.0,
                                    2.0,
                                    0.5,
                                    key="noise_level")
            t = np.linspace(0, 1, 1000)
            signal = np.random.normal(0, noise_level, len(t))

        if st.button("🔍 Выполнить анализ Фурье",
                     type="primary",
                     width='stretch'):
            fft_result = fft.fft(signal)
            freqs = fft.fftfreq(len(signal), t[1] - t[0])

            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=("Исходный сигнал", "Амплитудный спектр",
                                "Фазовый спектр", "Спектр мощности"),
                specs=[[{
                    "colspan": 2
                }, None], [{}, {}]])

            fig.add_trace(go.Scatter(x=t,
                                     y=signal,
                                     mode='lines',
                                     name='Сигнал'),
                          row=1,
                          col=1)

            mask = freqs > 0
            fig.add_trace(go.Scatter(x=freqs[mask],
                                     y=np.abs(fft_result[mask]),
                                     mode='lines',
                                     name='Амплитуда'),
                          row=2,
                          col=1)

            fig.add_trace(go.Scatter(x=freqs[mask],
                                     y=np.angle(fft_result[mask]),
                                     mode='lines',
                                     name='Фаза'),
                          row=2,
                          col=2)

            fig.update_xaxes(title_text="Время (с)", row=1, col=1)
            fig.update_xaxes(title_text="Частота (Гц)", row=2, col=1)
            fig.update_xaxes(title_text="Частота (Гц)", row=2, col=2)

            fig.update_yaxes(title_text="Амплитуда", row=1, col=1)
            fig.update_yaxes(title_text="Амплитуда", row=2, col=1)
            fig.update_yaxes(title_text="Фаза (рад)", row=2, col=2)

            fig.update_layout(height=700,
                              template=plotly_template,
                              showlegend=False)
            st.plotly_chart(fig,
                            use_container_width=True,
                            config={'displayModeBar': True})

            dominant_freq = freqs[mask][np.argmax(np.abs(fft_result[mask]))]
            st.success(f"**Доминирующая частота:** {dominant_freq:.2f} Гц")

            save_history("Анализ Фурье",
                         f"Доминирующая частота: {dominant_freq:.2f} Гц",
                         f"Тип сигнала: {signal_type}")

    elif analysis_type == "Корреляционный анализ":
        st.subheader("📈 Корреляция и ковариация")

        n_points = st.slider("Количество точек данных",
                             20,
                             500,
                             100,
                             key="n_corr_points")

        col1, col2 = st.columns(2)
        with col1:
            target_correlation = st.slider("Целевая корреляция",
                                           -1.0,
                                           1.0,
                                           0.7,
                                           key="target_corr")
        with col2:
            noise_level = st.slider("Уровень шума",
                                    0.0,
                                    1.0,
                                    0.2,
                                    key="corr_noise")

        if st.button("📊 Сгенерировать и проанализировать данные",
                     type="primary",
                     width='stretch'):
            mean = [0, 0]
            cov = [[1, target_correlation], [target_correlation, 1]]

            data = np.random.multivariate_normal(mean, cov, n_points)
            data += np.random.normal(0, noise_level, data.shape)

            x_data, y_data = data[:, 0], data[:, 1]

            actual_correlation = np.corrcoef(x_data, y_data)[0, 1]
            actual_covariance = np.cov(x_data, y_data)[0, 1]

            fig = px.scatter(
                x=x_data,
                y=y_data,
                title=
                f"Диаграмма рассеяния (корреляция: {actual_correlation:.4f})",
                labels={
                    'x': 'X',
                    'y': 'Y'
                })

            z = np.polyfit(x_data, y_data, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x_data.min(), x_data.max(), 100)

            fig.add_trace(
                go.Scatter(x=x_line,
                           y=p(x_line),
                           mode='lines',
                           name='Линия регрессии',
                           line=dict(color='red', width=2)))

            fig.update_layout(template=plotly_template, height=500)
            st.plotly_chart(fig,
                            use_container_width=True,
                            config={'displayModeBar': True})

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Корреляция", f"{actual_correlation:.4f}")
            with col2:
                st.metric("Ковариация", f"{actual_covariance:.4f}")
            with col3:
                st.metric("R²", f"{actual_correlation**2:.4f}")
            with col4:
                st.metric("Наклон", f"{z[0]:.4f}")

            save_history("Корреляционный анализ",
                         f"r={actual_correlation:.4f}",
                         f"R²={actual_correlation**2:.4f}")

with tabs[10]:
    st.header("🧮 Работа с дробями и целыми числами")

    fraction_mode = st.selectbox("Режим работы с дробями", [
        "Вычисления с дробями", "Преобразование дробей", "Арифметика НОД/НОК",
        "Операции с целыми числами", "Системы счисления"
    ],
                                 key="fraction_mode")

    if fraction_mode == "Вычисления с дробями":
        st.subheader("🧮 Калькулятор дробей")

        fraction_expr = st.text_input("Дробное выражение",
                                      "1/2 + 1/3 + 1/6",
                                      key="frac_expr")

        if st.button("Вычислить дробь", type="primary", width='stretch'):
            try:
                result = fraction_engine.evaluate_fraction(fraction_expr)
                st.success(f"**Результат:** {result}")

                decimal_result = parser.evaluate_expression(fraction_expr)
                if abs(decimal_result.imag) < 1e-12:
                    st.info(
                        f"**Десятичное представление:** {decimal_result.real:.10f}"
                    )

                save_history(f"Дробное выражение: {fraction_expr}", result)

            except Exception as e:
                st.error(f"Ошибка: {str(e)}")

    elif fraction_mode == "Преобразование дробей":
        st.subheader("Преобразование между дробями и десятичными числами")

        col1, col2 = st.columns(2)

        with col1:
            decimal_input = st.number_input("Десятичное число",
                                            -1000.0,
                                            1000.0,
                                            0.375,
                                            key="decimal_input")
            if st.button("→ Дробь", key="to_frac"):
                fraction_result = fraction_engine.to_fraction(decimal_input)
                mixed_result = fraction_engine.mixed_fraction(decimal_input)

                st.success(f"**Обычная дробь:** {fraction_result}")
                if mixed_result != fraction_result:
                    st.success(f"**Смешанная дробь:** {mixed_result}")

        with col2:
            fraction_input = st.text_input("Дробь (a/b)",
                                           "3/8",
                                           key="frac_input")
            if st.button("→ Десятичное", key="to_decimal"):
                try:
                    parts = fraction_input.split('/')
                    if len(parts) == 2:
                        numerator = float(parts[0])
                        denominator = float(parts[1])
                        if denominator == 0:
                            st.error("Знаменатель не может быть нулем")
                        else:
                            decimal_value = numerator / denominator
                            st.success(
                                f"**Десятичное число:** {decimal_value:.10f}")
                    else:
                        st.error("Введите дробь в формате a/b")
                except:
                    st.error("Ошибка в формате дроби")

    elif fraction_mode == "Арифметика НОД/НОК":
        st.subheader(
            "Наибольший общий делитель (НОД) и наименьшее общее кратное (НОК)")
        numbers_input = st.text_input(
            "Введите целые числа (через пробел или запятую)",
            "12 18 24",
            key="gcd_lcm_input")

        if st.button("Вычислить НОД и НОК", type="primary", width='stretch'):
            try:
                raw_numbers = [
                    int(x.strip())
                    for x in numbers_input.replace(',', ' ').split()
                ]
            except ValueError:
                st.error("Некорректный ввод: введите только целые числа")
                raw_numbers = None

            if raw_numbers is not None and len(raw_numbers) > 0:
                has_zero = any(n == 0 for n in raw_numbers)
                non_zero_numbers = [n for n in raw_numbers if n != 0]

                if not non_zero_numbers:
                    st.error("Все числа — нули. НОД(0,0,...) не определён.")
                else:
                    abs_numbers = [abs(n) for n in non_zero_numbers]

                    gcd_val = abs_numbers[0]
                    for n in abs_numbers[1:]:
                        gcd_val = math_gcd(gcd_val, n)

                    if has_zero:
                        final_gcd = gcd_val
                        st.info(
                            "Наличие нуля не влияет на НОД: НОД(a, 0) = |a|")
                    else:
                        final_gcd = gcd_val

                    if has_zero:
                        lcm_result = "не определён (содержит ноль)"
                        st.warning(
                            "НОК не определён, так как одно из чисел — ноль.")
                    else:
                        lcm_val = abs_numbers[0]
                        for n in abs_numbers[1:]:
                            lcm_val = lcm_val * n // math_gcd(lcm_val, n)
                        lcm_result = str(lcm_val)

                    input_str = ', '.join(map(str, raw_numbers))
                    st.success(f"**НОД({input_str}) = {final_gcd}**")
                    st.success(f"**НОК({input_str}) = {lcm_result}**")

                    st.info("**Разложения на простые множители:**")
                    for num in abs_numbers[:5]:
                        if num == 1:
                            st.write("1 = 1 (по определению)")
                        else:
                            factors = []
                            n = num
                            d = 2
                            while d * d <= n:
                                while n % d == 0:
                                    factors.append(d)
                                    n //= d
                                d += 1
                            if n > 1:
                                factors.append(n)
                            st.write(
                                f"{num} = {' × '.join(map(str, factors))}")

                    save_history(f"НОД/НОК чисел: {raw_numbers}",
                                 f"НОД={final_gcd}, НОК={lcm_result}")
            else:
                st.warning("Введите хотя бы одно число")

    elif fraction_mode == "Операции с целыми числами":
        st.subheader("🔢 Операции с целыми числами")

        integer_input = st.number_input("Целое число",
                                        -10**6,
                                        10**6,
                                        100,
                                        key="integer_input")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Анализ числа", key="analyze_int"):
                n = int(integer_input)

                def is_prime(n):
                    if n < 2:
                        return False
                    for i in range(2, int(n**0.5) + 1):
                        if n % i == 0:
                            return False
                    return True

                st.success(
                    f"**{n} - {'простое' if is_prime(n) else 'составное'} число**"
                )

                if n > 1:
                    factors = []
                    temp = abs(n)
                    divisor = 2
                    while divisor * divisor <= temp:
                        while temp % divisor == 0:
                            factors.append(divisor)
                            temp //= divisor
                        divisor += 1
                    if temp > 1:
                        factors.append(temp)

                    st.write(
                        f"**Разложение на множители:** {' × '.join(map(str, factors))}"
                    )

                st.write(f"**Количество цифр:** {len(str(abs(n)))}")
                st.write(
                    f"**Четность:** {'четное' if n % 2 == 0 else 'нечетное'}")

        with col2:
            if st.button("Системы счисления", key="base_convert"):
                n = int(integer_input)
                st.success(f"**Системы счисления для {n}:**")
                st.write(f"**Двоичная:** {bin(n)}")
                st.write(f"**Восьмеричная:** {oct(n)}")
                st.write(f"**Шестнадцатеричная:** {hex(n)}")

    elif fraction_mode == "Системы счисления":
        st.subheader("Конвертер систем счисления")

        col1, col2 = st.columns(2)

        with col1:
            number_input = st.text_input("Число", "255", key="base_number")
            from_base = st.selectbox("Из системы", [2, 8, 10, 16],
                                     index=2,
                                     key="from_base")

        with col2:
            to_bases = st.multiselect("В системы", [2, 8, 10, 16],
                                      default=[2, 8, 16],
                                      key="to_bases")

        if st.button("Конвертировать", type="primary", width='stretch'):
            try:
                decimal_num = int(number_input, from_base)

                st.success(f"**Десятичное представление:** {decimal_num}")

                for base in to_bases:
                    if base == 2:
                        result = bin(decimal_num)
                    elif base == 8:
                        result = oct(decimal_num)
                    elif base == 16:
                        result = hex(decimal_num)
                    else:
                        result = str(decimal_num)

                    st.write(f"**Основание {base}:** {result}")

                save_history(
                    f"Конвертация систем счисления",
                    f"{number_input}₍{from_base}₎ = {decimal_num}₍₁₀₎",
                    f"В системы: {to_bases}")

            except Exception as e:
                st.error(f"Ошибка конвертации: {str(e)}")

with tabs[11]:
    st.header("🎲 Генерация случайных данных")

    st.subheader("Генератор распределений")

    col1, col2 = st.columns(2)
    with col1:
        dist_type = st.selectbox(
            "Распределение", ["Нормальное", "Равномерное", "Экспоненциальное"],
            key="gen_dist")
        n_samples = st.number_input("Количество samples", 10, 10000, 1000)

        if dist_type == "Нормальное":
            gen_mu = st.number_input("Среднее", -10.0, 10.0, 0.0, key="gen_mu")
            gen_sigma = st.number_input("Ст. отклонение",
                                        0.1,
                                        10.0,
                                        1.0,
                                        key="gen_sigma")
        elif dist_type == "Равномерное":
            a = st.number_input("a (минимум)", -10.0, 10.0, 0.0)
            b = st.number_input("b (максимум)", -10.0, 10.0, 1.0)
        else:
            scale = st.number_input("β (масштаб)", 0.1, 10.0, 1.0)

    with col2:
        if st.button("Сгенерировать данные", key="gen_data", type="primary"):
            np.random.seed(42)

            if dist_type == "Нормальное":
                data = np.random.normal(gen_mu, gen_sigma, n_samples)
            elif dist_type == "Равномерное":
                data = np.random.uniform(a, b, n_samples)
            else:
                data = np.random.exponential(scale, n_samples)

            st.session_state.generated_data = data
            st.success(f"Сгенерировано {n_samples} samples")

            fig = px.histogram(data,
                               nbins=50,
                               title=f"Гистограмма {dist_type} распределения")
            st.plotly_chart(fig, use_container_width=True)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Среднее", f"{np.mean(data):.4f}")
            col2.metric("Медиана", f"{np.median(data):.4f}")
            col3.metric("Ст. отклонение", f"{np.std(data):.4f}")
            col4.metric("Дисперсия", f"{np.var(data):.4f}")

            save_history(f"Генерация данных: {dist_type}", f"n={n_samples}",
                         f"μ={np.mean(data):.4f}, σ={np.std(data):.4f}")

with tabs[12]:
    st.header("📈 Дифференциальные уравнения и численные методы")

    ode_num_tabs = st.tabs(["Диффуры", "Численные методы", "Интегрирование"])

    with ode_num_tabs[0]:
        st.subheader("ОДУ первого порядка: y' = f(x, y)")

        ode_input = st.text_input("Правая часть f(x, y)",
                                  "y - x",
                                  key="ode_func")
        col1, col2, col3 = st.columns(3)

        with col1:
            x0 = st.number_input("x₀", -10.0, 10.0, 0.0)
            y0 = st.number_input("y₀", -10.0, 10.0, 1.0)
        with col2:
            x_end = st.number_input("Конечный x", -10.0, 20.0, 5.0)
            n_points = st.slider("Точек", 100, 5000, 1000)
        with col3:
            method = st.selectbox("Метод",
                                  ["Рунге-Кутта 4", "Эйлера", "Эйлера-Коши"],
                                  key="ode_method")

        if st.button("Решить ОДУ", key="solve_ode"):
            try:

                def ode_func(x, y):
                    return eval(ode_input, {
                        "x": x,
                        "y": y,
                        "np": np,
                        "math": math
                    })

                x_span = np.linspace(x0, x_end, n_points)

                if method == "Рунге-Кутта 4":
                    solution = integrate.solve_ivp(ode_func, [x0, x_end], [y0],
                                                   t_eval=x_span,
                                                   method='RK45')
                    success = solution.success
                    t_vals = solution.t
                    y_vals = solution.y[0]

                elif method == "Эйлера":
                    h = (x_end - x0) / n_points
                    t_vals = [x0]
                    y_vals = [y0]
                    for i in range(n_points):
                        x_new = t_vals[-1] + h
                        y_new = y_vals[-1] + h * ode_func(
                            t_vals[-1], y_vals[-1])
                        t_vals.append(x_new)
                        y_vals.append(y_new)
                    success = True
                    t_vals = np.array(t_vals)
                    y_vals = np.array(y_vals)

                else:
                    solution = integrate.solve_ivp(ode_func, [x0, x_end], [y0],
                                                   t_eval=x_span,
                                                   method='RK23')
                    success = solution.success
                    t_vals = solution.t
                    y_vals = solution.y[0]

                if success:
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(x=t_vals,
                                   y=y_vals,
                                   mode='lines',
                                   name='Решение'))
                    fig.update_layout(
                        title=f"Решение: y' = {ode_input}, y({x0}) = {y0}",
                        xaxis_title="x",
                        yaxis_title="y")
                    st.plotly_chart(fig, use_container_width=True)

                    st.subheader("Поле направлений")
                    x = np.linspace(x0 - 1, x_end + 1, 20)
                    y = np.linspace(min(y_vals) - 1, max(y_vals) + 1, 20)
                    X, Y = np.meshgrid(x, y)

                    U = np.ones_like(X)
                    V = np.array([[
                        ode_func(x_val, y_val)
                        for x_val, y_val in zip(x_row, y_row)
                    ] for x_row, y_row in zip(X, Y)])

                    norm = np.sqrt(U**2 + V**2)
                    U = U / norm
                    V = V / norm

                    fig2 = go.Figure()
                    fig2.add_trace(
                        go.Scatter(x=t_vals,
                                   y=y_vals,
                                   mode='lines',
                                   name='Решение',
                                   line=dict(width=3)))
                    fig2.add_trace(
                        go.Cone(x=X.flatten(),
                                y=Y.flatten(),
                                u=U.flatten(),
                                v=V.flatten(),
                                colorscale='Blues',
                                showscale=False,
                                sizemode="absolute",
                                sizeref=0.3))
                    fig2.update_layout(title="Поле направлений и решение",
                                       xaxis_title="x",
                                       yaxis_title="y")
                    st.plotly_chart(fig2, use_container_width=True)

                else:
                    st.error("Не удалось решить ОДУ")

            except Exception as e:
                st.error(f"Ошибка: {e}")

    with ode_num_tabs[1]:
        st.subheader("Интерполяция данных")

        st.write(
            "Введите точки (x,y) через запятую, каждая точка с новой строки:")
        points_input = st.text_area("Точки",
                                    "1, 1\n2, 4\n3, 9\n4, 16\n5, 25",
                                    height=150,
                                    key="interp_points")

        col1, col2 = st.columns(2)
        with col1:
            method = st.selectbox(
                "Метод интерполяции",
                ["Линейная", "Полином Лагранжа", "Кубические сплайны"],
                key="interp_method")
            n_interp = st.slider("Точек интерполяции",
                                 10,
                                 1000,
                                 100,
                                 key="n_interp_points")

        with col2:
            if st.button("Выполнить интерполяцию",
                         key="do_interp",
                         type="primary"):
                try:
                    points = [
                        line.split(',')
                        for line in points_input.strip().split('\n')
                        if line.strip()
                    ]
                    x_data = np.array([float(p[0].strip()) for p in points])
                    y_data = np.array([float(p[1].strip()) for p in points])

                    x_interp = np.linspace(min(x_data), max(x_data), n_interp)

                    if method == "Линейная":
                        y_interp = np.interp(x_interp, x_data, y_data)
                    elif method == "Полином Лагранжа":

                        def lagrange_poly(x, x_data, y_data):
                            total = 0
                            n = len(x_data)
                            for i in range(n):
                                xi, yi = x_data[i], y_data[i]
                                product = 1
                                for j in range(n):
                                    if i != j:
                                        product *= (x - x_data[j]) / (
                                            xi - x_data[j])
                                total += yi * product
                            return total

                        y_interp = [
                            lagrange_poly(xi, x_data, y_data)
                            for xi in x_interp
                        ]
                    else:
                        from scipy.interpolate import CubicSpline
                        cs = CubicSpline(x_data, y_data)
                        y_interp = cs(x_interp)

                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(x=x_data,
                                   y=y_data,
                                   mode='markers',
                                   name='Исходные точки',
                                   marker=dict(size=10)))
                    fig.add_trace(
                        go.Scatter(x=x_interp,
                                   y=y_interp,
                                   mode='lines',
                                   name=f'Интерполяция ({method})',
                                   line=dict(width=3)))
                    fig.update_layout(title=f"Интерполяция: {method}",
                                      xaxis_title="x",
                                      yaxis_title="y",
                                      template=plotly_template)
                    st.plotly_chart(fig, use_container_width=True)

                    save_history(
                        f"Интерполяция: {method}", f"{len(x_data)} точек",
                        f"Диапазон: [{min(x_data):.2f}, {max(x_data):.2f}]")

                except Exception as e:
                    st.error(f"Ошибка: {e}")

with ode_num_tabs[2]:
    st.subheader("📊 Численное интегрирование")

    int_func = st.text_input("Функция для интегрирования",
                             "sin(x)",
                             key="num_int_func")
    col1, col2 = st.columns(2)

    with col1:
        a_int = st.number_input("Нижний предел", -10.0, 10.0, 0.0, key="num_a")
        b_int = st.number_input("Верхний предел",
                                -10.0,
                                10.0,
                                np.pi,
                                key="num_b")

    with col2:
        method_int = st.selectbox(
            "Метод интегрирования",
            ["Квадратура", "Метод трапеций", "Метод Симпсона", "Монте-Карло"],
            key="int_method")
        n_segments = st.slider("Сегментов",
                               10,
                               10000,
                               1000,
                               key="n_segments_int")

    if st.button("Вычислить интеграл", key="calc_num_int"):
        try:
            compiled_func = parser.compile_function(int_func)
            func = lambda x: float(compiled_func(x).real)

            if method_int == "Квадратура":
                result, error = integrate.quad(func, a_int, b_int)
                st.success(f"**Результат:** {result:.8f}")
                st.info(f"**Оценка погрешности:** {error:.2e}")

            elif method_int == "Метод трапеций":
                x_vals = np.linspace(a_int, b_int, n_segments + 1)
                y_vals = np.array([func(x) for x in x_vals])
                result = np.trapz(y_vals, x_vals)
                st.success(f"**Результат (трапеции):** {result:.8f}")

            elif method_int == "Метод Симпсона":
                x_vals = np.linspace(a_int, b_int, n_segments + 1)
                y_vals = np.array([func(x) for x in x_vals])
                result = integrate.simpson(y_vals, x_vals)
                st.success(f"**Результат (Симпсон):** {result:.8f}")

            else:
                x_rand = np.random.uniform(a_int, b_int, n_segments)
                y_rand = np.array([func(x) for x in x_rand])
                result = (b_int - a_int) * np.mean(y_rand)
                st.success(f"**Результат (Монте-Карло):** {result:.8f}")

            x_plot = np.linspace(a_int, b_int, 1000)
            y_plot = np.array([func(x) for x in x_plot])

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=x_plot,
                           y=y_plot,
                           mode='lines',
                           name=f'f(x) = {int_func}',
                           fill='tozeroy'))
            fig.update_layout(
                title=
                f"Интеграл ∫{int_func}dx от {a_int} до {b_int} = {result:.6f}",
                xaxis_title="x",
                yaxis_title="f(x)",
                template=plotly_template)
            st.plotly_chart(fig, use_container_width=True)

            save_history(f"Численное интегрирование: {method_int}",
                         f"∫{int_func}dx",
                         f"[{a_int}, {b_int}] = {result:.6f}")

        except Exception as e:
            st.error(f"Ошибка: {e}")

with tabs[13]:
    st.header("Комплексный анализ")

    complex_tabs = st.tabs(["Визуализация", "Вычеты"])

with complex_tabs[0]:
    st.subheader("Визуализация комплексных функций")

    complex_func = st.text_input("Комплексная функция f(z)",
                                 "z**2",
                                 key="complex_func")
    st.caption("💡 Используйте z как комплексную переменную")

    col1, col2 = st.columns(2)
    with col1:
        re_min = st.number_input("Re(z) min", -5.0, 5.0, -2.0)
        re_max = st.number_input("Re(z) max", -5.0, 5.0, 2.0)
    with col2:
        im_min = st.number_input("Im(z) min", -5.0, 5.0, -2.0)
        im_max = st.number_input("Im(z) max", -5.0, 5.0, 2.0)

    resolution = st.slider("Разрешение", 50, 500, 100)
    plot_type = st.selectbox("Тип визуализации",
                             ["Модуль и фаза", "Re и Im", "Домен и образ"],
                             key="complex_plot")

    if st.button("Визуализировать", key="plot_complex"):
        try:
            x = np.linspace(re_min, re_max, resolution)
            y = np.linspace(im_min, im_max, resolution)
            X, Y = np.meshgrid(x, y)
            Z = X + 1j * Y

            if complex_func == "z**2":
                W = Z**2
            elif complex_func == "exp(z)":
                W = np.exp(Z)
            elif complex_func == "sin(z)":
                W = np.sin(Z)
            elif complex_func == "cos(z)":
                W = np.cos(Z)
            elif complex_func == "log(z)":
                W = np.log(Z)
            elif complex_func == "1/z":
                W = 1 / Z
            else:
                W = eval(complex_func, {"z": Z, "np": np, "cmath": cmath})

            if plot_type == "Модуль и фаза":
                fig = make_subplots(rows=1,
                                    cols=2,
                                    subplot_titles=("Модуль |f(z)|",
                                                    "Фаза arg(f(z))"))

                fig.add_trace(
                    go.Heatmap(x=x, y=y, z=np.abs(W), colorscale='Viridis'), 1,
                    1)
                fig.add_trace(
                    go.Heatmap(x=x,
                               y=y,
                               z=np.angle(W),
                               colorscale='HSV',
                               zmin=-np.pi,
                               zmax=np.pi), 1, 2)

            elif plot_type == "Re и Im":
                fig = make_subplots(rows=1,
                                    cols=2,
                                    subplot_titles=("Re(f(z))", "Im(f(z))"))

                fig.add_trace(
                    go.Heatmap(x=x, y=y, z=np.real(W), colorscale='RdBu'), 1,
                    1)
                fig.add_trace(
                    go.Heatmap(x=x, y=y, z=np.imag(W), colorscale='RdBu'), 1,
                    2)

            else:
                fig = make_subplots(rows=1,
                                    cols=2,
                                    subplot_titles=("Домен z", "Образ f(z)"))

                fig.add_trace(
                    go.Heatmap(x=x,
                               y=y,
                               z=np.angle(Z),
                               colorscale='HSV',
                               zmin=-np.pi,
                               zmax=np.pi), 1, 1)
                fig.add_trace(
                    go.Scatter(x=np.real(W).flatten(),
                               y=np.imag(W).flatten(),
                               mode='markers',
                               marker=dict(size=2,
                                           color=np.angle(W).flatten(),
                                           colorscale='HSV',
                                           colorbar=dict(title='Фаза'))), 1, 2)

            fig.update_layout(
                height=500, title_text=f"Визуализация: f(z) = {complex_func}")
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Ошибка: {e}")

    with complex_tabs[1]:
        st.subheader("Вычеты и особые точки")

        residue_func = st.text_input("Функция для анализа вычетов",
                                     "1/(z-1)",
                                     key="residue_func")
        point = st.text_input("Точка для анализа", "1+0j", key="residue_point")

        if st.button("Найти вычет", key="find_residue"):
            try:
                z0 = complex(point)

                def residue_simple(func_str, z0, epsilon=1e-6):
                    z = sp.Symbol('z')
                    f_expr = sp.sympify(func_str.replace('^', '**'))

                    try:
                        residue_expr = (z - z0) * f_expr
                        residue_val = sp.limit(residue_expr, z, z0)
                        return complex(residue_val)
                    except:
                        theta = np.linspace(0, 2 * np.pi, 1000)
                        circle = z0 + epsilon * np.exp(1j * theta)

                        f_lambda = parser.compile_function(
                            residue_func.replace('z', 'x'))
                        f_vals = f_lambda(circle)

                        integral = np.trapz(
                            f_vals * 1j * epsilon * np.exp(1j * theta), theta)
                        return integral / (2 * np.pi * 1j)

                residue = residue_simple(residue_func, z0)

                st.success(f"**Вычет в точке z₀ = {z0}:** {residue:.6f}")

                if abs(residue) > 1e-10:
                    st.info("**Тип точки:** Устранимая особая точка или полюс")
                else:
                    st.info(
                        "**Тип точки:** Существенно особая точка или регулярная"
                    )

            except Exception as e:
                st.error(f"Ошибка: {e}")

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.9rem;'>"
    "🧮 Math Calculator | by quik | 2025 "
    "<br>"
    "<span style='font-size: 0.8rem;'>"
    "Streamlit • NumPy • SciPy • SymPy • Plotly • Pandas"
    "<br>"
    "v2.0"
    "</span>"
    "</div>",
    unsafe_allow_html=True)

st.markdown(
    "<div style='font-size: 0.8rem;'>"
    "Исходный код: <a href='https://github.com/QUIK1001/Event-Horizon-Shell'>GitHub</a><br>"
    "Контакты: <a href='mailto:belov0507artem@gmail.com'>belov0507artem@gmail.com</a>"
    "</div>",
    unsafe_allow_html=True)
