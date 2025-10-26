import sys
import subprocess
import importlib

def ensure_packages():
    """Проверяет и устанавливает необходимые пакеты"""
    packages = ['numpy', 'scipy', 'matplotlib']
    missing = []
    
    for package in packages:
        try:
            importlib.import_module(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("Установка недостающих пакетов...")
        for package in missing:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"{package} установлен")
            except subprocess.CalledProcessError:
                print(f"Не удалось установить {package}")
                return False
    return True

if not ensure_packages():
    print("Некоторые функции могут быть недоступны")


import matplotlib.pyplot as plt
import numpy as np
import re
import math
from fractions import Fraction
from scipy import special, optimize, integrate
import os


class AdvancedMathParser:
    def __init__(self):
        self.operators = {'+', '-', '*', '/', '**', '//', '%'}
        self.functions = {
            'sin': np.sin, 'cos': np.cos, 'tan': np.tan, 'cot': lambda x: 1/np.tan(x),
            'arcsin': np.arcsin, 'arccos': np.arccos, 'arctan': np.arctan, 
            'arccot': lambda x: np.pi/2 - np.arctan(x),
            'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh,
            'arcsinh': np.arcsinh, 'arccosh': np.arccosh, 'arctanh': np.arctanh,
            'exp': np.exp, 'log': np.log, 'log10': np.log10, 'log2': np.log2,
            'ln': np.log, 'lg': np.log10, 'lb': np.log2,
            'sqrt': np.sqrt, 'cbrt': lambda x: np.sign(x) * np.abs(x)**(1/3),
            'ceil': np.ceil, 'floor': np.floor, 'round': np.round,
            'abs': np.abs, 'sign': np.sign,
            'factorial': self._factorial_safe,
            'gamma': special.gamma,
            'beta': lambda x: special.beta(x, x),
            'erf': special.erf,
            'erfc': special.erfc,
            'heaviside': lambda x: np.heaviside(x, 0.5),
            'step': lambda x: np.heaviside(x, 0.5),
            'C': self._combination,
            'P': self._permutation,
            'comb': self._combination,
            'perm': self._permutation,
        }
        self.constants = {
            'pi': np.pi, 'e': math.e, 'tau': 2*np.pi,
            'phi': (1 + np.sqrt(5)) / 2,
            'i': 1j,
            'j': 1j,
        }
        self.cache = {}
        self.history = []
        self.last_result = 0
        self.plot_config = {
            'x_min': -10,
            'x_max': 10,
            'points': 1000,
            'figsize': (12, 8),
        }

    def _factorial_safe(self, x):
        """Безопасное вычисление факториала"""
        x_arr = np.atleast_1d(x)
        if np.any(x_arr < 0) or np.any(x_arr != np.floor(x_arr)):
            raise ValueError("факториал определён только для неотрицательных целых чисел")
        if np.any(x_arr > 170):
            raise ValueError("факториал слишком большого числа (>170)")
        result = np.array([math.factorial(int(xi)) for xi in x_arr]).astype(float)
        return result[0] if np.isscalar(x) else result

    def _combination(self, n):
        """Число сочетаний C(n, k) - для использования как C(5, 2) нужна поддержка двух аргументов"""
        raise ValueError("Используйте комбинаторные функции через внешние функции combination(n, k)")

    def _permutation(self, n):
        """Число перестановок P(n, k) - для использования как P(5, 2) нужна поддержка двух аргументов"""
        raise ValueError("Используйте комбинаторные функции через внешние функции permutation(n, k)")

    def tokenize(self, expression):
        """Токенизация выражения с поддержкой комплексных чисел"""
        pattern = r'(\d+\.?\d*[jJ]|\d+/\d+|\d+\.?\d*([eE][+-]?\d+)?|[a-zA-Z_][a-zA-Z0-9_]*|\*\*|//|<=|>=|==|!=|[+\-*/()^%<>,])'
        tokens = re.findall(pattern, expression)
        return [t[0] for t in tokens]

    def handle_unary_operators(self, tokens):
        """Обработка унарных операторов (унарный минус и плюс)"""
        result = []
        for i, token in enumerate(tokens):
            if token in ['-', '+']:
                is_unary = (i == 0 or 
                           result[-1] in ['(', '+', '-', '*', '/', '//', '%', '**', '^'])
                
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

    def shunting_yard(self, tokens):
        """Алгоритм сортировочной станции с поддержкой унарных операторов"""
        output = []
        stack = []
        precedence = {
            '+': 1, '-': 1, 
            '*': 2, '/': 2, '//': 2, '%': 2,
            '**': 3, '^': 3,
            'unary_minus': 4, 'unary_plus': 4,
        }
        right_associative = {'**', '^', 'unary_minus', 'unary_plus'}

        for token in tokens:
            if self.is_fraction(token):
                output.append(token)
            elif self.is_number(token):
                output.append(token)
            elif token in self.constants:
                output.append(token)
            elif token in self.functions:
                stack.append(token)
            elif token == 'x':
                output.append(token)
            elif token == '(':
                stack.append(token)
            elif token == ')':
                if not stack:
                    raise ValueError("лишняя закрывающая скобка")
                while stack and stack[-1] != '(':
                    output.append(stack.pop())
                if not stack:
                    raise ValueError("несбалансированные скобки")
                stack.pop()
                if stack and stack[-1] in self.functions:
                    output.append(stack.pop())
            elif token in precedence:
                while stack and stack[-1] != '(' and stack[-1] in precedence:
                    top_op = stack[-1]
                    if (token not in right_associative and 
                        precedence[top_op] >= precedence[token]):
                        output.append(stack.pop())
                    elif (token in right_associative and 
                          precedence[top_op] > precedence[token]):
                        output.append(stack.pop())
                    else:
                        break
                stack.append(token)

        while stack:
            if stack[-1] in '()':
                raise ValueError("несбалансированные скобки")
            output.append(stack.pop())

        return output

    def is_number(self, token):
        """Проверка, является ли токен числом (включая комплексные)"""
        try:
            complex(token)
            return True
        except ValueError:
            return False

    def is_fraction(self, token):
        """Проверка, является ли токен дробью"""
        return re.match(r'^-?\d+/\d+$', token) is not None

    def has_variable(self, rpn):
        """Проверка, содержит ли выражение переменную x"""
        return 'x' in rpn

    def evaluate_rpn(self, rpn, x_value):
        """Вычисление RPN выражения с улучшенной обработкой ошибок"""
        stack = []

        for token in rpn:
            if token == 'x':
                stack.append(x_value)
            elif self.is_fraction(token):
                num, den = map(int, token.split('/'))
                if den == 0:
                    raise ValueError("деление на ноль в дроби")
                stack.append(num / den)
            elif self.is_number(token):
                stack.append(complex(token) if 'j' in token.lower() else float(token))
            elif token in self.constants:
                stack.append(self.constants[token])
            elif token in self.functions:
                if not stack:
                    raise ValueError(f"недостаточно аргументов для функции {token}")
                arg = stack.pop()
                try:
                    result = self.functions[token](arg)
                    stack.append(result)
                except Exception as e:
                    raise ValueError(f"ошибка в функции {token}: {e}")
            elif token == 'unary_minus':
                if not stack:
                    raise ValueError("недостаточно операндов для унарного минуса")
                stack.append(-stack.pop())
            elif token == 'unary_plus':
                if not stack:
                    raise ValueError("недостаточно операндов для унарного плюса")
                pass
            elif token in self.operators:
                if len(stack) < 2:
                    raise ValueError(f"недостаточно операндов для оператора {token}")
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
                            raise ValueError("деление на ноль")
                        stack.append(a / b)
                    elif token == '//':
                        if b == 0:
                            raise ValueError("целочисленное деление на ноль")
                        stack.append(a // b)
                    elif token == '%':
                        if b == 0:
                            raise ValueError("остаток от деления на ноль")
                        stack.append(a % b)
                    elif token in ['**', '^']:
                        stack.append(a ** b)
                except Exception as e:
                    raise ValueError(f"ошибка при выполнении операции {token}: {e}")

        if len(stack) != 1:
            raise ValueError("некорректное выражение")

        return stack[0]

    def evaluate_vectorized(self, rpn, x_values):
        """Векторизованное вычисление для ускорения построения графиков"""
        if not isinstance(x_values, np.ndarray):
            x_values = np.array(x_values)
        
        @np.vectorize
        def eval_single(x_val):
            try:
                return self.evaluate_rpn(rpn, x_val)
            except:
                return np.nan
        
        results = eval_single(x_values)
        
        if np.all(np.isreal(results)):
            return np.real(results)
        return results

    def preprocess_expression(self, expr):
        """Предобработка выражения"""
        expr = expr.replace(' ', '')
        expr = expr.replace('^', '**')
        expr = expr.replace(':', '/')
        expr = expr.replace('÷', '/')
        expr = expr.replace('√', 'sqrt')
        expr = expr.replace('"', '').replace("'", '')
        
        expr = re.sub(r'(\d)([a-zA-Z(])', r'\1*\2', expr)
        expr = re.sub(r'\)([a-zA-Z0-9(])', r')*\1', expr)
        
        return expr

    def parse_equation(self, equation):
        """Разбор уравнения"""
        if '=' not in equation:
            if any(c in equation for c in 'yY'):
                raise ValueError("используйте = для уравнений с y")
            return self.preprocess_expression(equation)

        parts = equation.split('=', 1)
        left_raw = parts[0].strip().replace(' ', '')
        right = self.preprocess_expression(parts[1].strip())

        if 'y' not in left_raw:
            return right

        if left_raw == 'y':
            return right

        if re.match(r'^\d+y$', left_raw):
            coeff = left_raw[:-1]
            return f'({right})/({coeff})'

        if re.match(r'^y[+-]\d+$', left_raw):
            if left_raw[1] == '+':
                const = left_raw[2:]
                return f'({right})-({const})'
            else:
                const = left_raw[2:]
                return f'({right})+({const})'

        return right

    def compile_function(self, expression):
        """Компиляция функции с кэшированием"""
        if expression in self.cache:
            return self.cache[expression]
        
        processed_expr = self.preprocess_expression(expression)
        tokens = self.tokenize(processed_expr)
        tokens = self.handle_unary_operators(tokens)
        rpn = self.shunting_yard(tokens)

        def f(x):
            if isinstance(x, (int, float, complex)):
                return self.evaluate_rpn(rpn, x)
            else:
                return self.evaluate_vectorized(rpn, x)

        self.cache[expression] = f
        return f

    def derivative(self, func, x, h=1e-5):
        """Численное дифференцирование"""
        return (func(x + h) - func(x - h)) / (2 * h)

    def integrate_numeric(self, func, a, b):
        """Численное интегрирование"""
        try:
            result, error = integrate.quad(lambda x: np.real(func(x)), a, b)
            return result
        except Exception as e:
            raise ValueError(f"ошибка интегрирования: {e}")

    def find_roots(self, func, x_start=-10.0, x_end=10.0, num_attempts=10):
        """Поиск корней функции"""
        roots = []
        x_guesses = np.linspace(x_start, x_end, num_attempts)
        
        for x0 in x_guesses:
            try:
                root = optimize.fsolve(func, x0, full_output=True)
                if root[2] == 1:
                    r = root[0][0]
                    if not any(abs(r - existing) < 0.01 for existing in roots):
                        if abs(func(r)) < 0.01:
                            roots.append(r)
            except:
                pass
        
        return sorted(roots)


def combination(n, k):
    """Число сочетаний C(n, k)"""
    if k > n or k < 0:
        return 0
    return math.factorial(int(n)) // (math.factorial(int(k)) * math.factorial(int(n - k)))


def permutation(n, k):
    """Число перестановок P(n, k)"""
    if k > n or k < 0:
        return 0
    return math.factorial(int(n)) // math.factorial(int(n - k))


def mean(values):
    """Среднее значение"""
    return np.mean(values)


def median(values):
    """Медиана"""
    return np.median(values)


def std_dev(values):
    """Среднеквадратичное отклонение"""
    return np.std(values)


def variance(values):
    """Дисперсия"""
    return np.var(values)


def run_comprehensive_tests():
    """Комплексное тестирование"""
    parser = AdvancedMathParser()
    test_cases = [
        ("1/2 + 1/3", [], [5/6]),
        ("3/4 * 2/3", [], [0.5]),
        ("1/2 - 1/4", [], [0.25]),
        ("2/3 / 1/3", [], [2.0]),
        
        ("-5", [], [-5.0]),
        ("-5 + 3", [], [-2.0]),
        ("-(5 + 3)", [], [-8.0]),
        ("2 * -3", [], [-6.0]),
        
        ("1/2 + 0.5", [], [1.0]),
        ("3/4 * 2", [], [1.5]),
        ("1 + 1/3", [], [4/3]),
        
        ("(1/2 + 1/3) * 6", [], [5.0]),
        ("sqrt(1/4)", [], [0.5]),
        ("sin(pi/2)", [], [1.0]),
        ("log(e)", [], [1.0]),
        
        ("factorial(5)", [], [120.0]),
        ("ceil(1/3)", [], [1.0]),
        ("floor(5/2)", [], [2.0]),
        
        ("cot(pi/4)", [], [1.0]),
        ("arccot(1)", [], [math.pi/4]),
        
        ("sign(-5)", [], [-1.0]),
        ("sign(5)", [], [1.0]),
        ("heaviside(-1)", [], [0.0]),
        ("heaviside(1)", [], [1.0]),
        
        ("x", [1, 2], [1, 2]),
        ("x ** 2", [2, 3], [4, 9]),
        ("sqrt(x)", [4, 9], [2, 3]),
    ]

    print("запуск комплексных тестов:")
    all_passed = True
    passed_count = 0
    failed_count = 0

    for expr, x_vals, expected in test_cases:
        try:
            if '=' in expr:
                parsed_expr = parser.parse_equation(expr)
            else:
                parsed_expr = parser.preprocess_expression(expr)
            func = parser.compile_function(parsed_expr)

            if x_vals:
                result = [round(float(func(x)), 10) for x in x_vals]
                expected_rounded = [round(float(v), 10) for v in expected]
                if result != expected_rounded:
                    print(f"провал: {expr} -> {result} != {expected_rounded}")
                    all_passed = False
                    failed_count += 1
                else:
                    passed_count += 1
            else:
                result_val = round(float(func(0)), 10)
                expected_val = round(float(expected[0]), 10)
                if result_val != expected_val:
                    print(f"провал: {expr} -> {result_val} != {expected_val}")
                    all_passed = False
                    failed_count += 1
                else:
                    passed_count += 1

        except Exception as e:
            print(f"ошибка в тесте {expr}: {e}")
            all_passed = False
            failed_count += 1

    print(f"\nвсего тестов: {passed_count + failed_count}")
    print(f"пройдено: {passed_count}")
    print(f" провалено: {failed_count}")
    
    if all_passed:
        print("\nвсе тесты пройдены успешно!")
    else:
        print("\nнекоторые тесты не пройдены")

    return all_passed


def show_help():
    """Показать справку"""
    print("\n" + "="*60)
    print("СПРАВКА ПО КАЛЬКУЛЯТОРУ")
    print("="*60)
    print("\nОСНОВНЫЕ ОПЕРАЦИИ:")
    print("  +, -, *, /, //, %, **  - арифметические операции")
    print("  1/2, 3/4               - дроби")
    print("  -5, -(2+3)             - унарный минус")
    
    print("\nТРИГОНОМЕТРИЯ:")
    print("  sin, cos, tan, cot")
    print("  arcsin, arccos, arctan, arccot")
    print("  sinh, cosh, tanh       - гиперболические функции")
    
    print("\nЛОГАРИФМЫ И СТЕПЕНИ:")
    print("  log, ln                - натуральный логарифм")
    print("  log10, lg              - десятичный логарифм")
    print("  log2, lb               - двоичный логарифм")
    print("  exp, sqrt, cbrt        - экспонента и корни")
    
    print("\nСПЕЦИАЛЬНЫЕ ФУНКЦИИ:")
    print("  abs, sign              - модуль и знак")
    print("  ceil, floor, round     - округление")
    print("  factorial              - факториал")
    print("  gamma, erf, erfc       - гамма-функция и функции ошибок")
    print("  heaviside, step        - функция Хевисайда")
    
    print("\nКОМБИНАТОРИКА:")
    print("  Используйте в команде stats:")
    print("  combination(n, k)      - число сочетаний C(n, k)")
    print("  permutation(n, k)      - число перестановок P(n, k)")
    
    print("\nКОНСТАНТЫ:")
    print("  pi, e, tau, phi, i     - математические константы")
    
    print("\nКОМАНДЫ:")
    print("  help                   - показать эту справку")
    print("  history                - показать историю вычислений")
    print("  clear                  - очистить историю")
    print("  config                 - настроить параметры графиков")
    print("  derive <выражение>     - найти производную")
    print("  integrate <выражение>  - вычислить интеграл")
    print("  solve <выражение>      - найти корни уравнения")
    print("  stats <числа>          - статистика (mean, median, std)")
    print("  выход, exit, quit      - выход из калькулятора")
    
    print("\nПРИМЕРЫ:")
    print("  1/2 + 1/3              - дроби")
    print("  sin(pi/2)              - тригонометрия")
    print("  -5 + 3                 - унарный минус")
    print("  y = x**2 - 4           - график")
    print("  derive x**3            - производная")
    print("  solve x**2 - 4         - найти корни")
    print("  stats 1,2,3,4,5        - статистика")
    print("\n  Комбинаторика (вне калькулятора):")
    print("  >>> combination(5, 2)  = 10")
    print("  >>> permutation(5, 2)  = 20")
    print("="*60 + "\n")

def show_copyright():
    print("="*60)
    print("Advanced Mathematical Calculator Plugin for EHS")
    print("2025, by quik")

def main():
    """Интерактивный калькулятор с расширенными возможностями"""
    parser = AdvancedMathParser()
    show_copyright()
    print("\n" + "="*60)
    print("РАСШИРЕННЫЙ МАТЕМАТИЧЕСКИЙ КАЛЬКУЛЯТОР")
    print("="*60)
    print("\nВведите 'help' для справки\n")

    while True:
        try:
            expr = input(">>> ").strip()
            
            if not expr:
                continue
            
            if expr.lower() in ['выход', 'exit', 'quit', 'q']:
                print("\n👋 До свидания!")
                break
            
            elif expr.lower() == 'help':
                show_help()
                continue
            
            elif expr.lower() == 'history':
                if not parser.history:
                    print("История пуста")
                else:
                    print("\nИстория вычислений:")
                    for i, (expression, result) in enumerate(parser.history[-10:], 1):
                        print(f"  {i}. {expression} = {result}")
                print()
                continue
            
            elif expr.lower() == 'clear':
                parser.history = []
                parser.cache = {}
                print("История и кэш очищены")
                continue
            
            elif expr.lower() == 'config':
                print("\nНастройки графиков:")
                try:
                    x_min = float(input(f"  Минимум x [{parser.plot_config['x_min']}]: ") or parser.plot_config['x_min'])
                    x_max = float(input(f"  Максимум x [{parser.plot_config['x_max']}]: ") or parser.plot_config['x_max'])
                    points = int(input(f"  Количество точек [{parser.plot_config['points']}]: ") or parser.plot_config['points'])
                    
                    parser.plot_config['x_min'] = x_min
                    parser.plot_config['x_max'] = x_max
                    parser.plot_config['points'] = points
                    print("Настройки обновлены\n")
                except:
                    print("Ошибка ввода настроек\n")
                continue
            
            elif expr.lower().startswith('derive '):
                func_expr = expr[7:].strip()
                try:
                    func = parser.compile_function(func_expr)
                    x_val = float(input("  Значение x: "))
                    deriv = parser.derivative(func, x_val)
                    print(f"Производная {func_expr} в точке x={x_val}: {deriv}\n")
                except Exception as e:
                    print(f"Ошибка: {e}\n")
                continue
            
            elif expr.lower().startswith('integrate '):
                func_expr = expr[10:].strip()
                try:
                    func = parser.compile_function(func_expr)
                    a = float(input("  Нижний предел: "))
                    b = float(input("  Верхний предел: "))
                    integral = parser.integrate_numeric(func, a, b)
                    print(f"∫ {func_expr} dx от {a} до {b} = {integral}\n")
                except Exception as e:
                    print(f"Ошибка: {e}\n")
                continue
            
            elif expr.lower().startswith('solve '):
                func_expr = expr[6:].strip()
                try:
                    func = parser.compile_function(func_expr)
                    roots = parser.find_roots(func, parser.plot_config['x_min'], parser.plot_config['x_max'])
                    if roots:
                        print(f"Найдены корни уравнения {func_expr} = 0:")
                        for r in roots:
                            print(f"  x = {r:.6f}")
                    else:
                        print(f"Корни не найдены в диапазоне [{parser.plot_config['x_min']}, {parser.plot_config['x_max']}]")
                    print()
                except Exception as e:
                    print(f"Ошибка: {e}\n")
                continue
            
            elif expr.lower().startswith('stats '):
                try:
                    numbers_str = expr[6:].strip()
                    numbers = [float(x.strip()) for x in numbers_str.split(',')]
                    print(f"\nСтатистика для {len(numbers)} чисел:")
                    print(f"  Среднее: {mean(numbers):.4f}")
                    print(f"  Медиана: {median(numbers):.4f}")
                    print(f"  Ст. отклонение: {std_dev(numbers):.4f}")
                    print(f"  Дисперсия: {variance(numbers):.4f}")
                    print(f"  Минимум: {min(numbers):.4f}")
                    print(f"  Максимум: {max(numbers):.4f}\n")
                except Exception as e:
                    print(f"Ошибка: {e}\n")
                continue
            
            if '=' in expr:
                processed_expr = parser.parse_equation(expr)
                func = parser.compile_function(processed_expr)
                
                x_vals = np.linspace(
                    parser.plot_config['x_min'], 
                    parser.plot_config['x_max'], 
                    parser.plot_config['points']
                )
                y_vals = func(x_vals)
                
                plt.figure(figsize=parser.plot_config['figsize'])
                
                if np.any(np.iscomplex(y_vals)):
                    plt.plot(x_vals, np.real(y_vals), 'b-', linewidth=2, label=f'Re({expr})')
                    plt.plot(x_vals, np.imag(y_vals), 'r-', linewidth=2, label=f'Im({expr})')
                else:
                    plt.plot(x_vals, y_vals, 'b-', linewidth=2, label=expr)
                
                plt.axhline(y=0, color='black', linewidth=0.5)
                plt.axvline(x=0, color='black', linewidth=0.5)
                plt.grid(True, alpha=0.3)
                plt.legend(fontsize=12)
                plt.xlabel('x', fontsize=12)
                plt.ylabel('y', fontsize=12)
                plt.title(f'График: {expr}', fontsize=14)

                y_finite = y_vals[np.isfinite(y_vals)]
                if len(y_finite) > 0:
                    if np.all(np.isreal(y_finite)):
                        y_min, y_max = np.min(np.real(y_finite)), np.max(np.real(y_finite))
                    else:
                        y_min = min(np.min(np.real(y_finite)), np.min(np.imag(y_finite)))
                        y_max = max(np.max(np.real(y_finite)), np.max(np.imag(y_finite)))
                    margin = (y_max - y_min) * 0.1 if y_max != y_min else 1
                    plt.ylim(y_min - margin, y_max + margin)

                plt.tight_layout()
                
                save_path = 'plot.png'
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
                print(f"График построен и сохранён в {save_path}")
                plt.show()
                
            else:
                func = parser.compile_function(expr)
                
                tokens = parser.tokenize(parser.preprocess_expression(expr))
                tokens = parser.handle_unary_operators(tokens)
                rpn = parser.shunting_yard(tokens)
                
                if parser.has_variable(rpn):
                    x_input = input("  Введите значение x (или диапазон 'a:b:step'): ").strip()
                    
                    if ':' in x_input:
                        parts = x_input.split(':')
                        x_values = np.array([])
                        if len(parts) == 2:
                            a, b = float(parts[0]), float(parts[1])
                            x_values = np.linspace(a, b, 10)
                        elif len(parts) == 3:
                            a, b, step = float(parts[0]), float(parts[1]), float(parts[2])
                            x_values = np.arange(a, b + step, step)
                        
                        if len(x_values) > 0:
                            print(f"\nРезультаты для {expr}:")
                            for x_val in x_values:
                                result = func(x_val)
                                print(f"  x = {x_val:.4f} → {result}")
                            print()
                    else:
                        x_val = float(x_input)
                        result = func(x_val)
                        print(f"Результат: {result}\n")
                        parser.history.append((f"{expr} (x={x_val})", result))
                        parser.last_result = result
                else:
                    result = func(0)
                    print(f"Результат: {result}\n")
                    parser.history.append((expr, result))
                    parser.last_result = result

        except KeyboardInterrupt:
            print("\n\nДо свидания!")
            break
        except Exception as e:
            print(f"Ошибка: {e}")
            print("Введите 'help' для справки\n")


if __name__ == "__main__":
    success = run_comprehensive_tests()
    print("\n" + "="*60 + "\n")
    
    if success:
        main()
    else:
        print("Обнаружены ошибки в тестах")
        response = input("Продолжить работу калькулятора? (y/n): ")
        if response.lower() in ['y', 'yes', 'д', 'да']:
            main()
# by quik
