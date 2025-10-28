import sys
import subprocess
import os
import importlib
print("Сейчас будет много ошибок. Не волнуйтесь, плагин будет работать корректно.")
def ensure_packages():
    """Проверяет и устанавливает необходимые пакеты"""
    packages = [
        'numpy', 
        'scipy', 
        'matplotlib', 
        'streamlit',
        'plotly',
        'pandas',
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
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", package
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"{package} успешно установлен")
            except subprocess.CalledProcessError:
                print(f" Не удалось установить {package}")
                return False
    else:
        print("Все зависимости установлены!")
    return True

if not ensure_packages():
    print("Не удалось установить все зависимости. Приложение может работать некорректно.")
    sys.exit(1)

def is_streamlit_script(file_path):
    """Проверяет, является ли файл Streamlit скриптом"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            streamlit_indicators = [
                'import streamlit',
                'from streamlit',
                'st.',
                'streamlit.',
                'st.set_page_config',
                'st.title(',
                'st.write('
            ]
            return any(indicator in content for indicator in streamlit_indicators)
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
            "--server.port", "8501",
            "--browser.serverAddress", "localhost"
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
    if not ensure_packages():
        print("Не удалось установить все зависимости. Приложение может работать некорректно.")
        input("Press Enter to continue...")
        return
    
    current_file = os.path.abspath(__file__)
    
    if is_streamlit_script(current_file):
        success = run_streamlit_app(current_file)
        if not success:
            input("Press Enter to continue...")
        return
    

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sympy as sp
from scipy import optimize, integrate, linalg, fft
from scipy.interpolate import make_interp_spline
import json
from datetime import datetime
import base64
import math

st.set_page_config(
    page_title="Математический калькулятор",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.caption("by quik")
st.markdown("""
<style>
    .main { padding: 0rem 1rem; }
    .stTabs [data-baseweb="tab-list"] button { font-size: 1.1rem; }
    h1 { color: #1f77b4; }
    .stAlert { border-radius: 10px; }
    .stButton>button {
        background-color: #1F77B4;
        color: white;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1668A1;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

if 'history' not in st.session_state:
    st.session_state.history = []
if 'saved_sessions' not in st.session_state:
    st.session_state.saved_sessions = {}

def save_to_history(operation, result, details=""):
    st.session_state.history.append({
        'time': datetime.now().strftime("%H:%M:%S"),
        'operation': operation,
        'result': result,
        'details': details
    })

def export_session():
    session_data = {
        'timestamp': datetime.now().isoformat(),
        'history': st.session_state.history
    }
    return json.dumps(session_data, indent=2, ensure_ascii=False)

def download_link(data, filename, text):
    b64 = base64.b64encode(data.encode()).decode()
    return f'<a href="data:application/json;base64,{b64}" download="{filename}">{text}</a>'

def parse_matrix_row(row_str, expected_cols=None):
    try:
        values = [float(x.strip()) for x in row_str.split()]
        if expected_cols is not None and len(values) != expected_cols:
            raise ValueError(f"Ожидается {expected_cols} элементов, получено {len(values)}")
        return values
    except ValueError as e:
        if "could not convert" in str(e):
            raise ValueError("Все элементы должны быть числами")
        raise

def parse_vector(vec_str):
    try:
        values = [float(x.strip()) for x in vec_str.split()]
        if len(values) == 0:
            raise ValueError("Вектор не может быть пустым")
        return np.array(values)
    except ValueError as e:
        if "could not convert" in str(e):
            raise ValueError("Все элементы вектора должны быть числами")
        raise

def parse_number_list(data_str):
    try:
        values = [float(x.strip()) for x in data_str.replace(',', ' ').split()]
        if len(values) == 0:
            raise ValueError("Список не может быть пустым")
        return np.array(values)
    except ValueError as e:
        if "could not convert" in str(e):
            raise ValueError("Все элементы должны быть числами")
        raise

col1, col2 = st.columns([3, 1])
with col1:
    st.title("🧮 Математический калькулятор")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Настройки")
    graph_theme = st.selectbox(
        "Тема графиков",
        ["plotly", "plotly_white", "plotly_dark", "none"]
    )
    st.markdown("---")
    st.subheader("🎨 Настройки графиков")
    line_width = st.slider("Толщина линий", 1, 5, 2)
    grid_alpha = st.slider("Прозрачность сетки", 0.0, 1.0, 0.3)
    st.markdown("---")
    st.subheader("📊 История операций")
    if st.button("Очистить историю"):
        st.session_state.history = []
        st.rerun()
    if st.button("Экспорт сессии"):
        session_json = export_session()
        st.download_button(
            label="Скачать JSON",
            data=session_json,
            file_name=f"math_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json")
    if st.session_state.history:
        st.markdown("**Последние 5 операций:**")
        for item in reversed(st.session_state.history[-5:]):
            with st.expander(f"{item['time']} - {item['operation'][:30]}..."):
                st.write(f"**Результат:** {item['result']}")
                if item['details']:
                    st.write(f"**Детали:** {item['details']}")

tabs = st.tabs([
    "🔢 Калькулятор", "📈 Графики функций", "🎲 Матрицы", "🔺 Векторы",
    "🌐 3D Графики", "🔧 Системы уравнений", "♾️ Символьные вычисления",
    "⚡ Оптимизация", "📊 Анализ данных", "🧩 Дифференциальные уравнения"
])

with tabs[0]:
    st.header("Калькулятор выражений")
    col1, col2 = st.columns([3, 1])
    with col1:
        expression = st.text_input(
            "Введите математическое выражение",
            placeholder="Пример: sin(pi/2) + sqrt(16) * e**2",
            help="Поддерживаются: sin, cos, tan, exp, log, sqrt, pi, e и другие"
        )
    with col2:
        precision = st.slider("Точность", 2, 15, 6)
    if expression:
        try:
            x = sp.Symbol('x')
            expr = sp.sympify(expression)
            if expr.free_symbols:
                st.warning("Выражение содержит переменные. Используйте вкладку 'Графики функций'")
            else:
                result = float(expr.evalf())
                st.success(f"### Результат: `{result:.{precision}f}`")
                st.markdown("**Дополнительная информация:**")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Научная нотация", f"{result:.{precision}e}")
                with col_b:
                    st.metric("Округлено", f"{round(result, precision)}")
                with col_c:
                    try:
                        frac = sp.nsimplify(expr, rational=True)
                        st.metric("Дробь", str(frac))
                    except:
                        st.metric("Дробь", "N/A")
                save_to_history(expression, f"{result:.{precision}f}")
        except Exception as e:
            st.error(f"Ошибка: {str(e)}")
    st.markdown("---")
    st.subheader("Быстрые функции")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown("**Тригонометрия**")
        angle = st.number_input("Угол (градусы)", value=45.0)
        if st.button("Вычислить sin/cos/tan"):
            rad = np.radians(angle)
            st.write(f"sin({angle}°) = {np.sin(rad):.6f}")
            st.write(f"cos({angle}°) = {np.cos(rad):.6f}")
            st.write(f"tan({angle}°) = {np.tan(rad):.6f}")
    
    with col2:
        st.markdown("**Логарифмы**")
        log_val = st.number_input("Число", value=100.0, min_value=0.01)
        if st.button("Вычислить логарифмы"):
            st.write(f"ln({log_val}) = {np.log(log_val):.6f}")
            st.write(f"log10({log_val}) = {np.log10(log_val):.6f}")
            st.write(f"log2({log_val}) = {np.log2(log_val):.6f}")
    
    with col3:
        st.markdown("**Комбинаторика**")
        n = st.number_input("n", value=10, min_value=0, step=1)
        k = st.number_input("k", value=3, min_value=0, step=1)
        if st.button("Вычислить C(n,k) и P(n,k)"):
            from math import comb, perm
            st.write(f"C({n},{k}) = {comb(n, k)}")
            st.write(f"P({n},{k}) = {perm(n, k)}")
    
    with col4:
        st.markdown("**Статистика**")
        data_input = st.text_input("Данные (через запятую)", "1,2,3,4,5")
        if st.button("Анализ"):
            try:
                data = [float(x.strip()) for x in data_input.split(',')]
                st.write(f"Среднее: {np.mean(data):.4f}")
                st.write(f"Медиана: {np.median(data):.4f}")
                st.write(f"Ст. откл: {np.std(data):.4f}")
            except:
                st.error("Неверный формат данных")
    
    with col5:
        st.markdown("**Арифметика**")
        numbers_input = st.text_input("Целые числа (через запятую)", "12, 18, 24")
        if st.button("НОД / НОК"):
            try:
                nums = [int(float(x.strip())) for x in numbers_input.split(',')]
                if any(n <= 0 for n in nums):
                    st.warning("НОД/НОК определены только для положительных целых")
                else:
                    gcd_val = nums[0]
                    for n in nums[1:]:
                        gcd_val = math.gcd(gcd_val, n)
                    lcm_val = abs(nums[0] * nums[1]) // math.gcd(nums[0], nums[1])
                    for n in nums[2:]:
                        lcm_val = abs(lcm_val * n) // math.gcd(lcm_val, n)
                    st.write(f"**НОД**: {gcd_val}")
                    st.write(f"**НОК**: {lcm_val}")
                    
                    def is_prime(n):
                        if n < 2:
                            return False
                        for i in range(2, int(n**0.5)+1):
                            if n % i == 0:
                                return False
                        return True
                    
                    def prime_factors(n):
                        i = 2
                        factors = []
                        while i * i <= n:
                            while n % i == 0:
                                factors.append(i)
                                n //= i
                            i += 1
                        if n > 1:
                            factors.append(n)
                        return factors
                    
                    n0 = nums[0]
                    st.write(f"**{n0} — {'простое' if is_prime(n0) else 'составное'}**")
                    st.write(f"Разложение: {' × '.join(map(str, prime_factors(n0)))}")
                    
                    st.write(f"Двоичная: {bin(n0)}")
                    st.write(f"Восьмеричная: {oct(n0)}")
                    st.write(f"Шестнадцатеричная: {hex(n0)}")
                    
                    st.toast("Арифметика вычислена!", icon="✅")
            except Exception as e:
                st.error(f"Ошибка: введите целые числа. {str(e)}")

with tabs[1]:
    st.header("Графики функций")

    col1, col2 = st.columns([2, 1])

    with col1:
        func_expr = st.text_input("Функция f(x)",
                                  value="sin(x) + cos(2*x)",
                                  help="Используйте x как переменную")

    with col2:
        graph_type = st.selectbox(
            "Тип графика",
            ["Обычный", "С производной", "С интегралом", "Все вместе"])

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        x_min = st.number_input("x min", value=-10.0)
    with col_b:
        x_max = st.number_input("x max", value=10.0)
    with col_c:
        points = st.slider("Точек", 100, 2000, 500)

    if st.button("Построить график", type="primary"):
        try:
            x_sym = sp.Symbol('x')
            expr = sp.sympify(func_expr)
            f_lambda = sp.lambdify(x_sym, expr, 'numpy')

            x_vals = np.linspace(x_min, x_max, points)
            y_vals = f_lambda(x_vals)

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(x=x_vals,
                           y=y_vals,
                           mode='lines',
                           name=f'f(x) = {func_expr}',
                           line=dict(width=2, color='blue')))

            if graph_type in ["С производной", "Все вместе"]:
                derivative = sp.diff(expr, x_sym)
                df_lambda = sp.lambdify(x_sym, derivative, 'numpy')
                dy_vals = df_lambda(x_vals)

                fig.add_trace(
                    go.Scatter(x=x_vals,
                               y=dy_vals,
                               mode='lines',
                               name=f"f'(x) = {derivative}",
                               line=dict(width=2, color='red', dash='dash')))

            if graph_type in ["С интегралом", "Все вместе"]:
                integral = sp.integrate(expr, x_sym)
                int_lambda = sp.lambdify(x_sym, integral, 'numpy')
                int_vals = int_lambda(x_vals)

                fig.add_trace(
                    go.Scatter(x=x_vals,
                               y=int_vals,
                               mode='lines',
                               name=f"∫f(x)dx = {integral}",
                               line=dict(width=2, color='green', dash='dot')))

            fig.update_layout(title=f"График функции: {func_expr}",
                              xaxis_title="x",
                              yaxis_title="y",
                              hovermode='x unified',
                              template=theme,
                              height=600)

            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Анализ функции")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Производная:**")
                derivative = sp.diff(expr, x_sym)
                st.latex(sp.latex(derivative))

            with col2:
                st.markdown("**Интеграл:**")
                integral = sp.integrate(expr, x_sym)
                st.latex(sp.latex(integral))

            with col3:
                st.markdown("**Предел при x→∞:**")
                try:
                    limit = sp.limit(expr, x_sym, sp.oo)
                    st.latex(sp.latex(limit))
                except:
                    st.write("Не существует")

            save_to_history(f"График: {func_expr}", "Успешно построен")

        except Exception as e:
            st.error(f"Ошибка: {str(e)}")

    st.markdown("---")
    st.subheader("Найти корни уравнения")

    col1, col2 = st.columns(2)
    with col1:
        root_expr = st.text_input("Уравнение (= 0)", value="x**2 - 4")
    with col2:
        root_range = st.slider("Диапазон поиска", -100.0, 100.0, (-10.0, 10.0))

    if st.button("Найти корни"):
        try:
            x_sym = sp.Symbol('x')
            expr = sp.sympify(root_expr)
            f_lambda = sp.lambdify(x_sym, expr, 'numpy')

            roots = []
            for x0 in np.linspace(root_range[0], root_range[1], 20):
                try:
                    root = optimize.fsolve(f_lambda, x0)[0]
                    if root_range[0] <= root <= root_range[1]:
                        if not any(abs(root - r) < 0.01 for r in roots):
                            if abs(f_lambda(root)) < 0.001:
                                roots.append(root)
                except:
                    pass

            if roots:
                st.success(f"Найдено корней: {len(roots)}")
                for i, root in enumerate(sorted(roots), 1):
                    st.write(f"x_{i} = {root:.6f}")
            else:
                st.warning("Корни не найдены в указанном диапазоне")

        except Exception as e:
            st.error(f"Ошибка: {str(e)}")

with tabs[2]:
    st.header("Матричные операции")

    operation = st.selectbox("Выберите операцию", [
        "Сложение/Вычитание", "Умножение", "Поэлементное умножение", 
        "Определитель", "Обратная матрица", "Возведение в степень",
        "Собственные значения", "LU-разложение", "QR-разложение", "SVD"
    ])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Матрица A")
        rows_a = st.number_input("Строк A", 2, 10, 3, key="rows_a")
        cols_a = st.number_input("Столбцов A", 2, 10, 3, key="cols_a")

        matrix_a_data = []
        matrix_a_valid = True
        for i in range(rows_a):
            row = st.text_input(
                f"Строка {i+1} (через пробел)",
                value=" ".join(["1" if j == i else "0"
                                for j in range(cols_a)]),
                key=f"a_{i}")
            try:
                matrix_a_data.append(parse_matrix_row(row, cols_a))
            except ValueError as e:
                st.error(f"Ошибка в строке {i+1}: {str(e)}")
                matrix_a_valid = False
                break

        if matrix_a_valid and matrix_a_data:
            matrix_a = np.array(matrix_a_data)
            st.write("**Матрица A:**")
            st.dataframe(pd.DataFrame(matrix_a))
        else:
            matrix_a = None

    with col2:
        if operation in ["Сложение/Вычитание", "Умножение", "Поэлементное умножение"]:
            st.subheader("Матрица B")
            rows_b = st.number_input("Строк B", 2, 10, 3, key="rows_b")
            cols_b = st.number_input("Столбцов B", 2, 10, 3, key="cols_b")

            matrix_b_data = []
            matrix_b_valid = True
            for i in range(rows_b):
                row = st.text_input(f"Строка {i+1} (через пробел)",
                                    value=" ".join([
                                        "2" if j == i else "0"
                                        for j in range(cols_b)
                                    ]),
                                    key=f"b_{i}")
                try:
                    matrix_b_data.append(parse_matrix_row(row, cols_b))
                except ValueError as e:
                    st.error(f"Ошибка в строке {i+1}: {str(e)}")
                    matrix_b_valid = False
                    break

            if matrix_b_valid and matrix_b_data:
                matrix_b = np.array(matrix_b_data)
                st.write("**Матрица B:**")
                st.dataframe(pd.DataFrame(matrix_b))
            else:
                matrix_b = None
        else:
            matrix_b = None

    if operation == "Возведение в степень":
        power = st.number_input("Степень", value=2, min_value=1, max_value=10)

    if st.button("Вычислить", type="primary"):
        if matrix_a is None:
            st.error("Ошибка в матрице A. Проверьте введенные данные.")
        elif operation in ["Сложение/Вычитание", "Умножение", "Поэлементное умножение"] and matrix_b is None:
            st.error("Ошибка в матрице B. Проверьте введенные данные.")
        else:
            try:
                if operation == "Сложение/Вычитание":
                    op = st.radio("Операция", ["Сложение", "Вычитание"],
                                  horizontal=True)
                    if matrix_a.shape == matrix_b.shape:
                        result = matrix_a + matrix_b if op == "Сложение" else matrix_a - matrix_b
                        st.success(f"Результат ({op}):")
                        st.dataframe(pd.DataFrame(result))
                    else:
                        st.error("Размеры матриц должны совпадать")

                elif operation == "Умножение":
                    if matrix_a.shape[1] == matrix_b.shape[0]:
                        result = matrix_a @ matrix_b
                        st.success("Результат (A × B):")
                        st.dataframe(pd.DataFrame(result))
                    else:
                        st.error(
                            "Число столбцов A должно равняться числу строк B")

                elif operation == "Поэлементное умножение":
                    if matrix_a.shape == matrix_b.shape:
                        result = matrix_a * matrix_b
                        st.success("Результат (поэлементное умножение A ⊙ B):")
                        st.dataframe(pd.DataFrame(result))
                    else:
                        st.error("Размеры матриц должны совпадать")

                elif operation == "Определитель":
                    if matrix_a.shape[0] == matrix_a.shape[1]:
                        det = linalg.det(matrix_a)
                        st.success(f"Определитель: **{det:.6f}**")
                        save_to_history("Определитель матрицы", f"{det:.6f}")
                    else:
                        st.error("Матрица должна быть квадратной")

                elif operation == "Обратная матрица":
                    if matrix_a.shape[0] == matrix_a.shape[1]:
                        try:
                            inv = linalg.inv(matrix_a)
                            st.success("Обратная матрица:")
                            st.dataframe(pd.DataFrame(inv))

                            verification = matrix_a @ inv
                            st.info("Проверка (A × A⁻¹):")
                            st.dataframe(pd.DataFrame(verification))
                        except:
                            st.error("Матрица вырожденная (определитель = 0)")
                    else:
                        st.error("Матрица должна быть квадратной")

                elif operation == "Возведение в степень":
                    if matrix_a.shape[0] == matrix_a.shape[1]:
                        result = np.linalg.matrix_power(matrix_a, power)
                        st.success(f"Результат (A^{power}):")
                        st.dataframe(pd.DataFrame(result))
                    else:
                        st.error("Матрица должна быть квадратной")

                elif operation == "Собственные значения":
                    if matrix_a.shape[0] == matrix_a.shape[1]:
                        eigenvalues, eigenvectors = linalg.eig(matrix_a)

                        st.success("Собственные значения:")
                        for i, val in enumerate(eigenvalues):
                            if np.isreal(val):
                                st.write(f"λ_{i+1} = {val.real:.6f}")
                            else:
                                st.write(f"λ_{i+1} = {val:.6f}")

                        st.info("Собственные векторы:")
                        st.dataframe(pd.DataFrame(eigenvectors.real))
                    else:
                        st.error("Матрица должна быть квадратной")

                elif operation == "LU-разложение":
                    if matrix_a.shape[0] == matrix_a.shape[1]:
                        P, L, U = linalg.lu(matrix_a)

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write("**Матрица P:**")
                            st.dataframe(pd.DataFrame(P))
                        with col2:
                            st.write("**Матрица L:**")
                            st.dataframe(pd.DataFrame(L))
                        with col3:
                            st.write("**Матрица U:**")
                            st.dataframe(pd.DataFrame(U))
                    else:
                        st.error("Матрица должна быть квадратной")

                elif operation == "QR-разложение":
                    Q, R = linalg.qr(matrix_a)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Матрица Q (ортогональная):**")
                        st.dataframe(pd.DataFrame(Q))
                    with col2:
                        st.write("**Матрица R (верхнетреугольная):**")
                        st.dataframe(pd.DataFrame(R))

                elif operation == "SVD":
                    U, s, Vt = linalg.svd(matrix_a)

                    st.success("Сингулярные значения:")
                    st.write(s)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write("**Матрица U:**")
                        st.dataframe(pd.DataFrame(U))
                    with col2:
                        st.write("**Сингулярные значения Σ:**")
                        st.dataframe(pd.DataFrame(np.diag(s)))
                    with col3:
                        st.write("**Матрица V^T:**")
                        st.dataframe(pd.DataFrame(Vt))

            except Exception as e:
                st.error(f"Ошибка: {str(e)}")

with tabs[3]:
    st.header("Векторные операции")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Вектор A")
        dim = st.number_input("Размерность", 2, 10, 3)
        vec_a_input = st.text_input("Координаты A (через пробел)", "1 2 3")
        try:
            vec_a = parse_vector(vec_a_input)
        except ValueError as e:
            st.error(f"Ошибка в векторе A: {str(e)}")
            vec_a = None

    with col2:
        st.subheader("Вектор B")
        vec_b_input = st.text_input("Координаты B (через пробел)", "4 5 6")
        try:
            vec_b = parse_vector(vec_b_input)
        except ValueError as e:
            st.error(f"Ошибка в векторе B: {str(e)}")
            vec_b = None

    if st.button("Вычислить операции", type="primary"):
        if vec_a is None or vec_b is None:
            st.error("Ошибка в векторах. Проверьте введенные данные.")
        elif len(vec_a) != len(vec_b):
            st.error(
                f"Векторы должны быть одинаковой размерности. A: {len(vec_a)}, B: {len(vec_b)}"
            )
        else:
            try:
                st.subheader("Результаты")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Скалярное произведение",
                              f"{np.dot(vec_a, vec_b):.6f}")
                    st.metric("Длина вектора A",
                              f"{np.linalg.norm(vec_a):.6f}")
                    st.metric("Длина вектора B",
                              f"{np.linalg.norm(vec_b):.6f}")

                with col2:
                    angle = np.arccos(
                        np.dot(vec_a, vec_b) /
                        (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))
                    st.metric("Угол между векторами (рад)", f"{angle:.6f}")
                    st.metric("Угол между векторами (град)",
                              f"{np.degrees(angle):.6f}")

                    if len(vec_a) == 3 and len(vec_b) == 3:
                        cross = np.cross(vec_a, vec_b)
                        st.metric(
                            "Векторное произведение",
                            f"[{cross[0]:.2f}, {cross[1]:.2f}, {cross[2]:.2f}]"
                        )

                with col3:
                    st.write("**A + B:**", vec_a + vec_b)
                    st.write("**A - B:**", vec_a - vec_b)
                    proj = (np.dot(vec_a, vec_b) /
                            np.dot(vec_b, vec_b)) * vec_b
                    st.write("**Проекция A на B:**", proj)

                if len(vec_a) == 3 and len(vec_b) == 3:
                    fig = go.Figure()

                    origin = [0, 0, 0]
                    fig.add_trace(
                        go.Scatter3d(x=[origin[0], vec_a[0]],
                                     y=[origin[1], vec_a[1]],
                                     z=[origin[2], vec_a[2]],
                                     mode='lines+markers',
                                     name='Вектор A',
                                     line=dict(color='blue', width=5),
                                     marker=dict(size=8)))

                    fig.add_trace(
                        go.Scatter3d(x=[origin[0], vec_b[0]],
                                     y=[origin[1], vec_b[1]],
                                     z=[origin[2], vec_b[2]],
                                     mode='lines+markers',
                                     name='Вектор B',
                                     line=dict(color='red', width=5),
                                     marker=dict(size=8)))

                    cross = np.cross(vec_a, vec_b)
                    fig.add_trace(
                        go.Scatter3d(x=[origin[0], cross[0]],
                                     y=[origin[1], cross[1]],
                                     z=[origin[2], cross[2]],
                                     mode='lines+markers',
                                     name='A × B',
                                     line=dict(color='green',
                                               width=5,
                                               dash='dash'),
                                     marker=dict(size=8)))

                    fig.update_layout(title="3D визуализация векторов",
                                      scene=dict(xaxis_title='X',
                                                 yaxis_title='Y',
                                                 zaxis_title='Z'),
                                      template=graph_theme,
                                      height=600)

                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Ошибка: {str(e)}")

with tabs[4]:
    st.header("3D Графики")

    graph_3d_type = st.selectbox(
        "Тип 3D графика",
        ["Поверхность z=f(x,y)", "Параметрическая кривая", "Векторное поле"])

    if graph_3d_type == "Поверхность z=f(x,y)":
        func_3d = st.text_input("Функция z = f(x, y)",
                                value="sin(sqrt(x**2 + y**2))",
                                help="Используйте x и y как переменные")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            x_min_3d = st.number_input("x min", value=-5.0, key="x_min_3d")
        with col2:
            x_max_3d = st.number_input("x max", value=5.0, key="x_max_3d")
        with col3:
            y_min_3d = st.number_input("y min", value=-5.0, key="y_min_3d")
        with col4:
            y_max_3d = st.number_input("y max", value=5.0, key="y_max_3d")

        points_3d = st.slider("Разрешение", 20, 200, 50)

        if st.button("Построить 3D график", type="primary"):
            try:
                x_sym, y_sym = sp.symbols('x y')
                expr = sp.sympify(func_3d)
                f_lambda = sp.lambdify((x_sym, y_sym), expr, 'numpy')

                x = np.linspace(x_min_3d, x_max_3d, points_3d)
                y = np.linspace(y_min_3d, y_max_3d, points_3d)
                X, Y = np.meshgrid(x, y)
                Z = f_lambda(X, Y)

                fig = go.Figure(
                    data=[go.Surface(x=X, y=Y, z=Z, colorscale='Viridis')])

                fig.update_layout(title=f"3D График: z = {func_3d}",
                                  scene=dict(xaxis_title='X',
                                             yaxis_title='Y',
                                             zaxis_title='Z'),
                                  template=graph_theme,
                                  height=700)

                st.plotly_chart(fig, use_container_width=True)
                save_to_history(f"3D график: {func_3d}", "Успешно построен")

            except Exception as e:
                st.error(f"Ошибка: {str(e)}")

    elif graph_3d_type == "Параметрическая кривая":
        st.write("Параметрические уравнения:")
        col1, col2, col3 = st.columns(3)
        with col1:
            x_param = st.text_input("x(t)", value="cos(t)")
        with col2:
            y_param = st.text_input("y(t)", value="sin(t)")
        with col3:
            z_param = st.text_input("z(t)", value="t")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            t_min = st.number_input("t min", value=0.0)
        with col_b:
            t_max = st.number_input("t max", value=10.0)
        with col_c:
            t_points = st.slider("Точек", 50, 1000, 200)

        if st.button("Построить кривую", type="primary"):
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
                                     color=t, colorscale='Rainbow', width=5))
                ])

                fig.update_layout(title="Параметрическая кривая",
                                  scene=dict(xaxis_title='X',
                                             yaxis_title='Y',
                                             zaxis_title='Z'),
                                  template=graph_theme,
                                  height=700)

                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Ошибка: {str(e)}")

    elif graph_3d_type == "Векторное поле":
        st.subheader("Векторное поле F(x,y) = [P(x,y), Q(x,y)]")
        
        col1, col2 = st.columns(2)
        with col1:
            p_func = st.text_input("P(x,y) - x-компонента", 
                                 value="-y",
                                 help="Компонента по оси X")
        with col2:
            q_func = st.text_input("Q(x,y) - y-компонента", 
                                 value="x",
                                 help="Компонента по оси Y")

        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            vf_x_min = st.number_input("x min", value=-2.0, key="vf_x_min")
        with col_b:
            vf_x_max = st.number_input("x max", value=2.0, key="vf_x_max")
        with col_c:
            vf_y_min = st.number_input("y min", value=-2.0, key="vf_y_min")
        with col_d:
            vf_y_max = st.number_input("y max", value=2.0, key="vf_y_max")

        vf_points = st.slider("Плотность сетки", 5, 30, 15)
        arrow_scale = st.slider("Масштаб стрелок", 0.1, 2.0, 0.5)

        if st.button("Построить векторное поле", type="primary"):
            try:
                x_sym, y_sym = sp.symbols('x y')
                
                p_expr = sp.sympify(p_func)
                q_expr = sp.sympify(q_func)
                
                p_lambda = sp.lambdify((x_sym, y_sym), p_expr, 'numpy')
                q_lambda = sp.lambdify((x_sym, y_sym), q_expr, 'numpy')

                x = np.linspace(vf_x_min, vf_x_max, vf_points)
                y = np.linspace(vf_y_min, vf_y_max, vf_points)
                X, Y = np.meshgrid(x, y)
                
                U = p_lambda(X, Y)
                V = q_lambda(X, Y)
                
                magnitude = np.sqrt(U**2 + V**2)
                U_norm = U / (magnitude + 1e-8) * arrow_scale
                V_norm = V / (magnitude + 1e-8) * arrow_scale

                fig = go.Figure()

                fig.add_trace(go.Cone(
                    x=X.flatten(),
                    y=Y.flatten(),
                    z=np.zeros_like(X.flatten()),
                    u=U.flatten(),
                    v=V.flatten(),
                    w=np.zeros_like(U.flatten()),
                    colorscale='Blues',
                    sizemode="absolute",
                    sizeref=arrow_scale,
                    showscale=False
                ))

                fig.add_trace(go.Scatter(
                    x=X.flatten(),
                    y=Y.flatten(),
                    mode='markers',
                    marker=dict(
                        size=2,
                        color=magnitude.flatten(),
                        colorscale='Viridis',
                        showscale=True
                    ),
                    name='Магнитуда'
                ))

                fig.update_layout(
                    title=f"Векторное поле: F(x,y) = [{p_func}, {q_func}]",
                    scene=dict(
                        xaxis_title='X',
                        yaxis_title='Y', 
                        zaxis_title='',
                        aspectmode='data'
                    ),
                    template=graph_theme,
                    height=700
                )

                st.plotly_chart(fig, use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Максимальная магнитуда", f"{np.max(magnitude):.4f}")
                with col2:
                    st.metric("Минимальная магнитуда", f"{np.min(magnitude):.4f}")
                with col3:
                    divergence = sp.diff(p_expr, x_sym) + sp.diff(q_expr, y_sym)
                    st.metric("Дивергенция", f"{sp.latex(divergence)}")
                
                save_to_history(f"Векторное поле: [{p_func}, {q_func}]", "Успешно построено")

            except Exception as e:
                st.error(f"Ошибка: {str(e)}")

with tabs[5]:
    st.header("Системы линейных уравнений")

    st.markdown("Решение системы вида **Ax = b**")

    n_equations = st.number_input("Число уравнений/неизвестных", 2, 10, 3)

    st.subheader("Матрица коэффициентов A")
    A_data = []
    for i in range(n_equations):
        row = st.text_input(f"Уравнение {i+1} (коэффициенты через пробел)",
                            value=" ".join([
                                str(j + 1) if j == i else "1"
                                for j in range(n_equations)
                            ]),
                            key=f"eq_{i}")
        A_data.append([float(x) for x in row.split()])

    A = np.array(A_data)

    st.subheader("Вектор свободных членов b")
    b_input = st.text_input("Свободные члены (через пробел)",
                            " ".join(["1"] * n_equations))
    b = np.array([float(x) for x in b_input.split()])

    if st.button("Решить систему", type="primary"):
        try:
            st.write("**Матрица A:**")
            st.dataframe(pd.DataFrame(A))
            st.write("**Вектор b:**", b)

            det = linalg.det(A)
            st.info(f"Определитель матрицы A: {det:.6f}")

            if abs(det) > 1e-10:
                x = linalg.solve(A, b)

                st.success("Решение системы:")
                for i, val in enumerate(x):
                    st.write(f"x_{i+1} = {val:.6f}")

                verification = A @ x
                st.info("Проверка (Ax):")
                st.write(verification)
                st.write("Ожидалось (b):")
                st.write(b)
                st.write("Погрешность:")
                st.write(np.abs(verification - b))

                save_to_history("Система уравнений", f"Решение: {x}")
            else:
                st.warning(
                    "Система вырожденная или имеет бесконечно много решений")
                st.write("Попытка найти решение методом наименьших квадратов:")
                x, residuals, rank, s = linalg.lstsq(A, b)
                st.write("Приближенное решение:")
                for i, val in enumerate(x):
                    st.write(f"x_{i+1} = {val:.6f}")

        except Exception as e:
            st.error(f"Ошибка: {str(e)}")

with tabs[6]:
    st.header("Символьные вычисления")

    sym_operation = st.selectbox("Выберите операцию", [
        "Упрощение", "Разложение на множители", "Раскрытие скобок",
        "Производная", "Интеграл", "Предел", "Ряд Тейлора", "Решение уравнения"
    ])

    expr_input = st.text_input("Введите выражение",
                               value="(x**2 - 1)/(x - 1)" if sym_operation
                               == "Упрощение" else "x**2 + 2*x + 1",
                               help="Используйте синтаксис sympy")

    if st.button("Выполнить", type="primary"):
        try:
            x = sp.Symbol('x')
            expr = sp.sympify(expr_input)

            if sym_operation == "Упрощение":
                result = sp.simplify(expr)
                st.success("Упрощенное выражение:")
                st.latex(sp.latex(result))
                st.code(str(result))

            elif sym_operation == "Разложение на множители":
                result = sp.factor(expr)
                st.success("Разложение на множители:")
                st.latex(sp.latex(result))
                st.code(str(result))

            elif sym_operation == "Раскрытие скобок":
                result = sp.expand(expr)
                st.success("Раскрытые скобки:")
                st.latex(sp.latex(result))
                st.code(str(result))

            elif sym_operation == "Производная":
                order = st.number_input("Порядок производной", 1, 5, 1)
                result = sp.diff(expr, x, order)
                st.success(f"Производная порядка {order}:")
                st.latex(sp.latex(result))
                st.code(str(result))

            elif sym_operation == "Интеграл":
                definite = st.checkbox("Определенный интеграл")
                if definite:
                    col1, col2 = st.columns(2)
                    with col1:
                        a = st.number_input("Нижний предел", value=0.0)
                    with col2:
                        b = st.number_input("Верхний предел", value=1.0)
                    result = sp.integrate(expr, (x, a, b))
                    st.success(f"Определенный интеграл от {a} до {b}:")
                else:
                    result = sp.integrate(expr, x)
                    st.success("Неопределенный интеграл:")
                st.latex(sp.latex(result))
                st.code(str(result))

            elif sym_operation == "Предел":
                point = st.text_input("Точка (например: 0, oo, -oo)",
                                      value="0")
                point_val = sp.oo if point == "oo" else (
                    -sp.oo if point == "-oo" else float(point))
                direction = st.selectbox("Направление", ["+", "-", "±"])

                if direction == "±":
                    result = sp.limit(expr, x, point_val)
                else:
                    result = sp.limit(expr, x, point_val, direction)

                st.success(f"Предел при x → {point} ({direction}):")
                st.latex(sp.latex(result))
                st.code(str(result))

            elif sym_operation == "Ряд Тейлора":
                point = st.number_input("Точка разложения", value=0.0)
                order = st.number_input("Порядок", 1, 20, 5)
                result = sp.series(expr, x, point, order)
                st.success(f"Ряд Тейлора в точке {point} до порядка {order}:")
                st.latex(sp.latex(result))
                st.code(str(result))

            elif sym_operation == "Решение уравнения":
                st.info("Решаем уравнение: выражение = 0")
                solutions = sp.solve(expr, x)
                st.success(f"Найдено решений: {len(solutions)}")
                for i, sol in enumerate(solutions, 1):
                    st.write(f"x_{i} =")
                    st.latex(sp.latex(sol))
                    st.code(str(sol))

            save_to_history(f"{sym_operation}: {expr_input}", str(result))

        except Exception as e:
            st.error(f"Ошибка: {str(e)}")

with tabs[7]:
    st.header("Оптимизация функций")
    opt_type = st.selectbox("Тип оптимизации", [
        "Минимум одномерной функции", "Минимум многомерной функции",
        "Минимум с ограничениями", "Подбор параметров кривой"
    ])
    if opt_type == "Минимум одномерной функции":
        func_opt = st.text_input("Функция f(x)", value="x**4 - 3*x**3 + 2")
        col1, col2 = st.columns(2)
        with col1:
            x_start = st.number_input("Начальная точка", value=0.0)
        with col2:
            method_1d = st.selectbox("Метод",
                                     ["BFGS", "Nelder-Mead", "Powell"])
        if st.button("Найти минимум", type="primary"):
            try:
                x_sym = sp.Symbol('x')
                expr = sp.sympify(func_opt)
                f_lambda = sp.lambdify(x_sym, expr, 'numpy')
                result = optimize.minimize(f_lambda, x_start, method=method_1d)
                st.success("Результат оптимизации:")
                st.write(f"**x_min = {result.x[0]:.6f}**")
                st.write(f"**f(x_min) = {result.fun:.6f}**")
                st.write(
                    f"**Итераций: {result.nit if hasattr(result, 'nit') else 'N/A'}**"
                )
                st.write(f"**Успех: {result.success}**")
                x_plot = np.linspace(result.x[0] - 5, result.x[0] + 5, 500)
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
                fig.update_layout(title="График функции с найденным минимумом",
                                  xaxis_title="x",
                                  yaxis_title="f(x)",
                                  template=graph_theme)
                st.plotly_chart(fig, use_container_width=True)
                save_to_history(f"Оптимизация: {func_opt}",
                                f"x_min={result.x[0]:.6f}")
            except Exception as e:
                st.error(f"Ошибка: {str(e)}")

    elif opt_type == "Минимум многомерной функции":
        st.subheader("Минимизация функции f(x₁, x₂, ..., xₙ)")
        func_multi = st.text_input("Функция (используйте x[0], x[1], ...)", value="x[0]**2 + x[1]**2")
        n_vars = st.number_input("Число переменных", min_value=2, max_value=5, value=2)
        initial_guess = st.text_input("Начальное приближение (через запятую)", value="1, 1")
        method_multi = st.selectbox("Метод", ["BFGS", "Nelder-Mead", "Powell", "CG"])

        if st.button("Найти минимум", type="primary"):
            try:
                def func(x):
                    allowed_names = {"x": x, "__builtins__": {}}
                    code = compile(func_multi, "<string>", "eval")
                    for name in code.co_names:
                        if name not in ["x"]:
                            raise ValueError("Используйте только переменную x (например: x[0], x[1])")
                    return eval(code, allowed_names)

                x0 = np.array([float(v.strip()) for v in initial_guess.split(',')])
                if len(x0) != n_vars:
                    st.error(f"Ожидалось {n_vars} начальных значений, получено {len(x0)}")
                else:
                    result = optimize.minimize(func, x0, method=method_multi)
                    st.success("Результат оптимизации:")
                    st.write(f"**x_min = [{', '.join(f'{v:.4f}' for v in result.x)}]**")
                    st.write(f"**f(x_min) = {result.fun:.6f}**")
                    st.write(f"**Успех: {result.success}**")
                    if n_vars == 2:
                        x1 = np.linspace(x0[0]-3, x0[0]+3, 100)
                        x2 = np.linspace(x0[1]-3, x0[1]+3, 100)
                        X1, X2 = np.meshgrid(x1, x2)
                        Z = np.zeros_like(X1)
                        for i in range(X1.shape[0]):
                            for j in range(X1.shape[1]):
                                Z[i, j] = func([X1[i, j], X2[i, j]])
                        fig = go.Figure(data=go.Contour(x=x1, y=x2, z=Z, colorscale='Viridis'))
                        fig.add_trace(go.Scatter(x=[result.x[0]], y=[result.x[1]],
                                                 mode='markers', marker=dict(size=12, color='red'),
                                                 name='Минимум'))
                        fig.update_layout(title="Контурный график функции", template=graph_theme)
                        st.plotly_chart(fig, use_container_width=True)
                    save_to_history(f"Многомерная оптимизация: {func_multi}", f"x_min={result.x}")
            except Exception as e:
                st.error(f"Ошибка: {str(e)}")

    elif opt_type == "Минимум с ограничениями":
        st.subheader("Минимизация с ограничениями (неравенства и равенства)")
        func_constr = st.text_input("Целевая функция f(x[0], x[1])", value="x[0]**2 + x[1]**2")
        constr_type = st.radio("Тип ограничения", ["Неравенство (≥0)", "Равенство (=0)"])
        constr_expr = st.text_input("Ограничение g(x) = ...", value="x[0] + x[1] - 1")
        x0_input = st.text_input("Начальное приближение", value="0.5, 0.5")

        if st.button("Решить с ограничениями", type="primary"):
            try:
                def objective(x):
                    allowed = {"x": x, "__builtins__": {}}
                    return eval(compile(func_constr, "<string>", "eval"), allowed)

                def constraint_func(x):
                    allowed = {"x": x, "__builtins__": {}}
                    return eval(compile(constr_expr, "<string>", "eval"), allowed)

                x0 = np.array([float(v.strip()) for v in x0_input.split(',')])
                if constr_type == "Неравенство (≥0)":
                    constr = {"type": "ineq", "fun": constraint_func}
                else:
                    constr = {"type": "eq", "fun": constraint_func}

                result = optimize.minimize(objective, x0, constraints=constr, method='SLSQP')
                st.success("Результат:")
                st.write(f"**x = [{', '.join(f'{v:.4f}' for v in result.x)}]**")
                st.write(f"**f(x) = {result.fun:.6f}**")
                st.write(f"**Ограничение выполнено: {constraint_func(result.x):.6f}**")
                st.write(f"**Успех: {result.success}**")

                if len(x0) == 2:
                    x1 = np.linspace(-2, 2, 100)
                    x2 = np.linspace(-2, 2, 100)
                    X1, X2 = np.meshgrid(x1, x2)
                    Z_obj = np.zeros_like(X1)
                    Z_constr = np.zeros_like(X1)
                    for i in range(X1.shape[0]):
                        for j in range(X1.shape[1]):
                            pt = [X1[i, j], X2[i, j]]
                            Z_obj[i, j] = objective(pt)
                            Z_constr[i, j] = constraint_func(pt)

                    fig = go.Figure()
                    fig.add_trace(go.Contour(x=x1, y=x2, z=Z_obj, colorscale='Blues', showscale=True, name='f(x)'))
                    if constr_type == "Равенство (=0)":
                        fig.add_trace(go.Contour(x=x1, y=x2, z=Z_constr, contours=dict(coloring='lines', showlabels=True, start=0, end=0, size=0.1), line_width=2, showscale=False, name='g(x)=0'))
                    else:
                        fig.add_trace(go.Contour(x=x1, y=x2, z=Z_constr, contours=dict(coloring='lines', showlabels=True, start=0, end=0, size=0.1), line_width=2, showscale=False, name='g(x)≥0'))
                    fig.add_trace(go.Scatter(x=[result.x[0]], y=[result.x[1]], mode='markers', marker=dict(size=12, color='red'), name='Решение'))
                    fig.update_layout(title="Оптимизация с ограничениями", template=graph_theme)
                    st.plotly_chart(fig, use_container_width=True)

                save_to_history(f"Оптимизация с ограничениями: {func_constr}", f"x={result.x}")
            except Exception as e:
                st.error(f"Ошибка: {str(e)}")

    elif opt_type == "Подбор параметров кривой":
        st.subheader("Аппроксимация данных полиномом")
        data_points = st.text_area("Введите точки (x,y) по одной в строке",
                                   value="0,1\n1,2\n2,5\n3,10\n4,17\n5,26")
        degree = st.slider("Степень полинома", 1, 10, 2)
        if st.button("Подобрать кривую", type="primary"):
            try:
                points = [
                    line.split(',') for line in data_points.strip().split('\n')
                ]
                x_data = np.array([float(p[0]) for p in points])
                y_data = np.array([float(p[1]) for p in points])
                coeffs = np.polyfit(x_data, y_data, degree)
                poly = np.poly1d(coeffs)
                x_smooth = np.linspace(x_data.min(), x_data.max(), 300)
                y_smooth = poly(x_smooth)
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(x=x_data,
                               y=y_data,
                               mode='markers',
                               marker=dict(size=10, color='blue'),
                               name='Данные'))
                fig.add_trace(
                    go.Scatter(x=x_smooth,
                               y=y_smooth,
                               mode='lines',
                               line=dict(color='red', width=2),
                               name=f'Полином степени {degree}'))
                fig.update_layout(title="Аппроксимация данных",
                                  xaxis_title="x",
                                  yaxis_title="y",
                                  template=graph_theme)
                st.plotly_chart(fig, use_container_width=True)
                st.success("Коэффициенты полинома:")
                st.code(str(poly))
                mse = np.mean((y_data - poly(x_data))**2)
                st.metric("Среднеквадратичная ошибка", f"{mse:.6f}")
                save_to_history(f"Аппроксимация полиномом (степень {degree})", f"MSE={mse:.6f}")
            except Exception as e:
                st.error(f"Ошибка: {str(e)}")

with tabs[8]:
    st.header("Анализ данных")

    analysis_type = st.selectbox("Тип анализа", [
        "Преобразование Фурье", "Корреляция и ковариация",
        "Регрессионный анализ"
    ])

    if analysis_type == "Преобразование Фурье":
        st.subheader("FFT - Быстрое преобразование Фурье")

        signal_type = st.selectbox(
            "Тип сигнала", ["Синус", "Сумма синусов", "Пользовательский"])

        if signal_type == "Синус":
            freq = st.slider("Частота (Гц)", 1.0, 50.0, 5.0)
            amplitude = st.slider("Амплитуда", 0.1, 10.0, 1.0)

            t = np.linspace(0, 1, 1000)
            signal = amplitude * np.sin(2 * np.pi * freq * t)

        elif signal_type == "Сумма синусов":
            st.write("Создаем сигнал: sin(2πf₁t) + 0.5·sin(2πf₂t)")
            col1, col2 = st.columns(2)
            with col1:
                f1 = st.number_input("Частота f₁", value=5.0)
            with col2:
                f2 = st.number_input("Частота f₂", value=20.0)

            t = np.linspace(0, 1, 1000)
            signal = np.sin(
                2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)

        if st.button("Выполнить FFT", type="primary"):
            fft_result = fft.fft(signal)
            freqs = fft.fftfreq(len(signal), t[1] - t[0])

            fig = make_subplots(rows=2,
                                cols=1,
                                subplot_titles=("Исходный сигнал",
                                                "Спектр Фурье"))

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
                                     name='Амплитудный спектр'),
                          row=2,
                          col=1)

            fig.update_xaxes(title_text="Время (с)", row=1, col=1)
            fig.update_xaxes(title_text="Частота (Гц)", row=2, col=1)
            fig.update_yaxes(title_text="Амплитуда", row=1, col=1)
            fig.update_yaxes(title_text="Мощность", row=2, col=1)

            fig.update_layout(height=800, template=graph_theme)

            st.plotly_chart(fig, use_container_width=True)

            dominant_freq = freqs[mask][np.argmax(np.abs(fft_result[mask]))]
            st.success(f"Доминирующая частота: {dominant_freq:.2f} Гц")

    elif analysis_type == "Корреляция и ковариация":
        st.subheader("Анализ связи между переменными")

        n_points = st.slider("Количество точек", 10, 200, 50)

        col1, col2 = st.columns(2)
        with col1:
            correlation = st.slider("Корреляция", -1.0, 1.0, 0.8)
        with col2:
            noise_level = st.slider("Уровень шума", 0.0, 2.0, 0.5)

        if st.button("Сгенерировать данные", type="primary"):
            mean = [0, 0]
            cov = [[1, correlation], [correlation, 1]]
            data = np.random.multivariate_normal(mean, cov, n_points)
            data += np.random.normal(0, noise_level, data.shape)

            x_data = data[:, 0]
            y_data = data[:, 1]

            actual_corr = np.corrcoef(x_data, y_data)[0, 1]
            actual_cov = np.cov(x_data, y_data)[0, 1]

            fig = px.scatter(x=x_data,
                             y=y_data,
                             labels={
                                 'x': 'X',
                                 'y': 'Y'
                             },
                             title="Диаграмма рассеяния")

            z = np.polyfit(x_data, y_data, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x_data.min(), x_data.max(), 100)

            fig.add_trace(
                go.Scatter(x=x_line,
                           y=p(x_line),
                           mode='lines',
                           name='Линия тренда',
                           line=dict(color='red', width=2)))

            fig.update_layout(template=graph_theme, height=600)
            st.plotly_chart(fig, use_container_width=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Корреляция", f"{actual_corr:.4f}")
            with col2:
                st.metric("Ковариация", f"{actual_cov:.4f}")
            with col3:
                st.metric("R²", f"{actual_corr**2:.4f}")

    elif analysis_type == "Регрессионный анализ":
        st.subheader("Множественная линейная регрессия")

        st.write("Генерация случайных данных для демонстрации")

        n_samples = st.slider("Количество наблюдений", 20, 200, 50)
        n_features = st.slider("Количество признаков", 1, 5, 2)

        if st.button("Выполнить регрессию", type="primary"):
            X = np.random.randn(n_samples, n_features)
            true_coeffs = np.random.randn(n_features)
            y = X @ true_coeffs + np.random.randn(n_samples) * 0.5

            coeffs = linalg.lstsq(X, y)[0]
            y_pred = X @ coeffs

            residuals = y - y_pred
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r_squared = 1 - (ss_res / ss_tot)

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=y, y=y_pred, mode='markers', name='Предсказания'))

            fig.add_trace(
                go.Scatter(x=[y.min(), y.max()],
                           y=[y.min(), y.max()],
                           mode='lines',
                           name='Идеальная линия',
                           line=dict(dash='dash', color='red')))

            fig.update_layout(title="Фактические vs Предсказанные значения",
                              xaxis_title="Фактические",
                              yaxis_title="Предсказанные",
                              template=graph_theme)

            st.plotly_chart(fig, use_container_width=True)

            st.success("Коэффициенты регрессии:")
            for i, coef in enumerate(coeffs):
                st.write(f"β_{i+1} = {coef:.4f}")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R²", f"{r_squared:.4f}")
            with col2:
                st.metric("RMSE", f"{np.sqrt(np.mean(residuals**2)):.4f}")
            with col3:
                st.metric("MAE", f"{np.mean(np.abs(residuals)):.4f}")

with tabs[9]:
    st.header("Решение дифференциальных уравнений")
    ode_type = st.selectbox("Тип задачи", [
        "ОДУ 1-го порядка: dy/dt = f(t, y)",
        "ОДУ 2-го порядка → система",
        "Система ОДУ"
    ])

    if ode_type == "ОДУ 1-го порядка: dy/dt = f(t, y)":
        st.subheader("dy/dt = f(t, y)")
        f_expr = st.text_input("Правая часть f(t, y)", value="t - y")
        t0 = st.number_input("Начальное время t₀", value=0.0)
        y0 = st.number_input("Начальное значение y(t₀)", value=1.0)
        t_end = st.number_input("Конечное время", value=5.0)
        steps = st.slider("Число шагов", 50, 1000, 200)

        if st.button("Решить ОДУ", type="primary"):
            try:
                t_sym, y_sym = sp.symbols('t y')
                f_sympy = sp.sympify(f_expr)
                f_func = sp.lambdify((t_sym, y_sym), f_sympy, 'numpy')

                def dydt(t, y):
                    return f_func(t, y)

                t_eval = np.linspace(t0, t_end, steps)
                sol = integrate.solve_ivp(dydt, [t0, t_end], [y0], t_eval=t_eval, method='RK45')

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=sol.t, y=sol.y[0], mode='lines', name='y(t)'))
                fig.update_layout(title=f"Решение dy/dt = {f_expr}", xaxis_title="t", yaxis_title="y", template=graph_theme)
                st.plotly_chart(fig, use_container_width=True)

                st.success("Решение успешно построено!")
                save_to_history(f"ОДУ 1-го порядка: {f_expr}", f"y({t0})={y0}")
            except Exception as e:
                st.error(f"Ошибка при решении ОДУ: {str(e)}")

    elif ode_type == "ОДУ 2-го порядка → система":
        st.subheader("Приведение y'' = f(t, y, y') к системе")
        st.markdown("Пусть **y₁ = y**, **y₂ = y'**, тогда:\n- y₁' = y₂\n- y₂' = f(t, y₁, y₂)")
        f_expr = st.text_input("f(t, y, dy/dt)", value="-y")
        t0 = st.number_input("t₀", value=0.0)
        y0 = st.number_input("y(t₀)", value=1.0)
        dy0 = st.number_input("y'(t₀)", value=0.0)
        t_end = st.number_input("Конечное время", value=10.0)
        steps = st.slider("Число шагов", 50, 1000, 300)

        if st.button("Решить ОДУ 2-го порядка", type="primary"):
            try:
                t_sym, y_sym, dy_sym = sp.symbols('t y dy')
                f_sympy = sp.sympify(f_expr)
                f_func = sp.lambdify((t_sym, y_sym, dy_sym), f_sympy, 'numpy')

                def system(t, Y):
                    y1, y2 = Y
                    return [y2, f_func(t, y1, y2)]

                t_eval = np.linspace(t0, t_end, steps)
                sol = integrate.solve_ivp(system, [t0, t_end], [y0, dy0], t_eval=t_eval, method='RK45')

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=sol.t, y=sol.y[0], mode='lines', name='y(t)'))
                fig.add_trace(go.Scatter(x=sol.t, y=sol.y[1], mode='lines', name="y'(t)", line=dict(dash='dash')))
                fig.update_layout(title=f"Решение y'' = {f_expr}", xaxis_title="t", yaxis_title="y, y'", template=graph_theme)
                st.plotly_chart(fig, use_container_width=True)

                st.success("Система решена!")
                save_to_history(f"ОДУ 2-го порядка: y'' = {f_expr}", f"y({t0})={y0}, y'({t0})={dy0}")
            except Exception as e:
                st.error(f"Ошибка: {str(e)}")

    elif ode_type == "Система ОДУ":
        st.subheader("Система: dY/dt = F(t, Y)")
        n_eq = st.number_input("Число уравнений", 2, 4, 2)
        equations = []
        initials = []
        for i in range(n_eq):
            eq = st.text_input(f"dy{i}/dt =", value="y1" if i == 0 else "-y0", key=f"ode_eq_{i}")
            y0_val = st.number_input(f"y{i}(t₀)", value=1.0 if i == 0 else 0.0, key=f"ode_y0_{i}")
            equations.append(eq)
            initials.append(y0_val)

        t0 = st.number_input("Начальное время t₀", value=0.0, key="sys_t0")
        t_end = st.number_input("Конечное время", value=10.0, key="sys_tend")
        steps = st.slider("Число шагов", 50, 1000, 200, key="sys_steps")

        if st.button("Решить систему ОДУ", type="primary"):
            try:
                t_sym = sp.Symbol('t')
                y_syms = sp.symbols(' '.join([f'y{i}' for i in range(n_eq)]))
                funcs = []
                for eq in equations:
                    expr = sp.sympify(eq)
                    funcs.append(sp.lambdify((t_sym,) + y_syms, expr, 'numpy'))

                def sys_ode(t, Y):
                    return [f(t, *Y) for f in funcs]

                t_eval = np.linspace(t0, t_end, steps)
                sol = integrate.solve_ivp(sys_ode, [t0, t_end], initials, t_eval=t_eval, method='RK45')

                fig = go.Figure()
                for i in range(n_eq):
                    fig.add_trace(go.Scatter(x=sol.t, y=sol.y[i], mode='lines', name=f'y{i}(t)'))
                fig.update_layout(title="Решение системы ОДУ", xaxis_title="t", yaxis_title="yᵢ(t)", template=graph_theme)
                st.plotly_chart(fig, use_container_width=True)

                st.success("Система решена!")
                save_to_history(f"Система ОДУ: {', '.join(equations)}", f"Начальные условия: {initials}")
            except Exception as e:
                st.error(f"Ошибка: {str(e)}")

st.markdown("---")
st.markdown(
    "<div>"
    "🧮 <b>Математический калькулятор</b><br>"
    "<span style='font-size: 0.8rem;'>"
    "Использует: Streamlit • NumPy • SciPy • SymPy • Plotly • Pandas<br>"
    "2025 | by quik"
    "</span>"
    "</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div style='font-size: 0.8rem;'>"
    "Исходный код: <a href='https://github.com/QUIK1001/Event-Horizon-Shell'>GitHub</a><br>"
    "Контакты: <a href='mailto:belov0507artem@gmail.com'>belov0507artem@gmail.com</a>"
    "</div>",
    unsafe_allow_html=True
)
