# Move to EHS/ folder

import random
import json
import time
import os
import math
from fractions import Fraction
from colorama import init, Fore, Style

init(autoreset=True)

CONFIG_FILE = "quiz_config.json"
QUESTIONS_FILE = "custom_questions.json"
STATS_FILE = "quiz_stats.json"
MISTAKES_FILE = "quiz_mistakes.json"

def format_fraction(frac: Fraction) -> str:
    if frac.denominator == 1:
        return str(frac.numerator)
    sign = "-" if frac.numerator < 0 else ""
    num = abs(frac.numerator)
    den = frac.denominator
    if num < den:
        return f"{sign}{num}/{den}"
    else:
        whole = num // den
        remainder = num % den
        if remainder == 0:
            return f"{sign}{whole}"
        else:
            return f"{sign}{whole} {remainder}/{den}"

def round_or_int(x):
    """Округляет до 2 знаков, но убирает лишние нули."""
    if x == int(x):
        return str(int(x))
    return f"{x:.2f}".rstrip('0').rstrip('.')

def gen_linear_equation():
    a = random.randint(1, 10)
    x_true = random.randint(-10, 10)
    b = random.randint(-10, 10)
    c = a * x_true + b
    question = f"Решите уравнение: {a}x {'+' if b >= 0 else ''}{b} = {c}"
    correct = str(x_true)
    options = [correct]
    for _ in range(3):
        fake = str(x_true + random.randint(-5, 5))
        if fake != correct and fake not in options:
            options.append(fake)
    random.shuffle(options)
    return {"question": question, "options": options, "answer": options.index(correct)}

def gen_area():
    shape = random.choice(["прямоугольник", "треугольник", "круг"])
    if shape == "прямоугольник":
        a = random.randint(2, 15)
        b = random.randint(2, 15)
        area = a * b
        question = f"Найдите площадь прямоугольника со сторонами {a} и {b}."
        correct = str(area)
    elif shape == "треугольник":
        base = random.randint(2, 20)
        height = random.randint(2, 15)
        area = (base * height) / 2
        question = f"Найдите площадь треугольника с основанием {base} и высотой {height}."
        correct = round_or_int(area)
    else:
        r = random.randint(1, 7)
        area = math.pi * r * r
        question = f"Найдите площадь круга радиусом {r}. Ответ округлите до целого."
        correct = str(round(area))
    
    options = [correct]
    for _ in range(3):
        noise = float(correct) + random.randint(-10, 10)
        fake = str(max(1, int(noise)))
        if fake != correct and fake not in options:
            options.append(fake)
    random.shuffle(options)
    return {"question": question, "options": options, "answer": options.index(correct)}

def gen_coordinates():
    task_type = random.choice(["расстояние", "середина"])
    x1, y1 = random.randint(-10, 10), random.randint(-10, 10)
    x2, y2 = random.randint(-10, 10), random.randint(-10, 10)
    if x1 == x2 and y1 == y2:
        x2 += 1

    if task_type == "расстояние":
        dx = x2 - x1
        dy = y2 - y1
        dist = math.sqrt(dx*dx + dy*dy)
        question = f"Найдите расстояние между точками A({x1}, {y1}) и B({x2}, {y2}). Округлите до целого."
        correct = str(round(dist))
        options = [correct]
        for _ in range(3):
            fake = str(max(1, round(dist + random.randint(-3, 3))))
            if fake != correct and fake not in options:
                options.append(fake)
    else:
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        question = f"Найдите координаты середины отрезка AB, где A({x1}, {y1}), B({x2}, {y2})."
        mx_str = round_or_int(mx)
        my_str = round_or_int(my)
        correct = f"({mx_str}, {my_str})"
        options = [correct]
        for _ in range(3):
            fake_mx = round_or_int(mx + random.choice([-1, 1, 0.5, -0.5]))
            fake_my = round_or_int(my + random.choice([-1, 1, 0.5, -0.5]))
            fake = f"({fake_mx}, {fake_my})"
            if fake != correct and fake not in options:
                options.append(fake)
    
    random.shuffle(options)
    return {"question": question, "options": options, "answer": options.index(correct)}

def gen_graph():
    k = random.randint(-3, 3)
    while k == 0:
        k = random.randint(-3, 3)
    b = random.randint(-5, 5)
    x_val = random.randint(-3, 3)
    y_val = k * x_val + b
    question = f"Функция задана формулой y = {k}x {'+' if b >= 0 else ''}{b}. Чему равно y при x = {x_val}?"
    correct = str(y_val)
    options = [correct]
    for _ in range(3):
        fake = str(y_val + random.randint(-5, 5))
        if fake != correct and fake not in options:
            options.append(fake)
    random.shuffle(options)
    return {"question": question, "options": options, "answer": options.index(correct)}

def gen_fraction_operation():
    f1 = Fraction(random.randint(1, 9), random.randint(2, 10))
    f2 = Fraction(random.randint(1, 9), random.randint(2, 10))
    op = random.choice(['+', '-'])
    result = f1 + f2 if op == '+' else f1 - f2
    question = f"Чему равна {'сумма' if op == '+' else 'разность'} дробей {f1} и {f2}?"
    correct_str = format_fraction(result)
    options = [correct_str]
    attempts = 0
    while len(options) < 4 and attempts < 30:
        attempts += 1
        noise_num = result.numerator + random.randint(-5, 5)
        noise_den = result.denominator + random.randint(-3, 3)
        if noise_den <= 0:
            continue
        try:
            fake_frac = Fraction(noise_num, noise_den)
            fake_str = format_fraction(fake_frac)
            if fake_str != correct_str and fake_str not in options:
                options.append(fake_str)
        except:
            continue
    while len(options) < 4:
        fake = str(random.randint(-20, 20))
        if fake not in options:
            options.append(fake)
    random.shuffle(options)
    return {"question": question, "options": options, "answer": options.index(correct_str)}

def gen_fraction_to_decimal():
    num = random.randint(1, 20)
    den = random.choice([2, 4, 5, 8, 10, 20, 25, 50])
    frac = Fraction(num, den)
    decimal_val = round(float(frac), 2)
    correct_str = f"{decimal_val:.2f}".rstrip('0').rstrip('.')
    if '.' not in correct_str:
        correct_str += ".0"
    question = f"Переведите дробь {frac} в десятичную дробь (округлите до сотых)."
    options = [correct_str]
    for _ in range(3):
        noise = round(decimal_val + random.uniform(-0.5, 0.5), 2)
        fake = f"{noise:.2f}".rstrip('0').rstrip('.')
        if '.' not in fake:
            fake += ".0"
        if fake not in options:
            options.append(fake)
    random.shuffle(options)
    return {"question": question, "options": options, "answer": options.index(correct_str)}

def gen_percent_of_number():
    percent = random.choice([10, 20, 25, 30, 40, 50, 75])
    for _ in range(20):
        number = random.randint(10, 200)
        if (percent * number) % 100 == 0:
            break
    else:
        number = 100
    result = (percent * number) // 100
    question = f"Чему равны {percent}% от числа {number}?"
    correct_str = str(result)
    options = [correct_str]
    for _ in range(3):
        fake = str(result + random.randint(-15, 15))
        if fake != correct_str and fake not in options and int(fake) >= 0:
            options.append(fake)
    random.shuffle(options)
    return {"question": question, "options": options, "answer": options.index(correct_str)}

def gen_round_decimal():
    num = round(random.uniform(5.0, 99.999), 3)
    place_name, digits = random.choice([("целых", 0), ("десятых", 1), ("сотых", 2)])
    rounded = round(num, digits)
    correct_str = f"{rounded:.{digits}f}" if digits > 0 else str(int(rounded))
    question = f"Округлите число {num:.3f} до {place_name}."
    options = [correct_str]
    for _ in range(3):
        noise = round(num + random.uniform(-1, 1), digits)
        fake = f"{noise:.{digits}f}" if digits > 0 else str(int(noise))
        if fake not in options:
            options.append(fake)
    random.shuffle(options)
    return {"question": question, "options": options, "answer": options.index(correct_str)}

def gen_compare_fractions():
    a = Fraction(random.randint(1, 9), random.randint(2, 10))
    b = Fraction(random.randint(1, 9), random.randint(2, 10))
    if a == b:
        return gen_compare_fractions()
    question = f"Какая дробь больше: {a} или {b}?"
    correct = "первая" if a > b else "вторая"
    options = ["первая", "вторая", "равны", "одинаковые"]
    random.shuffle(options)
    try:
        ans_idx = options.index(correct)
    except ValueError:
        options = ["первая", "вторая", "не знаю", "одинаковые"]
        ans_idx = 0 if correct == "первая" else 1
    return {"question": question, "options": options, "answer": ans_idx}

ALL_GENERATORS = [
    gen_fraction_operation,
    gen_fraction_to_decimal,
    gen_percent_of_number,
    gen_round_decimal,
    gen_compare_fractions,
    gen_linear_equation,
    gen_area,
    gen_coordinates,
    gen_graph
]

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                return cfg.get("num_questions", 5)
        except:
            pass
    return 5

def save_config(num_questions):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump({"num_questions": num_questions}, f, indent=4)

def load_custom_questions():
    if not os.path.exists(QUESTIONS_FILE):
        return []
    try:
        with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []

def save_custom_questions(qs):
    with open(QUESTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(qs, f, ensure_ascii=False, indent=4)

def add_custom_question(custom_qs):
    print(f"\n{Fore.CYAN}Добавьте свой вопрос{Style.RESET_ALL}")
    q_text = input("Текст вопроса: ").strip()
    if not q_text:
        print(f"{Fore.RED}! Пустой вопрос.{Style.RESET_ALL}")
        return
    
    options = []
    print("Введите варианты (минимум 2, Enter для завершения):")
    while True:
        opt = input(f"Вариант {len(options)+1}: ").strip()
        if not opt:
            if len(options) >= 2:
                break
            else:
                print(f"{Fore.YELLOW}! Нужно минимум 2 варианта.{Style.RESET_ALL}")
                continue
        if opt in options:
            print(f"{Fore.YELLOW}! Такой вариант уже есть.{Style.RESET_ALL}")
            continue
        options.append(opt)
    
    print(f"\nВарианты ответов:")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    
    while True:
        try:
            ans = int(input(f"Номер правильного ответа (1–{len(options)}): "))
            if 1 <= ans <= len(options):
                break
            print(f"{Fore.YELLOW}! Неверный номер.{Style.RESET_ALL}")
        except ValueError:
            print(f"{Fore.YELLOW}! Введите число.{Style.RESET_ALL}")
    
    custom_qs.append({"question": q_text, "options": options, "answer": ans - 1})
    save_custom_questions(custom_qs)
    print(f"{Fore.GREEN} Вопрос сохранён!{Style.RESET_ALL}")

def delete_custom_question(custom_qs):
    """Удаляет выбранный пользовательский вопрос."""
    if not custom_qs:
        print(f"{Fore.YELLOW}! Нет своих вопросов для удаления.{Style.RESET_ALL}")
        return

    print(f"\n{Fore.CYAN}Удаление вопроса{Style.RESET_ALL}")
    for i, q in enumerate(custom_qs, 1):
        print(f"{i}. {q['question']}")
    
    try:
        choice = int(input(f"\nВведите номер вопроса для удаления (1–{len(custom_qs)}), или 0 для отмены: "))
        if choice == 0:
            print("! Удаление отменено.")
            return
        if 1 <= choice <= len(custom_qs):
            removed = custom_qs.pop(choice - 1)
            save_custom_questions(custom_qs)
            print(f"{Fore.GREEN}Вопрос удалён: {removed['question']}{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}! Неверный номер.{Style.RESET_ALL}")
    except ValueError:
        print(f"{Fore.RED}! Введите число.{Style.RESET_ALL}")

def load_statistics():
    """Загружает статистику из файла"""
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    return {"total_quizzes": 0, "best_score": 0, "average_score": 0, "total_questions": 0, "correct_answers": 0}

def save_statistics(stats):
    """Сохраняет статистику в файл"""
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)

def update_statistics(score, total_questions):
    """Обновляет статистику после теста"""
    stats = load_statistics()
    
    stats["total_quizzes"] += 1
    stats["total_questions"] += total_questions
    stats["correct_answers"] += score
    
    percent = (score / total_questions) * 100
    if percent > stats["best_score"]:
        stats["best_score"] = round(percent, 1)
    
    if stats["total_quizzes"] > 0:
        total_percent = (stats["correct_answers"] / stats["total_questions"]) * 100
        stats["average_score"] = round(total_percent, 1)
    
    save_statistics(stats)

def show_statistics():
    """Показывает статистику"""
    stats = load_statistics()
    
    print(f"\n{Fore.CYAN}СТАТИСТИКА{Style.RESET_ALL}")
    print(f"{'='*30}")
    print(f"Всего тестов пройдено: {stats['total_quizzes']}")
    print(f"Всего вопросов отвечено: {stats['total_questions']}")
    print(f"Правильных ответов: {stats['correct_answers']}")
    
    if stats['total_quizzes'] > 0:
        print(f"Лучший результат: {stats['best_score']}%")
        print(f"Средний результат: {stats['average_score']}%")
        
        if stats['average_score'] >= 90:
            level = "Математик!"
        elif stats['average_score'] >= 75:
            level = "Отличник!"
        elif stats['average_score'] >= 60:
            level = "Хорошист!"
        else:
            level = "Новичок"
        
        print(f"Ваш уровень: {level}")
    else:
        print(f"{Fore.YELLOW}Пройдите хотя бы один тест для получения статистики{Style.RESET_ALL}")

def load_mistakes():
    """Загружает список ошибок"""
    if os.path.exists(MISTAKES_FILE):
        try:
            with open(MISTAKES_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    return []

def save_mistakes(mistakes):
    """Сохраняет список ошибок"""
    with open(MISTAKES_FILE, "w", encoding="utf-8") as f:
        json.dump(mistakes, f, ensure_ascii=False, indent=4)

def add_mistake(question_data, user_answer, correct_answer):
    """Добавляет ошибку в список"""
    mistakes = load_mistakes()
    
    mistake = {
        "question": question_data["question"],
        "user_answer": user_answer,
        "correct_answer": correct_answer,
        "options": question_data["options"],
        "date": time.strftime("%Y-%m-%d %H:%M")
    }
    
    for existing in mistakes:
        if (existing["question"] == mistake["question"] and 
            existing["user_answer"] == mistake["user_answer"]):
            return
    
    mistakes.append(mistake)
    save_mistakes(mistakes)

def review_mistakes():
    """Показывает и позволяет повторить ошибки"""
    mistakes = load_mistakes()
    
    if not mistakes:
        print(f"\n{Fore.GREEN}Поздравляем! У вас нет ошибок для повторения!{Style.RESET_ALL}")
        return
    
    print(f"\n{Fore.RED}ПОВТОРЕНИЕ ОШИБОК ({len(mistakes)} вопросов){Style.RESET_ALL}")
    print(f"{'='*40}")
    
    correct_count = 0
    for i, mistake in enumerate(mistakes, 1):
        print(f"\n{Fore.YELLOW}Вопрос {i}:{Style.RESET_ALL} {mistake['question']}")
        print(f"{Fore.RED}Ваш ответ: {mistake['user_answer']}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Правильный ответ: {mistake['correct_answer']}{Style.RESET_ALL}")
        
        response = input("\nЗапомнили правильный ответ? (д/н): ").strip().lower()
        if response in ['д', 'да', 'y', 'yes']:
            correct_count += 1
    
    if correct_count == len(mistakes):
        save_mistakes([])
        print(f"\n{Fore.GREEN}Все ошибки исправлены! Список очищен.{Style.RESET_ALL}")
    else:
        print(f"\n{Fore.YELLOW}Осталось ошибок для повторения: {len(mistakes) - correct_count}{Style.RESET_ALL}")

def clear_mistakes():
    """Очищает список ошибок"""
    if os.path.exists(MISTAKES_FILE):
        os.remove(MISTAKES_FILE)
    print(f"{Fore.GREEN}Список ошибок очищен{Style.RESET_ALL}")

def run_quiz(num_questions):
    start_time = time.time()
    questions = []
    for _ in range(num_questions):
        gen = random.choice(ALL_GENERATORS)
        questions.append(gen())
    
    custom = load_custom_questions()
    all_qs = questions + custom
    random.shuffle(all_qs)
    
    if not all_qs:
        print(f"{Fore.RED}! Нет вопросов.{Style.RESET_ALL}")
        return

    score = 0
    wrong_answers = []
    
    for i, q in enumerate(all_qs):
        print(f"\n{Fore.BLUE}Вопрос {i+1}:{Style.RESET_ALL} {q['question']}")
        for j, opt in enumerate(q['options'], 1):
            print(f"  {j}. {opt}")
        try:
            user = int(input("Ваш ответ (номер): ")) - 1
            if user == q['answer']:
                print(f"{Fore.GREEN} Верно!{Style.RESET_ALL}")
                score += 1
            else:
                user_answer = q['options'][user] if 0 <= user < len(q['options']) else "неверный номер"
                correct_answer = q['options'][q['answer']]
                print(f"{Fore.RED}! Неверно.{Style.RESET_ALL} Правильный ответ: {correct_answer}")
                add_mistake(q, user_answer, correct_answer)
                
        except (ValueError, IndexError):
            print(f"{Fore.YELLOW}! Некорректный ввод.{Style.RESET_ALL}")
    end_time = time.time()
    duration = end_time - start_time
    total = len(all_qs)
    if total > 0:
        percent = score / total * 100
        color = Fore.GREEN if percent >= 80 else Fore.YELLOW if percent >= 60 else Fore.RED
        print(f"\n{Style.BRIGHT}Результат:{Style.RESET_ALL} {score} из {total} ({color}{percent:.1f}%{Style.RESET_ALL})")
        print(f"Время прохождения: {duration:.1f} секунд")
        update_statistics(score, total)
        
        if score < total:
            print(f"\n{Fore.CYAN}Хотите повторить вопросы, где ошиблись? (д/н){Style.RESET_ALL}")
            if input().strip().lower() in ['д', 'да', 'y', 'yes']:
                review_mistakes()

def show_file_paths():
    """Показывает пути к конфигурационным файлам"""
    print(f"\n{Fore.GREEN}Пути к файлам:{Style.RESET_ALL}")
    print(f"Файл вопросов: {os.path.abspath(QUESTIONS_FILE)}")
    print(f"Файл настроек: {os.path.abspath(CONFIG_FILE)}")
    print(f"\n{Fore.YELLOW}Скопируйте эти пути для переноса файлов{Style.RESET_ALL}")

def settings_menu():
    num = load_config()
    print(f"\n{Fore.CYAN}Настройки{Style.RESET_ALL}")
    print(f"Текущее количество случайных вопросов: {num}")
    try:
        new_num = input("Введите новое количество (1–20, Enter — оставить): ").strip()
        if new_num:
            new_num = int(new_num)
            if 1 <= new_num <= 20:
                save_config(new_num)
                print(f"{Fore.GREEN} Количество вопросов изменено на {new_num}.{Style.RESET_ALL}")
                return new_num
            else:
                print(f"{Fore.RED}! Допустимо от 1 до 20.{Style.RESET_ALL}")

    except ValueError:
        print(f"{Fore.RED}! Введите число.{Style.RESET_ALL}")
    return num

def main():
    try:
        num_questions = load_config()
        custom = load_custom_questions()
        while True:
            print(f"\n{Style.BRIGHT}{'='*40}{Style.RESET_ALL}")
            print(f"{Fore.LIGHTBLUE_EX}Математический тренажёр{Style.RESET_ALL}")
            print(f"1. Пройти тест ({num_questions} вопросов)")
            print("2. Добавить свой вопрос")
            print("3. Показать свои вопросы")
            print("4. Удалить вопрос")
            print("5. Показать путь к файлам с настройками") 
            print("6. Статистика")
            print("7. Повторить ошибки")
            print("8. Очистить ошибки")
            print("9. Настройки")
            print("10. Выход")
            print(f"{Style.BRIGHT}{'='*40}{Style.RESET_ALL}")
            choice = input("Выберите действие: ").strip()
            if choice == "1":
                run_quiz(num_questions)
            elif choice == "2":
                add_custom_question(custom)
                custom = load_custom_questions()
            elif choice == "3":
                if not custom:
                    print(f"{Fore.YELLOW}! Нет своих вопросов.{Style.RESET_ALL}")
                else:
                    for i, q in enumerate(custom, 1):
                        print(f"\n{i}. {q['question']}")
                        for j, opt in enumerate(q['options'], 1):
                            mark = f" {Fore.GREEN}←{Style.RESET_ALL}" if j-1 == q['answer'] else ""
                            print(f"   {j}. {opt}{mark}")
            elif choice == "4":
                delete_custom_question(custom)
                custom = load_custom_questions()
            elif choice == "5":
                show_file_paths()
            elif choice == "6":
                show_statistics()
            elif choice == "7":
                review_mistakes()
            elif choice == "8":
                clear_mistakes()
            elif choice == "9":
                num_questions = settings_menu()
            elif choice == "10":
                print(f"{Fore.CYAN}До новых встреч! {Style.RESET_ALL}")
                break
            else:
                print(f"{Fore.YELLOW}! Неверный выбор.{Style.RESET_ALL}")

    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Выход по Ctrl+C.{Style.RESET_ALL}")

if __name__ == "__main__":
    main()

# by quik
