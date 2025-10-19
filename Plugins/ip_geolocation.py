# Move to EHS/ folder.

def main():
    import requests
    from colorama import Fore,Style
    try:
        response = requests.get('https://ipinfo.io/json')
        response.raise_for_status()
        data = response.json()

        print(f"{Fore.GREEN}Ваш IP-адрес: {data.get('ip')}")
        print(f"{Fore.CYAN}Город: {data.get('city')}")
        print(f"{Fore.BLUE}Регион: {data.get('region')}")
        print(f"{Fore.RED}Страна: {data.get('country')}")
        print(f"{Fore.MAGENTA}Координаты (широта, долгота): {data.get('loc')}{Style.RESET_ALL}")
        print(f"{Style.BRIGHT}Часовой пояс: {data.get('timezone')}")

    except requests.exceptions.RequestException as e:
        print(f"{Fore.RED}Ошибка сети или запроса: {e}{Style.RESET_ALL}")

    except ValueError as e:
        print(f"{Fore.RED}Ошибка обработки JSON: {e}{Style.RESET_ALL}")

    except Exception as e:
        print(f"{Fore.RED}Неожиданная ошибка: {e}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
