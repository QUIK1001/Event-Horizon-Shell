import requests
from colorama import Fore, Style
from .chklib_clrscr import clear_screen
from .logger import logger
from .constants import config

def weather():
    """
    Get current weather information for a specified city.
    Uses OpenWeatherMap API to fetch weather data.
    """
    logger.log("Starting weather function", "INFO")
    try:
        city = input("⣦ Enter city name: ")
        logger.log(f"Weather request for city: {city}", "INFO")

        params = {
            'q': city,
            'appid': config.API_KEY,
            'units': 'metric',
            'lang': 'ru'
        }

        response = requests.get(config.BASE_URL, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()

        temperature = data['main']['temp']
        feels_like = data['main']['feels_like']
        humidity = data['main']['humidity']
        wind_speed = data['wind']['speed']
        description = data['weather'][0]['description'].capitalize()
        city_name = data['name']
        country = data['sys']['country']

        if temperature > 30:
            temp_color = Fore.RED
        elif temperature > 20:
            temp_color = Fore.YELLOW
        elif temperature > 10:
            temp_color = Fore.GREEN
        else:
            temp_color = Fore.CYAN

        print(f"\n{Fore.BLUE}⣦ Weather in {city_name}, {country}:{Style.RESET_ALL}")
        print(f"{temp_color} Temperature: {temperature}°C (feels like {feels_like}°C){Style.RESET_ALL}")
        print(f"{Fore.LIGHTBLUE_EX} Condition: {description}{Style.RESET_ALL}")
        print(f"{Fore.LIGHTCYAN_EX} Humidity: {humidity}%{Style.RESET_ALL}")
        print(f"{Fore.LIGHTMAGENTA_EX} Wind speed: {wind_speed} m/s{Style.RESET_ALL}")

        logger.log(f"Weather displayed for {city_name}: {temperature}°C, {description}", "OK")

    except requests.exceptions.RequestException as e:
        logger.log(f"Weather API error: {str(e)}", "FAIL")
        print(f"{Fore.RED}⣏!⣽ Error connecting to weather service{Style.RESET_ALL}")
    except KeyError as e:
        logger.log(f"Invalid weather data received: {str(e)}", "FAIL")
        print(f"{Fore.RED}⣏!⣽ City not found or invalid data{Style.RESET_ALL}")
    except Exception as e:
        logger.log(f"Unexpected error in weather function: {str(e)}", "FAIL")
        print(f"{Fore.RED}⣏!⣽ Error: {str(e)}{Style.RESET_ALL}")

    input("Press Enter to continue")
    clear_screen()
