import requests
from dotenv import load_dotenv
import os

load_dotenv()

BASE_URL = os.getenv("BASE_URL")


def reload_model():
    url = f"{BASE_URL}/reload-model/"
    response = requests.post(url)
    if response.status_code == 200:
        print("Model reloaded successfully")
    else:
        print(f"Failed to reload model: {response.status_code}")


if __name__ == "__main__":
    reload_model()
