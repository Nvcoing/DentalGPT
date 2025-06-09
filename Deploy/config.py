def load_ngrok_url(filepath="config.txt") -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        url = f.readline().strip()
    return url + "/model/generate/"

NGROK_URL = load_ngrok_url()
