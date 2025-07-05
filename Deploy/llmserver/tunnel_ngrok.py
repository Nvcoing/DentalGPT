from pyngrok import ngrok

def create_tunnel(port: int, auth_token: str):
    ngrok.set_auth_token(auth_token)
    public_url = ngrok.connect(port).public_url
    print(f"Public URL: {public_url}")
    return public_url
