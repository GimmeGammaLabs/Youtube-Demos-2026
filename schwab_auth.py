import os
import json
import time
import base64
from urllib.parse import urlparse, parse_qs
import requests
from dotenv import load_dotenv

# Load credentials from .env file
load_dotenv()

APP_KEY = os.getenv("SCHWAB_APP_KEY")
APP_SECRET = os.getenv("SCHWAB_APP_SECRET")
REDIRECT_URI = os.getenv("SCHWAB_REDIRECT_URI", "https://127.0.0.1")
TOKEN_FILE = "schwab_tokens.json"

class SchwabAuthenticator:
    def __init__(self):
        self.base_url = "https://api.schwabapi.com/v1/oauth"
        
    def get_basic_auth_header(self) -> str:
        """Schwab requires a Base64 encoded ClientID:ClientSecret string."""
        credentials = f"{APP_KEY}:{APP_SECRET}"
        encoded_creds = base64.b64encode(credentials.encode()).decode()
        return f"Basic {encoded_creds}"

    def generate_auth_url(self) -> str:
        """Step 1: Construct the user login URL."""
        return f"{self.base_url}/authorize?client_id={APP_KEY}&redirect_uri={REDIRECT_URI}"

    def extract_code_from_url(self, returned_url: str) -> str:
        """Helper to safely rip out the authorization code from the redirect URL."""
        try:
            parsed_url = urlparse(returned_url)
            code = parse_qs(parsed_url.query)['code'][0]
            return code
        except Exception as e:
            raise ValueError(f"Failed to parse authorization code: {e}. Ensure you pasted the entire URL.")

    def exchange_code_for_tokens(self, auth_code: str) -> dict:
        """Step 2: Swap the authorization code for Access & Refresh tokens."""
        url = f"{self.base_url}/token"
        headers = {
            "Authorization": self.get_basic_auth_header(),
            "Content-Type": "application/x-www-form-urlencoded"
        }
        payload = {
            "grant_type": "authorization_code",
            "code": auth_code,
            "redirect_uri": REDIRECT_URI
        }
        
        response = requests.post(url, headers=headers, data=payload)
        if response.status_code == 200:
            token_data = response.json()
            # Calculate absolute expiration timestamp (current epoch time + expires_in seconds)
            token_data["expires_at"] = int(time.time()) + token_data["expires_in"]
            self._save_tokens(token_data)
            return token_data
        else:
            raise Exception(f"Token exchange failed: {response.status_code} - {response.text}")

    def refresh_access_token(self, refresh_token: str) -> dict:
        """Step 3: Use the 7-day refresh token to grab a new 30-minute access token."""
        url = f"{self.base_url}/token"
        headers = {
            "Authorization": self.get_basic_auth_header(),
            "Content-Type": "application/x-www-form-urlencoded"
        }
        payload = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token
        }
        
        response = requests.post(url, headers=headers, data=payload)
        if response.status_code == 200:
            token_data = response.json()
            token_data["expires_at"] = int(time.time()) + token_data["expires_in"]
            
            # Carry over old refresh token if the response doesn't provide a new one
            if "refresh_token" not in token_data:
                token_data["refresh_token"] = refresh_token
                
            self._save_tokens(token_data)
            print("🔄 Access token refreshed successfully.")
            return token_data
        else:
            raise Exception(f"Token refresh failed: {response.status_code} - {response.text}")

    def _save_tokens(self, data: dict):
        with open(TOKEN_FILE, "w") as f:
            json.dump(data, f, indent=4)

    def get_valid_token(self) -> str:
        """Main orchestrator function for automated trading bots."""
        if not os.path.exists(TOKEN_FILE):
            print("⚠️ No token file found. You need to run initial authentication.")
            return None
            
        with open(TOKEN_FILE, "r") as f:
            tokens = json.load(f)
            
        # Refresh 2 minutes before actual expiration to handle script delays safely
        if int(time.time()) >= (tokens["expires_at"] - 120):
            print("⏳ Access token expired or close to expiring. Refreshing...")
            tokens = self.refresh_access_token(tokens["refresh_token"])
            
        return tokens["access_token"]

# --- Quick Manual Test Logic ---
if __name__ == "__main__":
    auth = SchwabAuthenticator()
    
    if not os.path.exists(TOKEN_FILE):
        print("=== INITIAL SCHWAB OAUTH SETUP ===")
        print(f"1. Click this URL and log in:\n{auth.generate_auth_url()}\n")
        returned_url = input("2. Paste the full redirect URL (starts with https://127.0.0.1...): ")
        
        code = auth.extract_code_from_url(returned_url)
        print("Exchanging code for permanent tokens...")
        tokens = auth.exchange_code_for_tokens(code)
        print("✅ Tokens saved to schwab_tokens.json successfully!")
    else:
        print("Checking/Verifying existing token pipeline...")
        token = auth.get_valid_token()
        print(f"Active Access Token: {token[:10]}...[TRUNCATED]")