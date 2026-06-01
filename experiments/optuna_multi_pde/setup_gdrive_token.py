"""
One-time local setup: generate a Google OAuth2 refresh token for Drive uploads.

Steps:
  1. Go to https://console.cloud.google.com/  (free, any Google account)
  2. Create a project → APIs & Services → Enable "Google Drive API"
  3. APIs & Services → Credentials → Create Credentials → OAuth client ID
       Application type: Desktop app   Name: anything
  4. Download the JSON → paste client_id and client_secret below
  5. Run:  python setup_gdrive_token.py
  6. A browser opens → sign in → allow Drive access
  7. Copy the printed GDRIVE_TOKEN JSON → add as Kaggle Secret named GDRIVE_TOKEN

The token never expires as long as it is used at least once every 6 months.
"""

import json

# ── Fill these in from your GCP OAuth2 Desktop app credentials ────────────────
CLIENT_ID     = "PASTE_YOUR_CLIENT_ID_HERE"
CLIENT_SECRET = "PASTE_YOUR_CLIENT_SECRET_HERE"
# ─────────────────────────────────────────────────────────────────────────────

SCOPES    = ["https://www.googleapis.com/auth/drive"]
TOKEN_URI = "https://oauth2.googleapis.com/token"

def main():
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        print("Install first:  pip install google-auth-oauthlib")
        raise

    client_config = {
        "installed": {
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": TOKEN_URI,
            "redirect_uris": ["http://localhost"],
        }
    }

    flow = InstalledAppFlow.from_client_config(client_config, scopes=SCOPES)
    creds = flow.run_local_server(port=0, prompt="consent",
                                  access_type="offline")

    token_dict = {
        "client_id":     CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "refresh_token": creds.refresh_token,
        "token_uri":     TOKEN_URI,
        "scopes":        SCOPES,
    }

    print("\n" + "=" * 70)
    print("SUCCESS!  Add the JSON below as a Kaggle Secret named  GDRIVE_TOKEN")
    print("=" * 70)
    print(json.dumps(token_dict, indent=2))
    print("=" * 70)
    print("\nKaggle → Settings → Secrets → New Secret")
    print("  Name : GDRIVE_TOKEN")
    print("  Value: (paste the JSON above)")

if __name__ == "__main__":
    main()
