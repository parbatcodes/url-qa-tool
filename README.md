# URL

# Getting Reddit API Credentials

Follow these steps to generate Reddit API credentials for your project:

1. Go to [Reddit App Preferences](https://www.reddit.com/prefs/apps) (log in if required).
2. Scroll down to **Developed Applications** and click **Create App** or **Create Another App**.
3. Fill in the application details:
   - **Name**: Choose any name (e.g., `MyScraper`).
   - **Type**: Select **script** (for personal use).
   - **Redirect URI**: Use `http://localhost:8080` (or any valid URL).
   - **Description**: Optional, can be left blank.
4. Click **Create app**.
5. Once created, you will see:
   - **Client ID** → the 14-character string under your app name.
   - **Client Secret** → displayed right below it.
6. Keep these values safe. You will also need:
   - Your **Reddit username**
   - Your **Reddit password**
7. Use these credentials in your code for authentication.

---
✅ Example environment variables setup:

```env
REDDIT_CLIENT_ID=your_client_id_here
REDDIT_CLIENT_SECRET=your_client_secret_here
REDDIT_USERNAME=your_username
REDDIT_PASSWORD=your_password
REDDIT_USER_AGENT=your_app_name
