# Getting Google Cloud Vision API Credentials

Follow these steps to generate a service account key for Google Cloud Vision API:

1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project or select an existing one.
3. Enable the **Vision API**:
   - In the sidebar, go to **APIs & Services > Library**.
   - Search for **Vision API** and click **Enable**.
4. Create service account credentials:
   - Go to **APIs & Services > Credentials**.
   - Click **Create Credentials > Service Account**.
5. Enter a service account name, then assign a role (e.g., **Editor** or a custom role with Vision API access).
6. After creating the service account, open it and go to the **Keys** tab.
7. Click **Add Key > Create New Key** → select **JSON** → **Create**.
8. A `.json` key file will be downloaded to your computer. This is your **service account key**.

⚠️ **Important:**  
- Keep this file secure and **never commit it to GitHub**.  
- You will need to set the environment variable so your code can authenticate:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"
