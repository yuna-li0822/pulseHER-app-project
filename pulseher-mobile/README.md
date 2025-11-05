PulseHER Mobile (Expo)

Quick start:

1) Install Node.js (LTS) and Expo CLI:
   npm install -g expo-cli

2) From this folder run:
   npm install
   npx expo start

3) Open Expo Go on your iPhone and scan the QR code.

Notes:
- Prefer configuring the API base via Expo config (recommended) or environment variable instead of editing code:
   - In `app.json` or `app.config.js` set:
      {
         "expo": {
            "extra": {
               "EXPO_PUBLIC_API_BASE": "http://192.168.1.50:5000/api/ppg/upload-frame"
            }
         }
      }
   - Or in some setups you can set `EXPO_PUBLIC_API_BASE` in the environment / build config.
   - As a quick override you can open `AppCamera.js` and set `global.__API_BASE = 'http://192.168.1.50:5000/api/ppg/upload-frame'` in the dev console.
- Ensure your laptop and iPhone are on the same Wi-Fi network.
- For torch support in Expo Go you may need a custom dev build (EAS). See Expo docs.
