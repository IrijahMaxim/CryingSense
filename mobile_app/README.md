# CryingSense Mobile Application

React Native mobile application for the CryingSense infant cry interpretation system.

## Features

- 📱 Real-time cry detection notifications
- 📊 View cry classification history
- 📈 Detailed event information with probabilities
- ⚙️ Configurable notification settings
- 🔄 Auto-refresh for live updates

## Tech Stack

- **React Native** (v0.73) - Mobile framework
- **Expo** (v50) - Development and build toolchain
- **React Navigation** - Screen navigation
- **Axios** - API communication
- **Expo Notifications** - Push notifications
- **Day.js** - Date/time formatting

## Prerequisites

- Node.js (v20 or higher)
- npm or yarn
- Expo CLI (`npm install -g expo-cli`)
- iOS Simulator (Mac only) or Android Emulator
- Physical device with Expo Go app (optional)

## Installation

1. Navigate to the mobile app directory:
```bash
cd mobile_app/CryingSenseApp
```

2. Install dependencies:
```bash
npm install
# or
yarn install
```

## Development

### Start the development server:
```bash
npm start
# or
expo start
```

### Run on specific platform:
```bash
# iOS (Mac only)
npm run ios

# Android
npm run android

# Web
npm run web
```

### Using Expo Go (Physical Device)

1. Install Expo Go app on your phone:
   - iOS: Download from App Store
   - Android: Download from Play Store

2. Start the development server:
```bash
npm start
```

3. Scan the QR code with:
   - iOS: Camera app
   - Android: Expo Go app

## Configuration

### Backend Server URL

By default, the app connects to `http://localhost:3000`.

To change the backend URL:
1. Open the app
2. Navigate to Settings
3. Update the "Backend URL" field
4. Example: `http://192.168.1.100:3000` (use your server's IP)

### Notification Settings

Configure notifications in the Settings screen:
- Enable/disable notifications
- Toggle sound alerts
- Toggle vibration

## Project Structure

```
CryingSenseApp/
├── App.js                    # Main app entry point
├── app.json                  # Expo configuration
├── package.json              # Dependencies
│
├── screens/                  # Screen components
│   ├── HomeScreen.js         # Main dashboard
│   ├── HistoryScreen.js      # Event history list
│   ├── DetailsScreen.js      # Event detail view
│   └── SettingsScreen.js     # App settings
│
├── components/               # Reusable components
│   └── CryClassificationCard.js  # Event card component
│
├── services/                 # API and services
│   ├── api.js                # Backend API calls
│   └── notificationService.js # Notification handling
│
├── navigation/               # Navigation configuration
│   └── (future navigation files)
│
└── assets/                   # Images and resources
    └── (icons, images, etc.)
```

## Screens

### Home Screen
- Displays the latest cry detection
- Shows recent events
- Pull-to-refresh for updates
- Auto-refresh every 5 seconds

### History Screen
- Full list of all cry events
- Sorted by most recent first
- Tap any event to view details

### Details Screen
- Complete event information
- Confidence percentage
- All class probabilities (bar chart)
- Device information
- Processing time

### Settings Screen
- Notification preferences
- Backend server configuration
- App version information

## API Integration

The app communicates with the CryingSense backend via REST API:

### Endpoints Used:
- `GET /api/classifications` - Fetch recent classifications
- `GET /api/classifications/:id` - Get specific event details
- `GET /api/classifications/stats` - Get statistics

### Mock Data Mode

If the backend is unreachable, the app automatically falls back to mock data for development/testing purposes.

## Notifications

### Local Notifications
The app sends local notifications when new cry events are detected:
- Different emojis for each cry type
- Confidence percentage in notification body
- Tap notification to view details

### Notification Types:
- 🍼 Hunger
- 😴 Tired
- 😣 Discomfort
- 😖 Belly Pain
- 💨 Burp

## Building for Production

### Android APK:
```bash
expo build:android
```

### iOS IPA:
```bash
expo build:ios
```

### Standalone Apps with EAS Build:
```bash
npm install -g eas-cli
eas build --platform android
eas build --platform ios
```

## Testing

Run on a real device or emulator to test:
- API connectivity
- Notifications
- Real-time updates
- User interface

## Troubleshooting

### Cannot connect to backend
- Verify backend server is running
- Check backend URL in Settings
- Ensure devices are on same network
- Use IP address instead of localhost

### Notifications not working
- Grant notification permissions in device settings
- Check notification settings in app
- Restart the app

### App crashes on startup
- Clear cache: `expo start -c`
- Reinstall dependencies: `rm -rf node_modules && npm install`
- Update Expo: `expo upgrade`

## Contributing

1. Make changes in the appropriate files
2. Test on both iOS and Android if possible
3. Commit with clear messages
4. Submit pull request

## License

MIT

## Support

For issues or questions, please refer to the main project documentation or create an issue in the repository.
