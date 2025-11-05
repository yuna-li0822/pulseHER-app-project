<<<<<<< HEAD
# 💖 PulseHER - AI-Powered Heart Health Monitoring for Women

An intelligent heart health monitoring application that combines React frontend with Python ML backend for personalized cardiovascular insights.

## 📱 Features

- **Real-time Heart Monitoring**: Track BPM, blood pressure, stress levels
- **AI-Powered Analysis**: Machine learning insights and recommendations
- **3D Heart Visualizer**: Interactive anatomical models
- **Cross-Platform**: Works on iOS, Android, and Web
- **Cloud Database**: Firebase integration for data persistence

## 🏗️ Clean Project Structure (Reorganized)

```
PulseHER/
│
├── backend/                  ← Advanced Python Flask API & PPG Integration
│   ├── app.py               ← Main Flask server with ML integration
│   ├── ppg_api.py           ← PPG endpoints for heart rate monitoring
│   ├── ppg_processor.py     ← Camera-based signal processing
│   ├── requirements.txt     ← Python dependencies
│   ├── model/
│   │   ├── train_model.py   ← ML model training (99% accuracy)
│   │   ├── model.pkl        ← Trained RandomForest model
│   │   └── scaler.pkl       ← Data preprocessing scaler
│   └── test_*.py           ← Testing utilities
│
├── frontend/                ← Modern React Web Application  
│   ├── src/
│   │   ├── App.js          ← Main React router with PulseHer branding
│   │   ├── App.css         ← Modern CSS with women-focused design
│   │   └── components/     ← Reusable React components
│   │       ├── PPGMonitor.js ← Camera-based heart rate monitoring
│   │       └── PPGMonitor.css ← PPG component styling
│   ├── package.json        ← Node.js dependencies
│   └── package-lock.json   ← Dependency lock file
│
├── database/               ← Firebase Integration
│   ├── database.py        ← Firebase utilities
│   └── firebase.json      ← Firebase configuration
│
├── database/               ← Database configuration
│   ├── firebase.json       ← Firebase credentials
│   └── database.py         ← Database helper functions
│
└── README.md              ← This file
```

## 🚀 Quick Start

### 1. Backend Setup (Python)

```bash
cd backend/
pip install -r requirements.txt
python model/train_model.py  # Train the ML model
python app.py               # Start Flask server (localhost:5000)
```

### 2. Frontend Setup (React Native)

```bash
cd frontend/
npm install
npm run web    # Start web version (localhost:19006)
# or
npm start      # Start Expo dev server for mobile
```

### 3. Database Setup (Firebase)

1. Create a Firebase project at [firebase.google.com](https://firebase.google.com)
2. Download your service account key
3. Replace `database/firebase.json` with your credentials
4. Enable Firestore database in your Firebase console

## 🎯 Tech Stack

### Frontend
- **React Native** - Cross-platform mobile development
- **Expo** - Development platform and deployment
- **React Navigation** - Screen navigation
- **React Native Web** - Web compatibility

### Backend
- **Flask** - Python web framework
- **scikit-learn** - Machine learning models
- **NumPy/Pandas** - Data processing
- **Firebase Admin** - Database integration

### Database
- **Firebase Firestore** - NoSQL cloud database
- **Real-time sync** - Automatic data synchronization

## 📊 ML Model Features

The AI model analyzes multiple health factors:
- Age and demographic data
- Resting blood pressure
- Maximum heart rate
- Cholesterol levels
- Exercise habits
- Stress indicators
- Sleep patterns

**Output**: Risk assessment (Low/Medium/High) with confidence scores

## 🔗 API Endpoints

### Heart Data
- `GET /api/heart-data` - Retrieve user's heart data
- `POST /api/heart-data` - Add new heart measurements

### AI Analysis
- `POST /api/ai-analysis` - Get AI insights for metrics
- `POST /api/predict` - ML model risk prediction

## 📱 App Screens

1. **Home**: Overview dashboard and quick actions
2. **Stats**: Detailed heart metrics and trends
3. **AI Assistant**: Interactive health advice
4. **3D Visualizer**: Anatomical heart models
5. **Resources**: Educational content and links

## 🛠️ Development

### Adding New Features
1. Frontend components go in `frontend/src/components/`
2. Backend endpoints in `backend/app.py`
3. ML models in `backend/model/`

### Environment Variables
Create `.env` files for sensitive configuration:
- Firebase credentials
- API keys
- Database URLs

## 🚀 Deployment

### Frontend (Web)
```bash
cd frontend/
npm run build
# Deploy to Netlify, Vercel, or similar
```

### Backend (API)
```bash
# Deploy to Heroku, Railway, or similar
# Make sure to set environment variables
```

### Mobile App
```bash
cd frontend/
expo build:android  # Android APK
expo build:ios      # iOS app
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🆘 Support

For issues and questions:
- Create an issue on GitHub
- Check the documentation
- Review the API endpoints

---

**PulseHer** - Empowering women's heart health through AI 💖�
=======
# pulseHER-app-project
PulseHER is a women-centered mobile app that bridges the gender gap in cardiovascular care by integrating real-time PPG pulse analysis with female-specific physiological insights.
>>>>>>> 88bfe319c020014247cc0c6278c639e0b3436de3
