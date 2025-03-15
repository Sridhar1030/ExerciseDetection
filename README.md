# 🏋️‍♂️ Squat Counter - AI Exercise Detection App

A real-time AI-powered application that detects and counts squats using computer vision technology. Built with React and MediaPipe's pose detection model.

[![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)](https://reactjs.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-F54242?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev/)
[![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)

## ✨ Features

- **Real-time Pose Detection**: Tracks 33 body landmarks using MediaPipe's PoseLandmarker
- **Automatic Squat Counting**: Detects and counts squats with high accuracy
- **Pose Status Display**: Shows whether you're standing or squatting
- **Self-Calibration**: Adapts to your height and squat depth
- **Visual Feedback**: Overlays skeleton visualization on your webcam feed
- **Responsive Design**: Works on desktop and mobile devices

## 🖥️ Live Demo

Try the application live at: [ExerciseDetection](https://exercise-detection.vercel.app/)

## 🔧 Setup and Installation

### Prerequisites
- Node.js (v14 or higher)
- npm or yarn
- Modern web browser with camera access

### Local Development

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/squat-counter.git
   cd squat-counter
   ```

2. Install dependencies:
   ```bash
   npm install
   # or
   yarn install
   ```

3. Start the development server:
   ```bash
   npm run dev
   # or
   yarn dev
   ```

4. Open your browser and navigate to `http://localhost:5173`

## 📱 How to Use

1. Allow camera access when prompted
2. Stand in a position where your full body is visible to the camera
3. Remain still for a few seconds to let the app calibrate
4. Start performing squats - the counter will automatically increment
5. If the skeleton is not aligning properly, try the "Retry Camera Access" button

## 🧠 How It Works

The application uses MediaPipe's PoseLandmarker model to detect key body landmarks in real-time. The squat detection algorithm tracks the vertical movement of hip landmarks relative to a calibrated standing position. When your hips drop below a certain threshold and return to the standing position, a squat is counted.

The skeleton visualization is rendered on a transparent canvas overlay that matches the dimensions of your video feed, providing real-time feedback on how the AI is tracking your body.

## ⚙️ Technical Details

- **Front-end**: React with Vite
- **Styling**: Tailwind CSS
- **Pose Detection**: MediaPipe Pose Landmarker model
- **Camera Access**: Web API (getUserMedia)
- **Rendering**: HTML5 Canvas for skeleton visualization

## 🔒 Privacy

All processing happens locally in your browser. No video data is sent to any server.

## 🛠️ Future Improvements

- Additional exercise detection (push-ups, lunges, etc.)
- Rep timing and pace analysis
- Form correction feedback
- Workout history and statistics
- User profiles and progress tracking

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙌 Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for their powerful pose detection models
- [React](https://reactjs.org/) for the front-end framework
- [Tailwind CSS](https://tailwindcss.com/) for styling

---
