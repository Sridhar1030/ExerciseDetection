import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { Camera } from "@mediapipe/camera_utils";
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { PoseLandmarker, FilesetResolver, DrawingUtils } from "@mediapipe/tasks-vision";

// Add this constant at the top of your file, outside the component
const POSE_LANDMARKS = {
  0: 'nose',
  1: 'left_eye_inner',
  2: 'left_eye',
  3: 'left_eye_outer',
  4: 'right_eye_inner',
  5: 'right_eye',
  6: 'right_eye_outer',
  7: 'left_ear',
  8: 'right_ear',
  9: 'mouth_left',
  10: 'mouth_right',
  11: 'left_shoulder',
  12: 'right_shoulder',
  13: 'left_elbow',
  14: 'right_elbow',
  15: 'left_wrist',
  16: 'right_wrist',
  17: 'left_pinky',
  18: 'right_pinky',
  19: 'left_index',
  20: 'right_index',
  21: 'left_thumb',
  22: 'right_thumb',
  23: 'left_hip',
  24: 'right_hip',
  25: 'left_knee',
  26: 'right_knee',
  27: 'left_ankle',
  28: 'right_ankle',
  29: 'left_heel',
  30: 'right_heel',
  31: 'left_foot_index',
  32: 'right_foot_index'
};

export default function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [devices, setDevices] = useState([]);
  const [selectedDevice, setSelectedDevice] = useState('');
  const [exerciseCount, setExerciseCount] = useState(0);
  const [detectionStatus, setDetectionStatus] = useState({
    face: false,
    pose: false,
    leftHand: false,
    rightHand: false
  });

  // Squat detection state using useRef to persist between renders
  const squatState = useRef({
    startY: null,
    isInSquat: false,
    lastSquatTime: 0,
    calibrationFrames: 0,
    calibrationSum: 0,
    isCalibrating: true
  });

  // Constants for squat detection
  const SQUAT_SETTINGS = {
    THRESHOLD: 0.15,
    MIN_REP_TIME: 1000,
    CALIBRATION_FRAMES: 30
  };

  // Add new state for landmark data
  const [landmarkData, setLandmarkData] = useState({});

  // Add state for current pose
  const [currentPose, setCurrentPose] = useState("No pose detected");

  // Add a new state for error messages
  const [errorMessage, setErrorMessage] = useState('');

  // Improve the camera detection code
  useEffect(() => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
      setErrorMessage('Camera API not supported in your browser or requires HTTPS');
      return;
    }

    // Clear any previous errors
    setErrorMessage('');
    
    navigator.mediaDevices.enumerateDevices()
      .then(devices => {
        const videoDevices = devices.filter(device => device.kind === 'videoinput');
        setDevices(videoDevices);
        
        if (videoDevices.length === 0) {
          setErrorMessage('No camera detected. Please connect a camera and refresh.');
        } else {
          setSelectedDevice(videoDevices[0].deviceId);
          console.log('Cameras detected:', videoDevices.length);
        }
      })
      .catch(err => console.error('Error getting devices:', err));
  }, []);

  // Main effect for camera and pose detection
  useEffect(() => {
    if (!videoRef.current || !canvasRef.current || !selectedDevice) return;

    const canvasCtx = canvasRef.current.getContext('2d');
    const drawingUtils = new DrawingUtils(canvasCtx);
    
    let poseLandmarker;
    let webcamRunning = false;

    const drawResults = (results) => {
      if (!canvasRef.current || !results.landmarks || results.landmarks.length === 0) return;
      
      const canvasCtx = canvasRef.current.getContext('2d');
      
      // Clear with transparent background
      canvasCtx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      
      // Use default drawingUtils
      for (const landmarks of results.landmarks) {
        drawingUtils.drawConnectors(
          landmarks,
          PoseLandmarker.POSE_CONNECTIONS
        );
        
        drawingUtils.drawLandmarks(
          landmarks
        );
      }
    };

    const initializePoseLandmarker = async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );

        poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numPoses: 1,
          minPoseDetectionConfidence: 0.5,
          minPosePresenceConfidence: 0.5,
          minTrackingConfidence: 0.5
        });

        enableCam();
      } catch (error) {
        console.error("Error initializing pose landmarker:", error);
      }
    };

    const enableCam = () => {
      if (!webcamRunning) {
        navigator.mediaDevices.getUserMedia({ 
          video: { 
            deviceId: { exact: selectedDevice },
            width: { ideal: 640 },
            height: { ideal: 480 }
          } 
        })
        .then((stream) => {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => {
            videoRef.current.play();
            
            // Simply match the canvas to the container dimensions
            canvasRef.current.width = 640;
            canvasRef.current.height = 480;
            
            webcamRunning = true;
            predictWebcam();
          };
        })
        .catch((err) => {
          console.error("Error accessing webcam:", err);
        });
      }
    };

    const predictWebcam = async () => {
      if (!videoRef.current || !poseLandmarker) {
        requestAnimationFrame(predictWebcam);
        return;
      }

      const startTimeMs = performance.now();
      
      // Process the video frame
      const results = poseLandmarker.detectForVideo(videoRef.current, startTimeMs);
      
      // Draw results
      drawResults(results);
      
      // Call the next frame
      requestAnimationFrame(predictWebcam);
    };

    // Initialize everything
    initializePoseLandmarker();

    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        videoRef.current.srcObject.getTracks().forEach(track => track.stop());
      }
      webcamRunning = false;
    };
  }, [selectedDevice]);

  // Function to group landmarks by body part
  const groupedLandmarks = {
    face: Array.from({ length: 11 }, (_, i) => i),
    torso: [11, 12, 23, 24],
    leftArm: [11, 13, 15, 17, 19, 21],
    rightArm: [12, 14, 16, 18, 20, 22],
    leftLeg: [23, 25, 27, 29, 31],
    rightLeg: [24, 26, 28, 30, 32]
  };

  // Add specific thresholds for different body parts in the data panel
  const groupedLandmarksThresholds = {
    face: 0.85,    // Very strict for face landmarks
    torso: 0.7,    // Fairly strict for torso
    leftArm: 0.6,  // Moderate threshold for arms
    rightArm: 0.6,
    leftLeg: 0.6,  // Moderate threshold for legs
    rightLeg: 0.6
  };

  return (
    <div className="flex min-h-screen bg-gray-900 text-white p-4">
      <div className="flex flex-col items-center">
        <h1 className="text-3xl font-bold mb-4">Squat Counter 🏋️‍♂️</h1>

        <select
          className="mb-4 p-2 rounded bg-gray-700 text-white"
          value={selectedDevice}
          onChange={(e) => setSelectedDevice(e.target.value)}
        >
          {devices.map(device => (
            <option key={device.deviceId} value={device.deviceId}>
              {device.label || `Camera ${devices.indexOf(device) + 1}`}
            </option>
          ))}
        </select>

        {/* Fixed size container */}
        <div 
          style={{ 
            position: 'relative',
            width: '640px', 
            height: '480px',
            overflow: 'hidden',
            border: '2px solid #333'
          }}
        >
          {/* Video with transform to un-mirror */}
          <video
            ref={videoRef}
            style={{ 
              position: 'absolute',
              width: '100%', 
              height: '100%',
              objectFit: 'contain',
              backgroundColor: '#000',
              transform: 'scaleX(-1)'
            }}
            muted
            playsInline
          />
          
          {/* Canvas with matching transform */}
          <canvas
            ref={canvasRef}
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: '100%',
              zIndex: 999,
              transform: 'scaleX(-1)'
            }}
          />
        </div>
        
        <div className="mt-4 text-center">
          <p className="text-3xl font-bold">Squats: {exerciseCount}</p>
          <p className="text-xl">
            Status: {squatState.current.isInSquat ? "⬇️ Squatting" : "⬆️ Standing"}
          </p>
          {squatState.current.isCalibrating && (
            <p className="text-yellow-400">Calibrating... Please stand still</p>
          )}
          <p className="text-3xl font-bold">Current Pose: {currentPose}</p>
        </div>
      </div>

      {/* Right side - Data Panel */}
      <div className="ml-4 w-96 overflow-y-auto h-screen">
        <div className="sticky top-0 bg-gray-900 p-2 z-10">
          <h2 className="text-xl font-bold mb-2">Pose Data</h2>
          <p className="text-sm text-gray-400">
            Points Detected: {detectionStatus.pose ? "33/33 ✅" : "0/33 ❌"}
          </p>
          <button
            className="mt-2 bg-blue-500 hover:bg-blue-700 text-white font-bold py-1 px-3 rounded text-sm"
            onClick={() => {
              const dataStr = JSON.stringify(landmarkData, null, 2);
              const blob = new Blob([dataStr], { type: 'application/json' });
              const url = URL.createObjectURL(blob);
              const a = document.createElement('a');
              a.href = url;
              a.download = `pose-data-${new Date().toISOString()}.json`;
              a.click();
            }}
          >
            Export Data
          </button>
        </div>

        <div className="space-y-4 mt-4">
          {Object.entries(groupedLandmarks).map(([group, points]) => {
            // Use specific threshold for each body part
            const threshold = groupedLandmarksThresholds[group] || 0.7;

            // Filter points using the group-specific threshold
            const visiblePoints = points.filter(index =>
              landmarkData[index] &&
              parseFloat(landmarkData[index].visibility) > threshold
            );

            // Only show group if it has visible points
            if (visiblePoints.length === 0) return null;

            return (
              <div key={group} className="bg-gray-800 p-3 rounded-lg">
                <h3 className="text-md font-bold mb-2 capitalize">{group}</h3>
                <div className="space-y-2">
                  {visiblePoints.map(index => (
                    <div key={index} className="text-xs">
                      <div className="border-l-2 border-gray-600 pl-2">
                        <div className="font-bold text-gray-300">
                          {POSE_LANDMARKS[index]}
                        </div>
                        <div className="grid grid-cols-2 gap-x-2">
                          <span>x: {landmarkData[index].x}</span>
                          <span>y: {landmarkData[index].y}</span>
                          <span>z: {landmarkData[index].z}</span>
                          <span className="text-green-400">
                            vis: {landmarkData[index].visibility}
                          </span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

