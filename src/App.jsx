import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { Camera } from "@mediapipe/camera_utils";
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { PoseLandmarker, FilesetResolver, DrawingUtils } from "@mediapipe/tasks-vision";
import * as tf from '@tensorflow/tfjs';

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

// Constants for the exercise classification model
const NUM_JOINTS = 33;
const T = 100; // Sequence length
const NUM_CLASSES = 5;
const CLASS_NAMES = ['TreePose', 'Barbell Biceps Curl', 'Lunges', 'Push-Up', 'Squat'];

export default function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
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

  // Add webcamRunning as a ref so it's accessible outside the useEffect
  const webcamRunningRef = useRef(false);
  
  // Add state for exercise classification
  const [predictedExercise, setPredictedExercise] = useState("No exercise detected");
  const [confidence, setConfidence] = useState(0);

  // Reference for storing the sequence of poses
  const poseSequence = useRef([]);
  
  // Reference for the TFLite model
  const tfliteModelRef = useRef(null);
  
  // Add these new state variables and refs
  const [currentExercise, setCurrentExercise] = useState(null);
  const switchCounterRef = useRef(0);
  const confidenceThreshold = 0.6; // Minimum confidence to consider switching exercises
  const switchFrames = 10; // Number of consecutive frames needed to switch exercise
  
  // Load TFLite model
  useEffect(() => {
    async function loadTFLiteModel() {
      try {
        // Load TensorFlow.js core
        await tf.ready();
        
        // Dynamically load TFLite from CDN
        const tfliteScript = document.createElement('script');
        tfliteScript.src = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.8/dist/tf-tflite.min.js';
        tfliteScript.async = true;
        
        tfliteScript.onload = async () => {
          // Now window.tflite should be available
          const tflite = window.tflite;
          
          // Register the TFLite backend
          await tf.setBackend('wasm');
          // Initialize WASM for TFLite
          await tflite.setWasmPath(
            'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.8/dist/'
          );
          
          console.log("Loading TFLite model...");
          const tfliteModel = await tflite.loadTFLiteModel('/models/stgcn_exercise.tflite');
          tfliteModelRef.current = tfliteModel;
          console.log("TFLite model loaded successfully!");
        };
        
        document.body.appendChild(tfliteScript);
      } catch (error) {
        console.error("Error loading TFLite model:", error);
      }
    }
    
    loadTFLiteModel();
    
    // Cleanup
    return () => {
      const script = document.querySelector('script[src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.8/dist/tf-tflite.min.js"]');
      if (script) {
        document.body.removeChild(script);
      }
    };
  }, []);

  // Function to compute joint angles (converted from Python)
  const computeJointAngles = (keypoints) => {
    function angleBetweenVectors(v1, v2) {
      const dotProduct = tf.sum(tf.mul(v1, v2), -1);
      const normV1 = tf.norm(v1, 'euclidean', -1);
      const normV2 = tf.norm(v2, 'euclidean', -1);
      const cosTheta = tf.clipByValue(tf.div(dotProduct, tf.add(tf.mul(normV1, normV2), 1e-8)), -1.0, 1.0);
      return tf.mul(tf.acos(cosTheta), 180.0 / Math.PI);
    }

    const coords = keypoints.slice([0, 0, 0], [T, NUM_JOINTS, 3]);
    
    const v1LeftElbow = tf.sub(coords.gather([11], 1), coords.gather([13], 1));
    const v2LeftElbow = tf.sub(coords.gather([15], 1), coords.gather([13], 1));
    const leftElbowAngle = angleBetweenVectors(v1LeftElbow, v2LeftElbow);

    const v1RightElbow = tf.sub(coords.gather([12], 1), coords.gather([14], 1));
    const v2RightElbow = tf.sub(coords.gather([16], 1), coords.gather([14], 1));
    const rightElbowAngle = angleBetweenVectors(v1RightElbow, v2RightElbow);

    const v1LeftKnee = tf.sub(coords.gather([23], 1), coords.gather([25], 1));
    const v2LeftKnee = tf.sub(coords.gather([27], 1), coords.gather([25], 1));
    const leftKneeAngle = angleBetweenVectors(v1LeftKnee, v2LeftKnee);

    const v1RightKnee = tf.sub(coords.gather([24], 1), coords.gather([26], 1));
    const v2RightKnee = tf.sub(coords.gather([28], 1), coords.gather([26], 1));
    const rightKneeAngle = angleBetweenVectors(v1RightKnee, v2RightKnee);

    const angles = tf.stack([leftElbowAngle, rightElbowAngle, leftKneeAngle, rightKneeAngle], -1);
    return angles;
  };

  // Function to preprocess keypoints (converted from Python)
  const preprocessKeypoints = (sequence) => {
    return tf.tidy(() => {
      let keypoints;
      
      if (sequence.length < T) {
        // Pad the sequence if needed
        const padding = Array(T - sequence.length).fill().map(() => 
          Array(NUM_JOINTS).fill().map(() => [0, 0, 0, 0]));
        keypoints = tf.tensor([...sequence, ...padding]);
      } else {
        // Take the last T frames
        keypoints = tf.tensor(sequence.slice(-T));
      }
      
      // Compute joint angles
      const angles = computeJointAngles(keypoints);
      
      // Expand angles to match keypoints shape
      const anglesExpanded = tf.tile(
        tf.expandDims(angles, 2),
        [1, 1, NUM_JOINTS, 1]
      );
      
      // Concatenate keypoints and angles
      const enhancedData = tf.concat([keypoints, anglesExpanded], -1);
      
      return enhancedData.expandDims(0); // Add batch dimension
    });
  };

  // Function to validate if person is standing
  const validateStanding = (landmarks) => {
    if (!landmarks || landmarks.length < 29) return true; // Default to standing if no data
    
    // Helper function to compute angle between three points
    const computeAngle = (p1, p2, p3) => {
      // Convert landmarks to vectors
      const v1 = {
        x: p1.x - p2.x,
        y: p1.y - p2.y,
        z: p1.z - p2.z
      };
      
      const v2 = {
        x: p3.x - p2.x,
        y: p3.y - p2.y,
        z: p3.z - p2.z
      };
      
      // Compute dot product
      const dotProduct = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
      
      // Compute magnitudes
      const mag1 = Math.sqrt(v1.x * v1.x + v1.y * v1.y + v1.z * v1.z);
      const mag2 = Math.sqrt(v2.x * v2.x + v2.y * v2.y + v2.z * v2.z);
      
      // Compute angle in degrees
      const cosTheta = Math.min(Math.max(dotProduct / (mag1 * mag2), -1.0), 1.0);
      const angle = Math.acos(cosTheta) * (180.0 / Math.PI);
      
      return angle;
    };
    
    // Get angles for knees and hips
    const leftKneeAngle = computeAngle(landmarks[23], landmarks[25], landmarks[27]);
    const rightKneeAngle = computeAngle(landmarks[24], landmarks[26], landmarks[28]);
    const leftHipAngle = computeAngle(landmarks[11], landmarks[23], landmarks[25]);
    const rightHipAngle = computeAngle(landmarks[12], landmarks[24], landmarks[26]);
    
    const kneeThreshold = 160;
    const hipThreshold = 160;
    
    const kneesStraight = leftKneeAngle > kneeThreshold && rightKneeAngle > kneeThreshold;
    const hipsStraight = leftHipAngle > hipThreshold && rightHipAngle > hipThreshold;
    
    return kneesStraight || hipsStraight;
  };
  
  // Function to handle exercise switching with debouncing
  const handleExerciseSwitch = (predClass, confidence, landmarks) => {
    const standing = validateStanding(landmarks);
    console.log("Predicted class:", CLASS_NAMES[predClass]);
    
    if (standing) {
      // Person is standing, don't change exercise
      console.log("Standing detected");
    } else if (currentExercise === null || CLASS_NAMES[predClass] !== currentExercise) {
      // Different exercise detected
      if (confidence > confidenceThreshold) {
        switchCounterRef.current++;
        console.log(`Switch counter: ${switchCounterRef.current}/${switchFrames}`);
        
        if (switchCounterRef.current >= switchFrames) {
          // Switch exercise after consecutive consistent detections
          setCurrentExercise(CLASS_NAMES[predClass]);
          setPredictedExercise(CLASS_NAMES[predClass]);
          console.log("============================ Switched to:", CLASS_NAMES[predClass]);
          setExerciseCount(0); // Reset rep counter for new exercise
          switchCounterRef.current = 0;
        }
      } else {
        // Low confidence, reset counter
        switchCounterRef.current = 0;
      }
    }
  };
  
  // Update the exercise classification function to use the new logic
  const runExerciseClassification = async () => {
    if (!tfliteModelRef.current || poseSequence.current.length < T) return;
    
    try {
      // Convert sequence to the format expected by the model
      const input = preprocessKeypoints(poseSequence.current);
      
      // Run inference with TFLite model
      const output = await tfliteModelRef.current.predict(input);
      const values = await output.data();
      
      const classIdx = values.indexOf(Math.max(...values));
      const conf = values[classIdx];
      
      // Get the most recent frame's landmarks for standing validation
      const recentLandmarks = poseSequence.current[poseSequence.current.length - 1];
      const landmarksForValidation = Array.from({ length: NUM_JOINTS }, (_, i) => ({
        x: recentLandmarks[i][0],
        y: recentLandmarks[i][1],
        z: recentLandmarks[i][2],
        visibility: recentLandmarks[i][3]
      }));
      
      // Apply the new exercise switching logic
      handleExerciseSwitch(classIdx, conf, landmarksForValidation);
      
      // Update confidence even if we don't immediately switch exercise
      setConfidence(conf);
      
      input.dispose();
      output.dispose();
    } catch (error) {
      console.error("Error during TFLite inference:", error);
    }
  };

  // Main effect for camera and pose detection
  useEffect(() => {
    if (!videoRef.current || !canvasRef.current) return;

    const canvasCtx = canvasRef.current.getContext('2d');
    const drawingUtils = new DrawingUtils(canvasCtx);
    
    let poseLandmarker;

    const drawResults = (results) => {
      if (!canvasRef.current || !results.landmarks || results.landmarks.length === 0) return;
      
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
      if (!webcamRunningRef.current) {
        // Simply request camera access without specifying a device
        navigator.mediaDevices.getUserMedia({ 
          video: { 
            width: { ideal: 640 },
            height: { ideal: 480 }
          } 
        })
        .then((stream) => {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => {
            videoRef.current.play();
            
            // Set canvas dimensions
            canvasRef.current.width = 640;
            canvasRef.current.height = 480;
            
            webcamRunningRef.current = true;
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
      
      // Process landmarks for exercise detection
      if (results.landmarks && results.landmarks.length > 0) {
        const landmarks = results.landmarks[0];
        
        // Create a frame of keypoints similar to the Python code
        const keypoints = Array.from(landmarks).map(lm => [lm.x, lm.y, lm.z, lm.visibility]);
        
        // Add to sequence and maintain length
        poseSequence.current.push(keypoints);
        if (poseSequence.current.length > T) {
          poseSequence.current.shift();
        }
        
        // Update landmark data for display
        const landmarkObj = {};
        landmarks.forEach((landmark, i) => {
          landmarkObj[i] = {
            x: landmark.x.toFixed(3),
            y: landmark.y.toFixed(3),
            z: landmark.z.toFixed(3),
            visibility: landmark.visibility.toFixed(3)
          };
        });
        setLandmarkData(landmarkObj);
        
        // Run exercise classification when we have enough frames
        if (poseSequence.current.length === T && tfliteModelRef.current) {
          runExerciseClassification();
        }
        
        // Process for squat detection
        const hipY = (landmarks[23].y + landmarks[24].y) / 2;
        
        if (squatState.current.isCalibrating) {
          // Calibration phase
          squatState.current.calibrationFrames++;
          squatState.current.calibrationSum += hipY;
          
          if (squatState.current.calibrationFrames >= SQUAT_SETTINGS.CALIBRATION_FRAMES) {
            squatState.current.startY = squatState.current.calibrationSum / SQUAT_SETTINGS.CALIBRATION_FRAMES;
            squatState.current.isCalibrating = false;
          }
        } else {
          // Squat detection
          const startY = squatState.current.startY;
          const deltaY = hipY - startY;
          
          if (!squatState.current.isInSquat && deltaY > SQUAT_SETTINGS.THRESHOLD) {
            squatState.current.isInSquat = true;
          } else if (
            squatState.current.isInSquat && 
            deltaY < SQUAT_SETTINGS.THRESHOLD / 2 &&
            performance.now() - squatState.current.lastSquatTime > SQUAT_SETTINGS.MIN_REP_TIME
          ) {
            squatState.current.isInSquat = false;
            squatState.current.lastSquatTime = performance.now();
            setExerciseCount(prev => prev + 1);
          }
        }
      }
      
      // Call the next frame
      requestAnimationFrame(predictWebcam);
    };

    // Initialize everything
    initializePoseLandmarker();

    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        videoRef.current.srcObject.getTracks().forEach(track => track.stop());
      }
      webcamRunningRef.current = false;
    };
  }, []);

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
        <h1 className="text-3xl font-bold mb-4">Exercise Detector 🏋️‍♂️</h1>

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
          <p className="text-3xl font-bold">
            {currentExercise === 'Squat' ? `Squats: ${exerciseCount}` : `${currentExercise || 'No exercise'}: ${exerciseCount}`}
          </p>
          <p className="text-xl">
            Status: {squatState.current.isInSquat ? "⬇️ Squatting" : "⬆️ Standing"}
          </p>
          {squatState.current.isCalibrating && (
            <p className="text-yellow-400">Calibrating... Please stand still</p>
          )}
          
          {/* Exercise classification display */}
          <div className="mt-4 p-4 bg-gray-800 rounded-lg">
            <p className="text-2xl font-bold">Detected Exercise</p>
            <p className="text-3xl font-bold text-green-400">
              {currentExercise || "No exercise detected"}
            </p>
            <p className="text-md">
              Confidence: {(confidence * 100).toFixed(1)}%
            </p>
          </div>
        </div>

        {/* Keep the retry button for convenience */}
        <button 
          className="mt-4 bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
          onClick={() => {
            if (videoRef.current && videoRef.current.srcObject) {
              videoRef.current.srcObject.getTracks().forEach(track => track.stop());
            }
            webcamRunningRef.current = false;
            enableCam();
          }}
        >
          Retry Camera Access
        </button>
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

