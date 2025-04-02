import { useEffect, useRef, useState } from "react";
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
const T = 100; // Sequence length (50 frames)
const NUM_CLASSES = 4;
const CLASS_NAMES = ['TreePose', 'Lunges', 'Push-Up', 'Squat'];

// Update the MODEL_INPUT_SHAPE constant to match requirements
const MODEL_INPUT_SHAPE = [1, 50, 33, 8]; // [batch, frames, joints, features]

// Define a custom GraphConv layer to match the one in the model
class GraphConv extends tf.layers.Layer {
  constructor(config) {
    super(config);
    this.units = config.units || 32;
    this.activation = config.activation;
    this.use_bias = config.use_bias !== undefined ? config.use_bias : true;
  }

  build(inputShape) {
    // Create weights with shapes that match what's in the model
    this.kernel = this.addWeight(
      'kernel',
      [inputShape[inputShape.length - 1], this.units],
      'float32',
      tf.initializers.glorotUniform()
    );
    
    if (this.use_bias) {
      this.bias = this.addWeight(
        'bias',
        [this.units],
        'float32',
        tf.initializers.zeros()
      );
    }
    
    this.built = true;
  }

  call(inputs) {
    // This is a simplified implementation that approximates GraphConv
    // Real GraphConv would use the graph structure, but we approximate with regular conv
    let output = tf.matMul(inputs, this.kernel);
    
    if (this.use_bias) {
      output = tf.add(output, this.bias);
    }
    
    if (this.activation) {
      output = tf.layers.activation({activation: this.activation}).apply(output);
    }
    
    return output;
  }

  computeOutputShape(inputShape) {
    return [inputShape[0], inputShape[1], this.units];
  }

  getConfig() {
    const config = super.getConfig();
    Object.assign(config, {
      units: this.units,
      activation: this.activation,
      use_bias: this.use_bias
    });
    return config;
  }

  static get className() {
    return 'GraphConv';
  }
}

// Register the custom layer
tf.serialization.registerClass(GraphConv);

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
  
  // Add this state for debugging
  const [modelDebugInfo, setModelDebugInfo] = useState({ 
    inputShape: "Unknown", 
    lastPrediction: "None",
    inferenceTime: 0,
    modelLoaded: false,
    processedFrames: 0
  });

  // Add more detailed loading state
  const [modelLoadingState, setModelLoadingState] = useState({
    status: 'initializing', // initializing, loading, success, error
    message: 'Starting TensorFlow.js...',
    error: null,
    attempts: 0
  });

  // Now update the loadModel function
  const loadModel = async () => {
    const modelPath = '/public/models/stgcn_exercise_fine_tunned.tflite';
    
    try {
      setModelLoadingState(prev => ({
        ...prev,
        status: 'loading',
        message: 'Loading TensorFlow.js...'
      }));
      
      await tf.ready();
      console.log("TensorFlow.js ready:", tf.getBackend());
      
      if (typeof window.tflite === 'undefined') {
        console.log("TFLite not found, loading from CDN...");
        const script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@latest/dist/tf-tflite.min.js';
        script.async = true;
        const scriptLoadPromise = new Promise((resolve, reject) => {
          script.onload = resolve;
          script.onerror = () => reject(new Error("Failed to load TFLite script from CDN"));
        });
        document.head.appendChild(script);
        await scriptLoadPromise;
        console.log("TFLite script loaded from CDN");
      }
      
      if (window.tflite && window.tflite.setWasmPath) {
        window.tflite.setWasmPath(
          "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@latest/dist/"
        );
        console.log("TFLite WASM path set");
      } else {
        throw new Error("TFLite module not properly loaded");
      }
      
      setModelLoadingState(prev => ({
        ...prev,
        message: 'Loading TFLite model...'
      }));
      
      console.log("Loading TFLite model from:", modelPath);
      const model = await window.tflite.loadTFLiteModel(modelPath);
      console.log("✅ TFLite model loaded successfully!");
      
      if (!model) {
        throw new Error("Model is undefined after loading");
      }
      
      tfliteModelRef.current = model;
      
      setModelLoadingState({
        status: 'success',
        message: 'TFLite model loaded successfully!'
      });
      
      setModelDebugInfo({
        modelLoaded: true,
        modelType: "TFLite STGCN Model"
      });
      
      // Run exercise classification after model is loaded
      runExerciseClassification();
      
      return true;
    } catch (error) {
      console.error("Error in model loading process:", error);
      setModelLoadingState({
        status: 'error',
        message: `Error loading model: ${error.message}`,
        error: error.toString()
      });
      return false;
    }
  };

  // Update the function to correctly compute joint angles based on the requirements
  const computeJointAngles = (frame) => {
    try {
      if (!frame || frame.length < NUM_JOINTS) {
        return [0, 0, 0, 0]; // Default angles if frame is invalid
      }
      
      // Extract joint positions for angle calculations
      const leftShoulder = frame[11]; // POSE_LANDMARKS index 11 is left_shoulder
      const leftElbow = frame[13];    // POSE_LANDMARKS index 13 is left_elbow
      const leftWrist = frame[15];    // POSE_LANDMARKS index 15 is left_wrist
      
      const rightShoulder = frame[12]; // POSE_LANDMARKS index 12 is right_shoulder
      const rightElbow = frame[14];    // POSE_LANDMARKS index 14 is right_elbow
      const rightWrist = frame[16];    // POSE_LANDMARKS index 16 is right_wrist
      
      const leftHip = frame[23];      // POSE_LANDMARKS index 23 is left_hip
      const leftKnee = frame[25];     // POSE_LANDMARKS index 25 is left_knee
      const leftAnkle = frame[27];    // POSE_LANDMARKS index 27 is left_ankle
      
      const rightHip = frame[24];     // POSE_LANDMARKS index 24 is right_hip
      const rightKnee = frame[26];    // POSE_LANDMARKS index 26 is right_knee
      const rightAnkle = frame[28];   // POSE_LANDMARKS index 28 is right_ankle
      
      // Calculate angle between three points (in 3D)
      const calculateAngle = (a, b, c) => {
        if (!a || !b || !c || a[3] < 0.5 || b[3] < 0.5 || c[3] < 0.5) {
          return 0; // Return 0 for low confidence points
        }
        
        // Create vectors from points (using x, y, z coordinates)
        const vector1 = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        const vector2 = [c[0] - b[0], c[1] - b[1], c[2] - b[2]];
        
        // Calculate dot product
        const dotProduct = vector1[0] * vector2[0] + vector1[1] * vector2[1] + vector1[2] * vector2[2];
        
        // Calculate magnitudes
        const magnitude1 = Math.sqrt(vector1[0]**2 + vector1[1]**2 + vector1[2]**2);
        const magnitude2 = Math.sqrt(vector2[0]**2 + vector2[1]**2 + vector2[2]**2);
        
        // Calculate angle in radians and convert to degrees
        if (magnitude1 * magnitude2 === 0) return 0;
        
        const cos = Math.max(-1, Math.min(1, dotProduct / (magnitude1 * magnitude2)));
        return Math.round(Math.acos(cos) * (180 / Math.PI));
      };
      
      // Calculate the four required angles
      const leftElbowAngle = calculateAngle(leftShoulder, leftElbow, leftWrist);
      const rightElbowAngle = calculateAngle(rightShoulder, rightElbow, rightWrist);
      const leftKneeAngle = calculateAngle(leftHip, leftKnee, leftAnkle);
      const rightKneeAngle = calculateAngle(rightHip, rightKnee, rightAnkle);
      
      return [leftElbowAngle, rightElbowAngle, leftKneeAngle, rightKneeAngle];
    } catch (error) {
      console.error("Error computing joint angles:", error);
      return [0, 0, 0, 0];
    }
  };

  // Add a new function to prepare input data according to the requirements
  const prepareModelInput = (poseSequence) => {
    try {
      // Check if we have enough data
      if (!poseSequence || poseSequence.length === 0) {
        console.log("No pose sequence data available");
        return null;
      }
      
      // Take last 50 frames or pad if needed
      let sequence = [...poseSequence];
      if (sequence.length > T) {
        sequence = sequence.slice(-T); // Take last T frames
      } else if (sequence.length < T) {
        // Pad with zeros
        const padding = Array(T - sequence.length).fill().map(() => 
          Array(NUM_JOINTS).fill().map(() => [0, 0, 0, 0])
        );
        sequence = [...sequence, ...padding];
      }
      
      console.log(`Prepared sequence with ${sequence.length} frames`);
      
      // Now format the input according to requirements (1, 50, 33, 8)
      return tf.tidy(() => {
        const formattedFrames = [];
        
        // Process each frame
        for (let frameIdx = 0; frameIdx < T; frameIdx++) {
          const frame = sequence[frameIdx];
          
          // Compute angles for this frame
          const angles = computeJointAngles(frame);
          
          // Format each joint in the frame
          const formattedJoints = [];
          
          for (let jointIdx = 0; jointIdx < NUM_JOINTS; jointIdx++) {
            const joint = frame[jointIdx] || [0, 0, 0, 0];
            
            // Combine keypoint data with angles (tiled across all joints)
            const features = [
              joint[0], joint[1], joint[2], joint[3],  // x, y, z, visibility
              angles[0], angles[1], angles[2], angles[3]  // left elbow, right elbow, left knee, right knee
            ];
            
            formattedJoints.push(features);
          }
          
          formattedFrames.push(formattedJoints);
        }
        
        // Create tensor with correct shape [1, 50, 33, 8]
        return tf.tensor(formattedFrames).expandDims(0);
      });
    } catch (error) {
      console.error("Error preparing model input:", error);
      return null;
    }
  };

  // Function to run exercise classification
  const runExerciseClassification = async () => {
    // Check if model is loaded
    if (!tfliteModelRef.current) {
      console.error("Model reference is null");
      setModelDebugInfo(prev => ({ ...prev, lastPrediction: "Model not loaded" }));
      return;
    }
    
    // Prepare the input according to model requirements
    const inputTensor = tf.tensor([Array(50).fill(Array(33).fill([0.5, 0.6, 0.1, 0.95, 90.0, 85.0, 120.0, 115.0]))]);
    
    try {
      console.log(`Input tensor shape: ${inputTensor.shape}`);
      
      // Check if the model has a predict method
      if (typeof tfliteModelRef.current.predict !== 'function') {
        throw new Error("Model does not have a predict method");
      }
      
      // Run inference
      console.log("Running inference...");
      const output = tfliteModelRef.current.predict(inputTensor);
      
      if (!output) {
        throw new Error("Output is undefined after prediction");
      }
      
      const outputData = await output.data();
      console.log("Raw prediction:", outputData);
      
      // Find class with highest probability
      const maxIdx = outputData.indexOf(Math.max(...outputData));
      const confidence = outputData[maxIdx];
      const predictedClass = CLASS_NAMES[maxIdx];
      
      console.log(`Prediction: ${predictedClass} (${maxIdx}) with confidence ${confidence.toFixed(4)}`);
      
      // Update state with prediction
      setPredictedExercise(predictedClass);
      setConfidence(confidence);
      
      // Clean up
      inputTensor.dispose();
      output.dispose();
    } catch (error) {
      console.error("Error during inference:", error);
      setModelDebugInfo(prev => ({
        ...prev,
        lastPrediction: `Error: ${error.message}`
      }));
    }
  };

  // Simplify the preprocessKeypoints function to make it more robust
  const preprocessKeypoints = (sequence) => {
    return tf.tidy(() => {
      try {
        console.log(`Sequence length: ${sequence.length}`);
        let keypoints;
        
        if (sequence.length < T) {
          // Pad the sequence with zeros
          const padding = Array(T - sequence.length).fill().map(() => 
            Array(NUM_JOINTS).fill().map(() => [0, 0, 0, 0]));
          keypoints = tf.tensor([...sequence, ...padding]);
        } else {
          // Take the last T frames
          keypoints = tf.tensor(sequence.slice(-T));
        }
        
        // Ensure keypoints have the right shape
        if (!keypoints || !keypoints.shape || keypoints.shape.length !== 3) {
          console.error("Invalid keypoints tensor shape:", keypoints?.shape);
          return tf.zeros([1, T, NUM_JOINTS, 4]); // Return dummy data
        }
        
        console.log(`keypoints shape: ${keypoints.shape}`);
        
        // For simplicity, just use the keypoints without angles
        // This ensures the model gets valid input even if the angle calculation fails
        return keypoints.expandDims(0); // Add batch dimension
      } catch (error) {
        console.error("Error in preprocessKeypoints:", error);
        return tf.zeros([1, T, NUM_JOINTS, 4]); // Return dummy data
      }
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
    
    // Get angles for knees and hips - exact same calculations as C# code
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
      // Not resetting counter as in the C# code
    } 
    else if (currentExercise === null || CLASS_NAMES[predClass] !== currentExercise) {
      // Different exercise detected - matches C# logic
      if (confidence > confidenceThreshold) {
        switchCounterRef.current++;
        console.log(`Switch counter: ${switchCounterRef.current}/${switchFrames}`);
        
        if (switchCounterRef.current >= switchFrames) {
          // Switch exercise after consecutive consistent detections
          let newExercise = CLASS_NAMES[predClass];
          if (!newExercise) newExercise = "Detecting..."; // Match the C# null check
          
          setCurrentExercise(newExercise);
          setPredictedExercise(newExercise);
          console.log("============================ " + newExercise);
          setExerciseCount(0); // Reset rep counter for new exercise
          switchCounterRef.current = 0;
        }
      } else {
        // Low confidence, reset counter
        switchCounterRef.current = 0;
      }
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
        // Request camera access with explicit dimensions
        navigator.mediaDevices.getUserMedia({ 
          video: { 
            width: { ideal: 640, min: 320 },
            height: { ideal: 480, min: 240 },
            frameRate: { ideal: 30 }
          } 
        })
        .then((stream) => {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => {
            // Wait for video to be fully loaded before playing
            videoRef.current.play();
            
            // Set canvas dimensions to match video dimensions
            const videoWidth = videoRef.current.videoWidth;
            const videoHeight = videoRef.current.videoHeight;
            
            console.log(`Video dimensions: ${videoWidth}x${videoHeight}`);
            
            // Ensure we have valid dimensions before proceeding
            if (videoWidth > 0 && videoHeight > 0) {
              canvasRef.current.width = videoWidth;
              canvasRef.current.height = videoHeight;
              
              webcamRunningRef.current = true;
              
              // Small delay to ensure video is fully initialized
              setTimeout(() => {
                predictWebcam();
              }, 500);
            } else {
              console.error("Invalid video dimensions:", videoWidth, videoHeight);
            }
          };
        })
        .catch((err) => {
          console.error("Error accessing webcam:", err);
        });
      }
    };

    const predictWebcam = async () => {
      if (!videoRef.current || !poseLandmarker || !webcamRunningRef.current) {
        requestAnimationFrame(predictWebcam);
        return;
      }

      // Check if video is ready and has valid dimensions
      if (videoRef.current.readyState !== 4 || 
          videoRef.current.videoWidth === 0 || 
          videoRef.current.videoHeight === 0) {
        console.log("Video not ready yet, waiting...");
        requestAnimationFrame(predictWebcam);
        return;
      }

      const startTimeMs = performance.now();
      
      try {
        // Process the video frame
        const results = poseLandmarker.detectForVideo(videoRef.current, startTimeMs);
        
        // Draw results
        drawResults(results);
        
        // Process landmarks for exercise detection
        if (results.landmarks && results.landmarks.length > 0) {
          const landmarks = results.landmarks[0];
          
          // Create a frame of keypoints
          const keypoints = Array.from(landmarks).map(lm => [lm.x, lm.y, lm.z, lm.visibility]);
          
          // Add to sequence and maintain length
          poseSequence.current.push(keypoints);
          if (poseSequence.current.length > T * 2) { // Keep twice the needed frames to avoid frequent shifts
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
          if (poseSequence.current.length >= T && tfliteModelRef.current) {
            // Run inference every 5 frames to avoid overloading
            if (modelDebugInfo.processedFrames % 5 === 0) {
              console.log(`Running inference with ${poseSequence.current.length} frames`);
              runExerciseClassification();
            }
          }
        }
      } catch (error) {
        console.error("Error in pose detection:", error);
      }
      
      // Call the next frame
      requestAnimationFrame(predictWebcam);
    };

    // Initialize everything
    const setup = async () => {
      await initializePoseLandmarker();
      await loadModel();
    };
    
    setup();

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

        {/* Video container */}
        <div 
          style={{ 
            position: 'relative',
            width: '640px', 
            height: '480px',
            overflow: 'hidden',
            border: '2px solid #333'
          }}
        >
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
          
          {/* Model status overlay with more details */}
          <div 
            style={{
              position: 'absolute',
              bottom: 10,
              left: 10,
              background: 'rgba(0,0,0,0.7)',
              padding: '5px 10px',
              borderRadius: '5px',
              fontSize: '12px',
              zIndex: 1000
            }}
          >
            {modelLoadingState.status === 'success' ? (
              <span className="text-green-400">Model: ✅ Ready</span>
            ) : modelLoadingState.status === 'error' ? (
              <span className="text-red-400">Model: ❌ Error</span>
            ) : (
              <span className="text-yellow-400">
                Model: ⏳ {modelLoadingState.message}
              </span>
            )}
          </div>
        </div>
        
        {/* Model Loading Error Display */}
        {modelLoadingState.status === 'error' && (
          <div className="mt-4 w-full max-w-2xl bg-red-900 rounded-lg p-4 text-white">
            <h3 className="text-xl font-bold mb-2">Model Loading Error</h3>
            <p>{modelLoadingState.message}</p>
            <div className="mt-3 bg-red-800 p-3 rounded text-sm overflow-auto max-h-24">
              <code>{modelLoadingState.error}</code>
            </div>
            
            <button 
              className="mt-4 bg-red-700 hover:bg-red-600 text-white font-bold py-2 px-4 rounded"
              onClick={() => setModelLoadingState(prev => ({...prev, attempts: prev.attempts + 1}))}
            >
              Retry Model Loading
            </button>
          </div>
        )}
        
        {/* Model Output Panel - Show only when model is loaded */}
        {modelLoadingState.status === 'success' && (
          <div className="mt-4 w-full max-w-2xl bg-gray-800 rounded-lg p-4">
            <h2 className="text-xl font-bold mb-2">Model Output</h2>
            
            {/* Current Exercise */}
            <div className="mb-4 p-3 bg-gray-700 rounded-lg">
              <p className="text-gray-400 text-sm">Detected Exercise:</p>
              <p className="text-3xl font-bold text-green-400">
                {currentExercise || "No exercise detected"}
              </p>
              <div className="flex justify-between items-center mt-2">
                <p className="text-gray-400 text-sm">Confidence:</p>
                <p className="text-md font-semibold">
                  {(confidence * 100).toFixed(1)}%
                </p>
              </div>
              
              {/* Confidence bar */}
              <div className="w-full bg-gray-600 h-2 rounded-full mt-1 overflow-hidden">
                <div 
                  className="bg-green-500 h-full rounded-full" 
                  style={{ width: `${Math.max(confidence * 100, 0)}%` }}
                ></div>
              </div>
            </div>
            
            {/* Model Debug Information */}
            <div className="p-3 bg-gray-700 rounded-lg text-sm text-gray-300">
              <h3 className="font-bold mb-1 text-gray-400">Model Diagnostics</h3>
              <div className="grid grid-cols-2 gap-2">
                <p>Input Shape: {modelDebugInfo.inputShape}</p>
                <p>Inference Time: {modelDebugInfo.inferenceTime}ms</p>
                <p>Processed Frames: {modelDebugInfo.processedFrames}</p>
                <p>Last Raw Prediction: {modelDebugInfo.lastPrediction}</p>
              </div>
            </div>
          </div>
        )}

        {/* Show loading progress when model is loading */}
        {(modelLoadingState.status === 'initializing' || modelLoadingState.status === 'loading') && (
          <div className="mt-4 w-full max-w-2xl bg-gray-800 rounded-lg p-4 text-center">
            <h2 className="text-xl font-bold mb-2">Loading Model</h2>
            <div className="w-full bg-gray-700 h-2 rounded-full mb-3">
              <div className="bg-blue-500 h-full rounded-full w-1/2 animate-pulse"></div>
            </div>
            <p className="text-gray-300">{modelLoadingState.message}</p>
            <p className="text-xs text-gray-400 mt-2">This may take a few moments...</p>
          </div>
        )}

        {/* Camera retry button */}
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

