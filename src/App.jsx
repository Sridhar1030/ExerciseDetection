import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { Pose } from "@mediapipe/pose";
import { Camera } from "@mediapipe/camera_utils";
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

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

  useEffect(() => {
    if (!videoRef.current || !canvasRef.current) return;

    // Setup Three.js scene with better lighting
    const scene = new THREE.Scene();
    const threeCamera = new THREE.PerspectiveCamera(75, 640 / 480, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({
      canvas: canvasRef.current,
      alpha: true,
      antialias: true // Smoother lines
    });
    renderer.setSize(640, 480);
    threeCamera.position.z = 2;

    // Add ambient light
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    // Add point light
    const pointLight = new THREE.PointLight(0xffffff, 1);
    pointLight.position.set(0, 0, 2);
    scene.add(pointLight);

    const drawEnhancedLandmarks = (results) => {
      // Clear scene except lights
      scene.children = scene.children.filter(child => 
        child.type === 'AmbientLight' || child.type === 'PointLight'
      );

      if (results.poseLandmarks) {
        // Increase visibility threshold for drawing
        const VISIBILITY_THRESHOLD = 0.7; // Increased from 0.3 to 0.7

        // Updated connections to show more body parts
        const connections = [
          // Face
          { points: [[1, 2], [2, 3], [4, 5], [5, 6], [9, 10]], color: 0xffa500, width: 2 },
          // Neck
          { points: [[11, 12], [0, 12], [0, 11]], color: 0xff0000, width: 3 },
          // Arms
          { points: [[11, 13], [13, 15], [15, 17], [15, 19], [15, 21]], color: 0x00ff00, width: 2 },
          { points: [[12, 14], [14, 16], [16, 18], [16, 20], [16, 22]], color: 0x00ff00, width: 2 },
          // Torso
          { points: [[11, 23], [12, 24], [23, 24]], color: 0x0000ff, width: 3 },
          // Legs
          { points: [[23, 25], [25, 27], [27, 29], [27, 31], [29, 31]], color: 0xff00ff, width: 2 },
          { points: [[24, 26], [26, 28], [28, 30], [28, 32], [30, 32]], color: 0xff00ff, width: 2 },
        ];

        // Draw enhanced connections only for visible landmarks
        connections.forEach(({ points, color, width }) => {
          const material = new THREE.LineBasicMaterial({ 
            color, 
            linewidth: width,
            linecap: 'round',
            linejoin: 'round'
          });

          points.forEach(([i, j]) => {
            const start = results.poseLandmarks[i];
            const end = results.poseLandmarks[j];
            
            // Only draw connection if both points are visible enough
            if (start && end && 
                start.visibility > VISIBILITY_THRESHOLD && 
                end.visibility > VISIBILITY_THRESHOLD) {
              const geometry = new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(
                  (start.x - 0.5) * 2,
                  -(start.y - 0.5) * 2,
                  start.z || 0
                ),
                new THREE.Vector3(
                  (end.x - 0.5) * 2,
                  -(end.y - 0.5) * 2,
                  end.z || 0
                )
              ]);

              const line = new THREE.Line(geometry, material);
              scene.add(line);
            }
          });
        });

        // Only draw joints that are visible enough
        const visibleJointPositions = [];
        const visibleJointColors = [];
        
        results.poseLandmarks.forEach((point, index) => {
          if (point.visibility > VISIBILITY_THRESHOLD) {
            visibleJointPositions.push(
              (point.x - 0.5) * 2,
              -(point.y - 0.5) * 2,
              point.z || 0
            );
            
            let color;
            if (index <= 10) color = new THREE.Color(0xffa500);
            else if (index <= 16) color = new THREE.Color(0x00ff00);
            else if (index <= 22) color = new THREE.Color(0x00ffff);
            else if (index <= 24) color = new THREE.Color(0x0000ff);
            else if (index <= 28) color = new THREE.Color(0xff00ff);
            else color = new THREE.Color(0xff0000);
            
            visibleJointColors.push(color.r, color.g, color.b);
          }
        });

        if (visibleJointPositions.length > 0) {
          const jointGeometry = new THREE.BufferGeometry();
          jointGeometry.setAttribute(
            'position',
            new THREE.Float32BufferAttribute(visibleJointPositions, 3)
          );
          jointGeometry.setAttribute(
            'color',
            new THREE.Float32BufferAttribute(visibleJointColors, 3)
          );

          const jointMaterial = new THREE.PointsMaterial({
            size: 0.04,
            sizeAttenuation: true,
            vertexColors: true,
            map: new THREE.TextureLoader().load('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAWJQAAFiUBSVIk8AAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAAKwSURBVFiF7ZbPaxNBFMe/m4lttYtNBem9Db3Yf8CTXgQ9KAhePHhRFF0CQkXwIAjVS3tTBKEgAREv/gGCiGeTQBsI2oCiHjwWRE0DJd1us5v5edhN2KbZ7I+kVSjzgYHZmXnz/e57b2ZWiQh2s/TuVv4fgLhnZv4s0AJwFcA9AH0AL0Tk7jY9XNXZW9q2vwI4XGH6hojcKjtZNwKxGOL8BXDGaDwEcFlEfv8VACLyJo7j9wAeMPMNZt7PzEcBXALwKsuynQHgnB0ies3MZ0QkYeYxgGsALjLzGWb+1ul0JjsGICKfAZwDcArAC2vtOWZ+AuBQkiTnReQTM+/bUQAR+QHgYpqmx4hoHcA9a+0TAHestc+UUv2dBCiYZwAeOudOE9FQKbXmnLvFzE92FICIEqXUNwCrAB4rpV4qpb4CWBWRdKcBCvNPAUyYeQAgBjAQkXQ7xVg7AM65QwCOAHhvrf3unGsw8zFmPqCU+rBrAMzcBPAQwIq19qO1dpWZHwEYMPOB3QQAgB6A+wDa1toLAEBE94loAGCFmRu7CUBEtNYaM/M6gBVr7RfnnFFKrQEYaa3NrgEUEJeZ+SkzjwCMtNZvlVJjETG7DUBEpLUeM/MYwBdmHmutN5RSGxsb2a4CFEtxkZnXmXmstU6VUhsiYkQk2zUAInLMPGbmDa11qpT6ISKGiGw3B7UCEBEZYwyA1Fr7k4iMtdZsrU0B/GRmU2W+VgAiYowxrLVOrbVGRDJrbUZEGTNnRGTrBGDmEYAxM4+Y2RhjrIhkzJyJiGVmWwYRVQIw8zsAHQAdpdQbpdQvpdQvpdRbpdQ7AN04jt+X+a8E0Gq1+gDaABaJ6DwRnSOiRQBtAP1Wq9Uv818JoNvtbgI4CeA0gJMA/uxut9vdLPP/G0vxb9c/A/APARJ5iDCPZqsAAAAASUVORK5CYII='),
            transparent: true,
            alphaTest: 0.5
          });

          const joints = new THREE.Points(jointGeometry, jointMaterial);
          scene.add(joints);
        }
      }

      renderer.render(scene, threeCamera);
    };

    const pose = new Pose({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`
    });

    pose.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      enableSegmentation: false,
      smoothSegmentation: false,
      minDetectionConfidence: 0.2,
      minTrackingConfidence: 0.2
    });

    const detectPoses = (results) => {
      if (!results.poseLandmarks) return;

      // Common visibility threshold check
      const VISIBILITY_THRESHOLD = 0.7;

      // 1. T-Pose Detection
      const detectTPose = () => {
        const leftShoulder = results.poseLandmarks[11];
        const rightShoulder = results.poseLandmarks[12];
        const leftElbow = results.poseLandmarks[13];
        const rightElbow = results.poseLandmarks[14];
        const leftWrist = results.poseLandmarks[15];
        const rightWrist = results.poseLandmarks[16];

        // Check visibility
        if (![leftShoulder, rightShoulder, leftElbow, rightElbow, leftWrist, rightWrist]
            .every(point => point.visibility > VISIBILITY_THRESHOLD)) return false;

        // Arms should be horizontal (y coordinates similar) and extended
        const armsHorizontal = 
          Math.abs(leftWrist.y - leftShoulder.y) < 0.1 &&
          Math.abs(rightWrist.y - rightShoulder.y) < 0.1;
        
        const armsExtended = 
          Math.abs(leftWrist.x - leftShoulder.x) > 0.3 &&
          Math.abs(rightWrist.x - rightShoulder.x) > 0.3;

        return armsHorizontal && armsExtended;
      };

      // 2. Tree Pose Detection
      const detectTreePose = () => {
        const leftKnee = results.poseLandmarks[25];
        const rightKnee = results.poseLandmarks[26];
        const leftAnkle = results.poseLandmarks[27];
        const rightAnkle = results.poseLandmarks[28];
        const leftHip = results.poseLandmarks[23];
        const rightHip = results.poseLandmarks[24];

        // Check visibility
        if (![leftKnee, rightKnee, leftAnkle, rightAnkle, leftHip, rightHip]
            .every(point => point.visibility > VISIBILITY_THRESHOLD)) return false;

        // One foot should be raised to opposite knee
        const leftFootRaised = Math.abs(leftAnkle.y - rightKnee.y) < 0.1;
        const rightFootRaised = Math.abs(rightAnkle.y - leftKnee.y) < 0.1;

        return leftFootRaised || rightFootRaised;
      };

      // 3. Warrior Pose Detection
      const detectWarriorPose = () => {
        const leftKnee = results.poseLandmarks[25];
        const rightKnee = results.poseLandmarks[26];
        const leftAnkle = results.poseLandmarks[27];
        const rightAnkle = results.poseLandmarks[28];
        const leftHip = results.poseLandmarks[23];
        const rightHip = results.poseLandmarks[24];

        // Check visibility
        if (![leftKnee, rightKnee, leftAnkle, rightAnkle, leftHip, rightHip]
            .every(point => point.visibility > VISIBILITY_THRESHOLD)) return false;

        // One leg bent, one leg straight
        const legSpread = Math.abs(leftAnkle.x - rightAnkle.x) > 0.4;
        const oneKneeBent = 
          (Math.abs(leftKnee.y - leftAnkle.y) > 0.2) ||
          (Math.abs(rightKnee.y - rightAnkle.y) > 0.2);

        return legSpread && oneKneeBent;
      };

      // 4. Hands Up Pose Detection
      const detectHandsUp = () => {
        const leftShoulder = results.poseLandmarks[11];
        const rightShoulder = results.poseLandmarks[12];
        const leftWrist = results.poseLandmarks[15];
        const rightWrist = results.poseLandmarks[16];

        // Check visibility
        if (![leftShoulder, rightShoulder, leftWrist, rightWrist]
            .every(point => point.visibility > VISIBILITY_THRESHOLD)) return false;

        // Hands should be above shoulders
        return (leftWrist.y < leftShoulder.y - 0.2) &&
               (rightWrist.y < rightShoulder.y - 0.2);
      };

      // 5. Squat Detection (your existing detection)
      const detectSquat = () => {
        if (!results.poseLandmarks) return;

        const leftHip = results.poseLandmarks[23];
        const rightHip = results.poseLandmarks[24];
        const leftKnee = results.poseLandmarks[25];
        const rightKnee = results.poseLandmarks[26];
        const leftAnkle = results.poseLandmarks[27];
        const rightAnkle = results.poseLandmarks[28];

        // Lower the visibility threshold
        const VISIBILITY_THRESHOLD = 0.3;  // Reduced from 0.65
        const requiredPoints = [leftHip, rightHip, leftKnee, rightKnee, leftAnkle, rightAnkle];
        
        if (!requiredPoints.every(point => 
          point && point.visibility && point.visibility > VISIBILITY_THRESHOLD
        )) {
          console.log("Some required points not visible enough");
          return;
        }

        const hipY = (leftHip.y + rightHip.y) / 2;
        const kneeY = (leftKnee.y + rightKnee.y) / 2;
        const ankleY = (leftAnkle.y + rightAnkle.y) / 2;

        // Calibration phase
        if (squatState.current.isCalibrating) {
          squatState.current.calibrationSum += hipY;
          squatState.current.calibrationFrames++;

          if (squatState.current.calibrationFrames >= SQUAT_SETTINGS.CALIBRATION_FRAMES) {
            squatState.current.startY = squatState.current.calibrationSum / SQUAT_SETTINGS.CALIBRATION_FRAMES;
            squatState.current.isCalibrating = false;
            console.log('Calibration complete, starting Y:', squatState.current.startY);
          }
          return;
        }

        const now = Date.now();
        const timeSinceLastSquat = now - squatState.current.lastSquatTime;

        // Calculate distances for form checking
        const hipToKnee = Math.abs(hipY - kneeY);
        const kneeToAnkle = Math.abs(kneeY - ankleY);
        
        // Check if knees are bent and form is good
        const properForm = hipToKnee > 0.15 && kneeToAnkle > 0.1;
        const hipDropAmount = hipY - squatState.current.startY;

        // More sensitive thresholds
        const SQUAT_DOWN_THRESHOLD = 0.2;    // How low to go for squat
        const SQUAT_UP_THRESHOLD = 0.1;      // How high to come up to complete
        const MIN_REP_TIME = 500;            // Minimum ms between reps

        // Detect squat phases with stricter conditions
        if (!squatState.current.isInSquat && 
            hipDropAmount > SQUAT_DOWN_THRESHOLD && 
            properForm) {
          console.log('Entering squat', hipDropAmount);
          squatState.current.isInSquat = true;
        }
        else if (squatState.current.isInSquat && 
                 hipDropAmount < SQUAT_UP_THRESHOLD && 
                 timeSinceLastSquat > MIN_REP_TIME) {
          console.log('Completing squat', hipDropAmount);
          squatState.current.isInSquat = false;
          squatState.current.lastSquatTime = now;
          setExerciseCount(prev => prev + 1);
        }
      };

      // Check all poses and return the detected one
      if (detectTPose()) return "T-Pose";
      if (detectTreePose()) return "Tree Pose";
      if (detectWarriorPose()) return "Warrior Pose";
      if (detectHandsUp()) return "Hands Up";
      if (detectSquat()) return "Squat";

      return "No pose detected";
    };

    let frameCount = 0;
    pose.onResults((results) => {
      frameCount++;
      if (frameCount % 2 !== 0) return;

      if (results.poseLandmarks) {
        // Update landmark data
        const newLandmarkData = {};
        results.poseLandmarks.forEach((point, index) => {
          newLandmarkData[index] = {
            name: POSE_LANDMARKS[index],
            x: point.x.toFixed(3),
            y: point.y.toFixed(3),
            z: (point.z || 0).toFixed(2),
            visibility: point.visibility?.toFixed(3) || 0,
          };
        });
        setLandmarkData(newLandmarkData);
      }

      // Add logging to help diagnose issues
      if (!results.poseLandmarks) {
        console.log("No pose landmarks detected");
      } else {
        console.log("Pose detected with", results.poseLandmarks.length, "landmarks");
      }

      setDetectionStatus({
        pose: results.poseLandmarks?.length > 0,
        leftHand: false,
        rightHand: false
      });

      // Add pose detection
      const detectedPose = detectPoses(results);
      setCurrentPose(detectedPose);

      drawEnhancedLandmarks(results);
    });

    // Update camera setup with error handling
    const webcam = new Camera(videoRef.current, {
      onFrame: async () => {
        try {
          await pose.send({image: videoRef.current});
        } catch (error) {
          console.error("Error in pose detection:", error);
        }
      },
      width: 640,
      height: 480
    });

    webcam.start().catch(error => {
      console.error("Error starting camera:", error);
    });

    return () => {
      pose.close();
      webcam.stop();
    };
  }, []);

  // Function to group landmarks by body part
  const groupedLandmarks = {
    face: Array.from({length: 11}, (_, i) => i),
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
      {/* Left side - Video and Counter */}
      <div className="flex flex-col items-center">
        <h1 className="text-3xl font-bold mb-4">Squat Counter 🏋️‍♂️</h1>
        
        <div className="relative">
          <video 
            ref={videoRef} 
            className="rounded-lg transform scale-x-[-1]" 
            width="640" 
            height="480" 
            muted 
            playsInline
          />
          <canvas 
            ref={canvasRef} 
            className="absolute top-0 left-0 transform scale-x-[-1]"
            width="640"
            height="480"
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

