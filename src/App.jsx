import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { Holistic } from "@mediapipe/holistic";
import { Camera } from "@mediapipe/camera_utils";
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

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

      // Draw pose skeleton with enhanced visuals
      if (results.poseLandmarks) {
        const connections = [
          // Torso - yellow
          { points: [[11, 12]], color: 0xffff00, width: 3 },
          // Spine - orange
          { points: [[12, 24], [11, 23], [24, 23]], color: 0xff8c00, width: 3 },
          // Arms - cyan
          { points: [[11, 13], [13, 15], [12, 14], [14, 16]], color: 0x00ffff, width: 2 },
          // Legs - purple
          { points: [[23, 25], [25, 27], [24, 26], [26, 28]], color: 0x8a2be2, width: 2 }
        ];

        // Draw enhanced connections
        connections.forEach(({ points, color, width }) => {
          const material = new THREE.LineBasicMaterial({ 
            color, 
            linewidth: width,
            linecap: 'round',
            linejoin: 'round'
          });

          points.forEach(([i, j]) => {
            if (results.poseLandmarks[i] && results.poseLandmarks[j]) {
              const start = results.poseLandmarks[i];
              const end = results.poseLandmarks[j];

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

        // Draw enhanced joints
        const jointPositions = [];
        const jointColors = [];
        results.poseLandmarks.forEach(point => {
          jointPositions.push(
            (point.x - 0.5) * 2,
            -(point.y - 0.5) * 2,
            point.z || 0
          );
          
          // Rainbow color effect based on y-position
          const hue = (point.y * 360) % 360;
          const color = new THREE.Color(`hsl(${hue}, 100%, 50%)`);
          jointColors.push(color.r, color.g, color.b);
        });

        const jointGeometry = new THREE.BufferGeometry();
        jointGeometry.setAttribute(
          'position',
          new THREE.Float32BufferAttribute(jointPositions, 3)
        );
        jointGeometry.setAttribute(
          'color',
          new THREE.Float32BufferAttribute(jointColors, 3)
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

      // Draw hands with glowing effect
      const drawHand = (landmarks, color) => {
        if (!landmarks) return;
        
        const handMaterial = new THREE.LineBasicMaterial({ 
          color,
          linewidth: 2,
        });

        // Add glow effect
        const glowMaterial = new THREE.LineBasicMaterial({ 
          color,
          linewidth: 4,
          transparent: true,
          opacity: 0.3
        });

        const fingers = [
          [0, 1, 2, 3, 4],
          [0, 5, 6, 7, 8],
          [0, 9, 10, 11, 12],
          [0, 13, 14, 15, 16],
          [0, 17, 18, 19, 20]
        ];

        fingers.forEach(finger => {
          for (let i = 1; i < finger.length; i++) {
            const start = landmarks[finger[i - 1]];
            const end = landmarks[finger[i]];
            
            const points = [
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
            ];

            const geometry = new THREE.BufferGeometry().setFromPoints(points);
            
            // Main line
            const line = new THREE.Line(geometry, handMaterial);
            scene.add(line);
            
            // Glow effect
            const glowLine = new THREE.Line(geometry, glowMaterial);
            scene.add(glowLine);
          }
        });
      };

      // Draw hands with enhanced colors
      drawHand(results.leftHandLandmarks, 0x00ff00);   // Bright green
      drawHand(results.rightHandLandmarks, 0x4169e1);  // Royal blue

      renderer.render(scene, threeCamera);
    };

    const holistic = new Holistic({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
    });

    holistic.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    });

    const detectSquat = (results) => {
      if (!results.poseLandmarks) return;

      const leftHip = results.poseLandmarks[23];
      const rightHip = results.poseLandmarks[24];
      const leftKnee = results.poseLandmarks[25];
      const rightKnee = results.poseLandmarks[26];
      const leftAnkle = results.poseLandmarks[27];
      const rightAnkle = results.poseLandmarks[28];

      if (!leftHip || !rightHip || !leftKnee || !rightKnee || !leftAnkle || !rightAnkle) return;

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

    let frameCount = 0;
    holistic.onResults((results) => {
      frameCount++;
      if (frameCount % 2 !== 0) return;

      setDetectionStatus({
        face: results.faceLandmarks?.length > 0,
        pose: results.poseLandmarks?.length > 0,
        leftHand: results.leftHandLandmarks?.length > 0,
        rightHand: results.rightHandLandmarks?.length > 0
      });

      detectSquat(results);
      drawEnhancedLandmarks(results);
    });

    // MediaPipe camera setup (renamed to avoid conflict)
    const webcam = new Camera(videoRef.current, {
      onFrame: async () => {
        await holistic.send({image: videoRef.current});
      },
      width: 640,
      height: 480
    });
    webcam.start();

    return () => {
      holistic.close();
      webcam.stop();
    };
  }, []);

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-900 text-white">
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
      </div>
    </div>
  );
}

