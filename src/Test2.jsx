import React, { useEffect, useRef, useState } from 'react';
import Detection from './components/Detection';

const Test2 = () => {
    const keypointsSequenceRef = useRef([]);
    const outputRef = useRef(null);
    const [exerciseClass, setExerciseClass] = useState(null);
    const [exerciseConfidence, setExerciseConfidence] = useState(0);
    const [landmarksDetected, setLandmarksDetected] = useState(false);
    
    // Initialize queue with a fixed capacity of 50
    // This replaces keypointsSequenceRef = useRef([])
    const MAX_SEQUENCE_LENGTH = 50;
    const keypointsQueue = useRef([]);

    useEffect(() => {
        let tf, tflite, model;
        
        // Function to load scripts dynamically
        const loadScript = (src) => {
            return new Promise((resolve, reject) => {
                const script = document.createElement('script');
                script.src = src;
                script.async = true;
                script.onload = resolve;
                script.onerror = reject;
                document.head.appendChild(script);
            });
        };

        async function loadLibrariesAndModel() {
            try {
                // Load TensorFlow.js from CDN
                await loadScript('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs/dist/tf.min.js');
                // Load TensorFlow.js TFLite from CDN
                await loadScript('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite/dist/tf-tflite.min.js');
                
                // Now we can access the global tf and tflite objects
                tf = window.tf;
                tflite = window.tflite;
                
                await tf.ready();
                console.log("TensorFlow.js ready, backend:", tf.getBackend());
                
                tflite.setWasmPath(
                    "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@latest/dist/"
                );
                console.log("TFLite WASM path set");

                // Load the actual TFLite model
                console.log("Loading TFLite model from:", "http://localhost:5173/models/stgcn_exercise_fine_tunned.tflite");
                model = await tflite.loadTFLiteModel(
                    "http://localhost:5173/models/stgcn_exercise_fine_tunned.tflite"
                );
                
                // Verify it's a real TFLite model
                console.log("✅ TFLite model loaded! Model type:", typeof model);
                console.log("Model has predict method:", typeof model.predict === 'function');

                // Test the model with zeros to verify it works
                const inputTensor = tf.zeros([1, 50, 33, 8]); // Shape needed by the model
                console.log("Created test input tensor, shape:", inputTensor.shape);
                
                const output = model.predict(inputTensor);
                console.log("Model prediction successful, output type:", typeof output);
                
                const outputData = await output.data();
                console.log("🔍 Model output data:", outputData);
                console.log("Output length:", outputData.length); // Should match your number of classes

                inputTensor.dispose();
                output.dispose();
            } catch (error) {
                console.error("Error loading libraries or model:", error);
            }
        }

        // Called by Detection component when pose landmarks are detected
        function onPoseLandmarksReceived(result) {
            if (result && result.landmarks && result.landmarks.length > 0) {
                // Convert landmarks to the format our model needs
                const keypoints = convertLandmarksToVector4(result.landmarks[0]);
                
                // Add the keypoints to our sequence queue
                updateSequence(keypoints);
                
                // Only process when we have exactly 50 frames
                if (keypointsQueue.current.length === MAX_SEQUENCE_LENGTH) {
                    console.log("Full sequence of 50 frames collected. Processing input...");
                    prepareInputAndInvoke(result);
                } else {
                    console.log(`Building sequence: ${keypointsQueue.current.length}/${MAX_SEQUENCE_LENGTH} frames`);
                }
            }
        }

        function convertLandmarksToVector4(landmarks) {
            const keypoints = new Array(33);
            
            // Using landmark mapping to ensure correct indexing
            const POSE_LANDMARKS = {
                0: 'nose', 1: 'left_eye_inner', 2: 'left_eye', 3: 'left_eye_outer',
                4: 'right_eye_inner', 5: 'right_eye', 6: 'right_eye_outer',
                7: 'left_ear', 8: 'right_ear', 9: 'mouth_left', 10: 'mouth_right',
                11: 'left_shoulder', 12: 'right_shoulder', 13: 'left_elbow', 14: 'right_elbow',
                15: 'left_wrist', 16: 'right_wrist', 17: 'left_pinky', 18: 'right_pinky',
                19: 'left_index', 20: 'right_index', 21: 'left_thumb', 22: 'right_thumb',
                23: 'left_hip', 24: 'right_hip', 25: 'left_knee', 26: 'right_knee',
                27: 'left_ankle', 28: 'right_ankle', 29: 'left_heel', 30: 'right_heel',
                31: 'left_foot_index', 32: 'right_foot_index'
            };
            
            // Initialize all keypoints to zeros first
            for (let i = 0; i < 33; i++) {
                keypoints[i] = [0, 0, 0, 0]; // Default all to zero
            }
            
            // Populate with actual values if they exist and have good visibility
            for (let i = 0; i < 33; i++) {
                if (landmarks[i]) {
                    const lm = landmarks[i];
                    const visibility = lm.visibility || 0;
                    
                    // Only use landmarks with reasonable visibility
                    if (visibility > 0.3) {
                        keypoints[i] = [
                            lm.x || 0,
                            lm.y || 0, 
                            lm.z || 0, 
                            visibility
                        ];
                    }
                    // If visibility is low, the keypoint remains [0,0,0,0]
                }
            }
            
            return keypoints;
        }

        function updateSequence(keypoints) {
            // Add new frame to the queue
            keypointsQueue.current.push(keypoints);
            
            // If we exceed capacity, remove the oldest frame
            if (keypointsQueue.current.length > MAX_SEQUENCE_LENGTH) {
                keypointsQueue.current.shift();
            }
            
            // Debug check - log a sample of keypoints periodically
            if (keypointsQueue.current.length % 10 === 0) {
                console.log(`Queue size: ${keypointsQueue.current.length}/${MAX_SEQUENCE_LENGTH}`);
                // Log sample of non-zero keypoints to verify data
                const nonZeroCount = keypointsQueue.current[keypointsQueue.current.length-1]
                    .filter(kp => kp[0] !== 0 || kp[1] !== 0 || kp[2] !== 0)
                    .length;
                console.log(`Latest frame has ${nonZeroCount}/33 non-zero keypoints`);
            }
            
            // Only log when we have exactly 50 frames
            if (keypointsQueue.current.length === MAX_SEQUENCE_LENGTH) {
                console.log("Full sequence of 50 frames collected. Processing input...");
            }
        }

        async function prepareInputAndInvoke(result) {
            try {
                if (!model || !tf) {
                    console.error("Model or TensorFlow not loaded");
                    return;
                }
                
                // Throttle predictions to reduce CPU/GPU load
                const now = Date.now();
                if (window.lastPredictionTime && now - window.lastPredictionTime < 200) {
                    return; // Limit to max 5 predictions per second
                }
                window.lastPredictionTime = now;
                
                // Double-check we have exactly 50 frames
                if (keypointsQueue.current.length !== MAX_SEQUENCE_LENGTH) {
                    console.log(`Waiting for complete sequence, current length: ${keypointsQueue.current.length}/${MAX_SEQUENCE_LENGTH}`);
                    return;
                }
                
                // Use tf.tidy to automatically clean up intermediate tensors
                tf.tidy(() => {
                    // Create array to hold the augmented data
                    const augmentedData = [];
                    
                    // Process each frame from our sequence of 50
                    for (let frameIdx = 0; frameIdx < MAX_SEQUENCE_LENGTH; frameIdx++) {
                        const frame = keypointsQueue.current[frameIdx];
                        
                        // Calculate joint angles
                        const angles = computeJointAngles(frame);
                        
                        // Create augmented frame
                        const augmentedFrame = [];
                        
                        // Add features for each of the 33 keypoints
                        for (let jointIdx = 0; jointIdx < 33; jointIdx++) {
                            const joint = frame[jointIdx];
                            
                            // Combine keypoint data with angles
                            augmentedFrame.push([
                                joint[0], joint[1], joint[2], joint[3],  // x, y, z, visibility from original landmarks
                                angles[0]/180, angles[1]/180, angles[2]/180, angles[3]/180  // normalized angles
                            ]);
                        }
                        
                        augmentedData.push(augmentedFrame);
                    }
                    
                    // Create tensor with shape [1, 50, 33, 8]
                    const inputTensor = tf.tensor(augmentedData).expandDims(0);
                    console.log("Input tensor shape:", inputTensor.shape);
                    
                    // Make prediction with the TFLite model
                    const output = model.predict(inputTensor);
                    
                    // Get output data as array
                    output.data().then(outputData => {
                        // Find the index of the highest confidence class
                        const predClass = argMax(outputData);
                        const confidence = outputData[predClass];
                        
                        // Map numeric class to exercise name
                        const exerciseNames = ['TreePose', 'Lunges', 'Push-Up', 'Squat'];
                        
                        // Log all prediction values
                        console.log("Prediction values for all exercises:");
                        exerciseNames.forEach((name, index) => {
                            console.log(`${name}: ${(outputData[index] * 100).toFixed(2)}%`);
                        });
                        
                        // Only update UI for confident predictions
                        if (confidence > 0.4) {
                            const exerciseName = exerciseNames[predClass] || `Exercise ${predClass}`;
                            
                            console.log(`TFLite prediction: ${exerciseName} (class ${predClass}), Confidence: ${(confidence * 100).toFixed(1)}%`);
                            
                            // Update UI with results
                            handleExerciseSession(exerciseName, confidence);
                        }
                        
                        // IMPORTANT: Clear the queue after prediction to reduce memory pressure
                        keypointsQueue.current = [];
                        console.log("Cleared frame buffer after prediction. Building new sequence...");
                    });
                });
            } catch (error) {
                console.error("Error in prepareInputAndInvoke:", error);
                // Clear queue even on error to avoid getting stuck
                keypointsQueue.current = [];
            }
        }

        function argMax(array) {
            return array.indexOf(Math.max(...array));
        }

        function handleExerciseSession(exercise, confidence) {
            // Update state with prediction results
            setExerciseClass(exercise);
            setExerciseConfidence(confidence);
            
            if (outputRef.current) {
                outputRef.current.innerText = `Exercise: ${exercise}, Confidence: ${(confidence * 100).toFixed(2)}%`;
            }
        }

        // Add this function to compute joint angles with better error handling
        function computeJointAngles(frame) {
            try {
                if (!frame || frame.length < 33) {
                    return [0, 0, 0, 0]; // Default angles if frame is invalid
                }
                
                // Extract joint positions
                const leftShoulder = frame[11]; // left_shoulder
                const leftElbow = frame[13];    // left_elbow
                const leftWrist = frame[15];    // left_wrist
                
                const rightShoulder = frame[12]; // right_shoulder
                const rightElbow = frame[14];    // right_elbow
                const rightWrist = frame[16];    // right_wrist
                
                const leftHip = frame[23];      // left_hip
                const leftKnee = frame[25];     // left_knee
                const leftAnkle = frame[27];    // left_ankle
                
                const rightHip = frame[24];     // right_hip
                const rightKnee = frame[26];    // right_knee
                const rightAnkle = frame[28];   // right_ankle
                
                // Calculate angle between three points with better error handling
                const calculateAngle = (a, b, c) => {
                    // Skip calculation if any point is missing or has low visibility (fourth value)
                    if (!a || !b || !c || 
                        a[0] === 0 || b[0] === 0 || c[0] === 0 || 
                        a[3] < 0.5 || b[3] < 0.5 || c[3] < 0.5) {
                        return 0; // Return default angle
                    }
                    
                    try {
                        // Create vectors
                        const vector1 = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
                        const vector2 = [c[0] - b[0], c[1] - b[1], c[2] - b[2]];
                        
                        // Calculate magnitudes with safety checks
                        const magnitude1 = Math.sqrt(vector1[0]**2 + vector1[1]**2 + vector1[2]**2);
                        const magnitude2 = Math.sqrt(vector2[0]**2 + vector2[1]**2 + vector2[2]**2);
                        
                        // Prevent division by zero
                        if (magnitude1 < 0.0001 || magnitude2 < 0.0001) {
                            return 0;
                        }
                        
                        // Calculate dot product
                        const dotProduct = vector1[0] * vector2[0] + vector1[1] * vector2[1] + vector1[2] * vector2[2];
                        
                        // Ensure the value is in valid range for acos
                        const cosine = Math.max(-1, Math.min(1, dotProduct / (magnitude1 * magnitude2)));
                        
                        // Calculate and return angle in degrees
                        return Math.round(Math.acos(cosine) * (180 / Math.PI));
                    } catch (e) {
                        console.warn("Error calculating angle:", e);
                        return 0;
                    }
                };
                
                // Calculate angles with safety checks
                let leftElbowAngle = 0, rightElbowAngle = 0, leftKneeAngle = 0, rightKneeAngle = 0;
                
                try {
                    leftElbowAngle = calculateAngle(leftShoulder, leftElbow, leftWrist);
                } catch (e) { console.warn("Left elbow angle calculation failed"); }
                
                try {
                    rightElbowAngle = calculateAngle(rightShoulder, rightElbow, rightWrist);
                } catch (e) { console.warn("Right elbow angle calculation failed"); }
                
                try {
                    leftKneeAngle = calculateAngle(leftHip, leftKnee, leftAnkle);
                } catch (e) { console.warn("Left knee angle calculation failed"); }
                
                try {
                    rightKneeAngle = calculateAngle(rightHip, rightKnee, rightAnkle);
                } catch (e) { console.warn("Right knee angle calculation failed"); }
                
                return [leftElbowAngle, rightElbowAngle, leftKneeAngle, rightKneeAngle];
            } catch (error) {
                console.error("Error computing joint angles:", error);
                return [0, 0, 0, 0];
            }
        }

        loadLibrariesAndModel();

        // Expose the function to window for Detection component to call
        window.onPoseLandmarksReceived = onPoseLandmarksReceived;

        // Cleanup function
        return () => {
            if (model) {
                model.dispose();
            }
            
            // Remove the global function
            delete window.onPoseLandmarksReceived;
        };
    }, []);

    // Handler for Detection component
    const handlePoseLandmarksReceived = (results) => {
        if (results.landmarks && results.landmarks.length > 0) {
            setLandmarksDetected(true);
            
            // Log landmarks received from Detection component
            console.log("Landmarks received in handlePoseLandmarksReceived:", results.landmarks);

            // Send landmarks to parent component for TFLite model processing
            if (window.onPoseLandmarksReceived) {
                window.onPoseLandmarksReceived(results);
            } else {
                console.warn("window.onPoseLandmarksReceived is not defined");
            }
        }
    };

    return (
        <div className="exercise-detection-container">
            <h1>Exercise Detection</h1>
            <div id="exercise-output" ref={outputRef} className="exercise-output">
                Loading model...
            </div>
            
            {/* Prediction display */}
            {exerciseClass !== null && (
                <div className="prediction-display">
                    <h2>Current Exercise:</h2>
                    <div className="exercise-class">{exerciseClass}</div>
                    <div className="confidence-meter">
                        <div className="confidence-label">Confidence: {(exerciseConfidence * 100).toFixed(2)}%</div>
                        <div className="confidence-bar-container">
                            <div 
                                className="confidence-bar-fill" 
                                style={{ width: `${exerciseConfidence * 100}%` }}
                            ></div>
                        </div>
                    </div>
                </div>
            )}
            
            {/* MediaPipe Detection Component */}
            <Detection onPoseLandmarksReceived={handlePoseLandmarksReceived} />
        </div>
    );
};

export default Test2;