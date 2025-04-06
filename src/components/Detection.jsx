import React, { useEffect, useRef, useState } from 'react';
import { PoseLandmarker, FilesetResolver, DrawingUtils } from "@mediapipe/tasks-vision";

const Detection = ({ onPoseLandmarksReceived }) => {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const [landmarksDetected, setLandmarksDetected] = useState(false);
    const webcamRunningRef = useRef(false);
    const frameCountRef = useRef(0);
    const errorCountRef = useRef(0);
    const lastValidResultsRef = useRef(null);

    useEffect(() => {
        if (!videoRef.current || !canvasRef.current) return;

        const canvasCtx = canvasRef.current.getContext('2d');
        const drawingUtils = new DrawingUtils(canvasCtx);

        let poseLandmarker;

        const drawResults = (results) => {
            if (!canvasRef.current) return;
            
            // Always clear the canvas first, regardless of landmarks
            canvasCtx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
            
            // If we have valid landmarks, update our reference and draw them
            if (results && results.landmarks && results.landmarks.length > 0) {
                // Store as last valid results
                lastValidResultsRef.current = results;
                
                try {
                    for (const landmarks of results.landmarks) {
                        try {
                            drawingUtils.drawConnectors(
                                landmarks,
                                PoseLandmarker.POSE_CONNECTIONS
                            );
                            
                            drawingUtils.drawLandmarks(
                                landmarks
                            );
                        } catch (error) {
                            console.warn("Drawing error:", error.message);
                        }
                    }
                } catch (error) {
                    console.warn("Error in draw results:", error.message);
                }
            } 
            // If no landmarks but we have previous valid results, use those instead
            // This creates a "freezing" effect instead of disappearing
            else if (lastValidResultsRef.current && lastValidResultsRef.current.landmarks) {
                try {
                    for (const landmarks of lastValidResultsRef.current.landmarks) {
                        // Draw with slightly faded color to indicate it's from previous frame
                        try {
                            drawingUtils.drawConnectors(
                                landmarks,
                                PoseLandmarker.POSE_CONNECTIONS,
                                {color: 'rgba(255, 255, 255, 0.5)'} // More transparent to indicate old data
                            );
                            
                            drawingUtils.drawLandmarks(
                                landmarks,
                                {color: 'rgba(0, 255, 0, 0.5)'} // More transparent landmarks
                            );
                        } catch (error) {
                            console.warn("Drawing previous landmarks error:", error.message);
                        }
                    }
                } catch (error) {
                    console.warn("Error in draw previous results:", error.message);
                }
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
                    minPoseDetectionConfidence: 0.2,
                    minPosePresenceConfidence: 0.2,
                    minTrackingConfidence: 0.2
                });

                enableCam();
            } catch (error) {
                console.error("Error initializing pose landmarker:", error);
                // Try again with CPU delegate if GPU fails
                try {
                    const vision = await FilesetResolver.forVisionTasks(
                        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
                    );
    
                    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
                        baseOptions: {
                            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
                            delegate: "GPU" // Fallback to CPU
                        },
                        runningMode: "VIDEO",
                        numPoses: 1,
                        minPoseDetectionConfidence: 0.2,
                        minPosePresenceConfidence: 0.2,
                        minTrackingConfidence: 0.2
                    });
    
                    enableCam();
                } catch (fallbackError) {
                    console.error("Error initializing pose landmarker (fallback):", fallbackError);
                }
            }
        };

        const enableCam = () => {
            if (!webcamRunningRef.current) {
                navigator.mediaDevices.getUserMedia({
                    video: {
                        width: { ideal: 480 },
                        height: { ideal: 480 },
                        frameRate: { ideal: 30 }
                    }
                })
                    .then((stream) => {
                        if (videoRef.current) {
                            videoRef.current.srcObject = stream;
                            
                            // Add event listeners for video errors
                            videoRef.current.onerror = (e) => {
                                console.error("Video error:", e);
                            };
                            
                            videoRef.current.onended = () => {
                                console.warn("Video stream ended");
                                // Try to restart stream if it ends
                                setTimeout(enableCam, 1000);
                            };
                            
                            videoRef.current.onloadedmetadata = () => {
                                try {
                                    videoRef.current.play()
                                        .then(() => {
                                            const videoWidth = videoRef.current.videoWidth;
                                            const videoHeight = videoRef.current.videoHeight;
                                            
                                            console.log(`Video dimensions: ${videoWidth}x${videoHeight}`);
                                            
                                            if (videoWidth > 0 && videoHeight > 0 && canvasRef.current) {
                                                // Set exact dimensions to match video
                                                canvasRef.current.width = videoWidth;
                                                canvasRef.current.height = videoHeight;
                                                
                                                // Calculate the correct positioning
                                                const containerWidth = 640; // Your container width
                                                const containerHeight = 480; // Your container height
                                                
                                                // Calculate scaling and positioning
                                                const scaleX = containerWidth / videoWidth;
                                                const scaleY = containerHeight / videoHeight;
                                                const scale = Math.min(scaleX, scaleY);
                                                
                                                // Set the canvas style with exact positioning
                                                canvasRef.current.style.width = `${videoWidth}px`;
                                                canvasRef.current.style.height = `${videoHeight}px`;
                                                canvasRef.current.style.position = 'absolute';
                                                canvasRef.current.style.left = '50%';
                                                canvasRef.current.style.top = '50%';
                                                canvasRef.current.style.transform = `translate(-50%, -50%) scaleX(-1) scale(${scale})`;
                                                
                                                // Match video positioning
                                                videoRef.current.style.position = 'absolute';
                                                videoRef.current.style.left = '50%';
                                                videoRef.current.style.top = '50%';
                                                videoRef.current.style.transform = `translate(-50%, -50%) scaleX(-1) scale(${scale})`;
                                                
                                                webcamRunningRef.current = true;
                                                
                                                setTimeout(() => {
                                                    predictWebcam();
                                                }, 200);
                                            } else {
                                                console.error("Invalid video or canvas dimensions");
                                                setTimeout(enableCam, 1000); // Try again
                                            }
                                        })
                                        .catch(playError => {
                                            console.error("Error playing video:", playError);
                                            setTimeout(enableCam, 1000); // Try again
                                        });
                                } catch (e) {
                                    console.error("Error in video setup:", e);
                                    setTimeout(enableCam, 1000); // Try again
                                }
                            };
                        }
                    })
                    .catch((err) => {
                        console.error("Error accessing webcam:", err);
                        // Try again with different constraints if initial attempt fails
                        setTimeout(() => {
                            navigator.mediaDevices.getUserMedia({
                                video: true // Simplify constraints
                            }).then(stream => {
                                if (videoRef.current) {
                                    videoRef.current.srcObject = stream;
                                    videoRef.current.onloadedmetadata = () => videoRef.current.play();
                                }
                            }).catch(fallbackErr => {
                                console.error("Fallback webcam access failed:", fallbackErr);
                            });
                        }, 1000);
                    });
            }
        };

        const predictWebcam = async () => {
            if (!videoRef.current || !poseLandmarker || !webcamRunningRef.current) {
                requestAnimationFrame(predictWebcam);
                return;
            }

            if (videoRef.current.readyState !== 4 ||
                videoRef.current.videoWidth === 0 ||
                videoRef.current.videoHeight === 0) {
                requestAnimationFrame(predictWebcam);
                return;
            }

            const startTimeMs = performance.now();
            
            // Ensure video is still playing
            if (videoRef.current.paused || videoRef.current.ended) {
                try {
                    await videoRef.current.play();
                } catch (e) {
                    console.warn("Could not restart video:", e);
                }
            }

            try {
                // Add basic throttling back but much lighter (3 out of 4 frames)
                // This helps reduce processing load without causing noticeable stutters
                frameCountRef.current = (frameCountRef.current + 1) % 4;
                
                // Always redraw existing pose to maintain continuity
                // This ensures something is always drawn
                if (lastValidResultsRef.current) {
                    drawResults(lastValidResultsRef.current);
                }
                
                // Only do detection on 3 out of 4 frames
                if (frameCountRef.current < 3) {
                    try {
                        // Create a detection options object with image dimensions
                        const detectionOptions = {
                            imageWidth: videoRef.current.videoWidth,
                            imageHeight: videoRef.current.videoHeight
                        };
                        
                        // Pass the options to detectForVideo
                        const results = poseLandmarker.detectForVideo(videoRef.current, startTimeMs, detectionOptions);
                        
                        // Draw results only if we have landmarks
                        if (results && results.landmarks && results.landmarks.length > 0) {
                            drawResults(results);
                            setLandmarksDetected(true);
                            
                            // Process landmarks before sending them
                            const processedResults = {
                                ...results,
                                landmarks: results.landmarks.map(landmarks => {
                                    // Create a fixed-size array for all 33 landmarks, initialized to zeros
                                    const completeSet = Array(33).fill().map(() => ({ x: 0, y: 0, z: 0, visibility: 0 }));
                                    
                                    // Fill in actual landmarks with good visibility
                                    for (let i = 0; i < landmarks.length && i < 33; i++) {
                                        const landmark = landmarks[i];
                                        if (landmark && landmark.visibility > 0.5) {
                                            completeSet[i] = {
                                                x: landmark.x || 0,
                                                y: landmark.y || 0,
                                                z: landmark.z || 0,
                                                visibility: landmark.visibility
                                            };
                                        }
                                    }
                                    
                                    return completeSet;
                                })
                            };
                            
                            // Send results to parent component
                            if (onPoseLandmarksReceived) {
                                try {
                                    onPoseLandmarksReceived(processedResults);
                                } catch (callbackError) {
                                    console.error("Error in landmarks callback:", callbackError);
                                }
                            }
                        } else {
                            setLandmarksDetected(false);
                        }
                        
                        // Reset error counter on success
                        errorCountRef.current = 0;
                    } catch (detectionError) {
                        console.error("Detection error:", detectionError);
                        errorCountRef.current++;
                    }
                }
            } catch (error) {
                console.error("Error in main detection loop:", error);
                errorCountRef.current++;
                
                // Recovery logic for persistent errors
                if (errorCountRef.current > 5) {
                    console.warn("Multiple errors detected, attempting recovery...");
                    try {
                        setTimeout(() => {
                            initializePoseLandmarker();
                        }, 1000);
                        errorCountRef.current = 0;
                    } catch (e) {
                        console.error("Recovery failed:", e);
                    }
                }
            }

            // Always schedule next frame to keep video running smoothly
            requestAnimationFrame(predictWebcam);
        };

        // Initialize
        initializePoseLandmarker();

        // Cleanup
        return () => {
            if (videoRef.current && videoRef.current.srcObject) {
                videoRef.current.srcObject.getTracks().forEach(track => track.stop());
            }
            webcamRunningRef.current = false;
        };
    }, [onPoseLandmarksReceived]);

    // Add a useEffect to check video stream health
    useEffect(() => {
        // Check video stream health periodically
        const videoHealthCheck = setInterval(() => {
            if (videoRef.current && webcamRunningRef.current) {
                if (videoRef.current.readyState < 2 || videoRef.current.paused) {
                    console.warn("Video stream appears to be inactive, restarting...");
                    
                    // Try to restart the stream
                    if (videoRef.current.srcObject) {
                        const tracks = videoRef.current.srcObject.getTracks();
                        if (tracks.length === 0 || tracks[0].readyState !== "live") {
                            console.log("Restarting camera due to inactive stream");
                            webcamRunningRef.current = false;
                            setTimeout(() => {
                                enableCam();
                            }, 500);
                        } else {
                            // Stream exists but video is paused, try to play
                            videoRef.current.play().catch(e => {
                                console.warn("Could not restart playback:", e);
                            });
                        }
                    }
                }
            }
        }, 3000); // Check every 3 seconds
        
        return () => {
            clearInterval(videoHealthCheck);
        };
    }, []);

    return (
        <div 
            className="detection-container"
            style={{
                position: 'relative',
                width: '640px',
                height: '480px',
                overflow: 'hidden',
                border: '2px solid #333',
                margin: '0 auto',
                backgroundColor: '#000', // Add background color to container
            }}
        >
            <video
                ref={videoRef}
                style={{
                    position: 'absolute',
                    top: '50%',
                    left: '50%',
                    transform: 'translate(-50%, -50%) scaleX(-1)',
                    maxWidth: '100%',
                    maxHeight: '100%',
                    backgroundColor: 'transparent',
                }}
                muted
                playsInline
            />

            <canvas
                ref={canvasRef}
                style={{
                    position: 'absolute',
                    top: '50%',
                    left: '50%',
                    transform: 'translate(-50%, -50%) scaleX(-1)',
                    maxWidth: '100%',
                    maxHeight: '100%',
                    zIndex: 999,
                    backgroundColor: 'transparent',
                }}
            />

            {/* Status indicator */}
            <div
                style={{
                    position: 'absolute',
                    bottom: 10,
                    left: 10,
                    background: 'rgba(0,0,0,0.7)',
                    padding: '5px 10px',
                    borderRadius: '5px',
                    color: 'white',
                    fontSize: '12px',
                    zIndex: 1000
                }}
            >
                {landmarksDetected ?
                    "✅ Pose Detected" :
                    "⏳ Waiting for pose..."}
            </div>
        </div>
    );
};

export default Detection;
