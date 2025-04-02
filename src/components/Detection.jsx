import React, { useEffect, useRef, useState } from 'react';
import { PoseLandmarker, FilesetResolver, DrawingUtils } from "@mediapipe/tasks-vision";

const Detection = ({ onPoseLandmarksReceived }) => {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const [landmarksDetected, setLandmarksDetected] = useState(false);
    const webcamRunningRef = useRef(false);
    const frameCountRef = useRef(0);
    const errorCountRef = useRef(0);

    useEffect(() => {
        if (!videoRef.current || !canvasRef.current) return;

        const canvasCtx = canvasRef.current.getContext('2d');
        const drawingUtils = new DrawingUtils(canvasCtx);

        let poseLandmarker;

        const drawResults = (results) => {
            if (!canvasRef.current) return;
            
            // Clear with transparent background
            canvasCtx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

            // Only try to draw landmarks if they exist
            if (results && results.landmarks && results.landmarks.length > 0) {
                try {
                    for (const landmarks of results.landmarks) {
                        // Wrap drawing operations in try-catch to prevent crashes
                        try {
                            drawingUtils.drawConnectors(
                                landmarks,
                                PoseLandmarker.POSE_CONNECTIONS
                            );
                            
                            drawingUtils.drawLandmarks(
                                landmarks
                            );
                        } catch (error) {
                            // Don't throw error to UI - just log it
                            console.warn("Drawing error:", error.message);
                        }
                    }
                } catch (error) {
                    console.warn("Error in draw results:", error.message);
                    // Continue without breaking the rendering loop
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
                    minPoseDetectionConfidence: 0.5,
                    minPosePresenceConfidence: 0.5,
                    minTrackingConfidence: 0.5
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
                        minPoseDetectionConfidence: 0.5,
                        minPosePresenceConfidence: 0.5,
                        minTrackingConfidence: 0.5
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
                        width: { ideal: 640, min: 320 },
                        height: { ideal: 480, min: 240 },
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
                                            
                                            // console.log(`Video dimensions: ${videoWidth}x${videoHeight}`);
                                            
                                            if (videoWidth > 0 && videoHeight > 0 && canvasRef.current) {
                                                canvasRef.current.width = videoWidth;
                                                canvasRef.current.height = videoHeight;
                                                
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
                // Use stronger throttling for pose detection to reduce CPU/GPU load
                // Process every 3rd frame (adjustable based on performance)
                frameCountRef.current = (frameCountRef.current + 1) % 3;
                
                if (frameCountRef.current === 0) {
                    let results;
                    
                    try {
                        results = poseLandmarker.detectForVideo(videoRef.current, startTimeMs);
                        
                        // Always draw results to maintain visual feedback
                        drawResults(results);
                        
                        // Only update landmarks and state if we have valid landmarks
                        if (results.landmarks && results.landmarks.length > 0) {
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
                } else {
                    // On frames we skip detection for, still keep video visible
                    // Just draw the previous results to maintain visual continuity
                    // This gives smooth video while reducing computation load
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

    return (
        <div className="detection-container">
            <div
                style={{
                    position: 'relative',
                    width: '640px',
                    height: '480px',
                    overflow: 'hidden',
                    border: '2px solid #333',
                    margin: '0 auto'
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
        </div>
    );
};

export default Detection;
