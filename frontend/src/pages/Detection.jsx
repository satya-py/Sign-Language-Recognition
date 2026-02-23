import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, Camera, Play, Volume2, StopCircle, CheckCircle, AlertTriangle } from 'lucide-react';
import { clsx } from 'clsx';

// Text to Speech Helper
const speak = (text) => {
    if ('speechSynthesis' in window) {
        const utterance = new SpeechSynthesisUtterance(text);
        window.speechSynthesis.speak(utterance);
    }
};

const Detection = () => {
    const [mode, setMode] = useState('upload'); // 'upload' or 'camera'
    const [file, setFile] = useState(null);
    const [videoPreview, setVideoPreview] = useState(null);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [isDetecting, setIsDetecting] = useState(false);
    const [predictions, setPredictions] = useState([]); // History of predictions
    const [liveSessionId, setLiveSessionId] = useState(null);

    // Refs
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const intervalRef = useRef(null);

    // Camera handling
    const startCamera = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            if (videoRef.current) {
                videoRef.current.srcObject = stream;
            }
        } catch (err) {
            console.error("Camera error:", err);
            alert("Could not access camera");
        }
    };

    const stopCamera = () => {
        if (videoRef.current && videoRef.current.srcObject) {
            const tracks = videoRef.current.srcObject.getTracks();
            tracks.forEach(track => track.stop());
            videoRef.current.srcObject = null;
        }
        stopLiveDetection();
    };

    // Live Detection Logic
    const startLiveDetection = async () => {
        try {
            const res = await axios.post('http://127.0.0.1:5000/api/session/start');
            const sessionId = res.data.session_id;

            setLiveSessionId(sessionId);
            setIsDetecting(true);
            setPredictions([]);
            setResult(null);

            intervalRef.current = setInterval(() => {
                captureAndSendFrame(sessionId);
            }, 200);

        } catch (err) {
            console.error("Session start error", err);
        }
    };


    const stopLiveDetection = () => {
        setIsDetecting(false);
        if (intervalRef.current) clearInterval(intervalRef.current);
    };

    const captureAndSendFrame = async (sessionId) => {
        if (!videoRef.current || !canvasRef.current || !sessionId) return;

        if (videoRef.current.videoWidth === 0) return;

        const ctx = canvasRef.current.getContext('2d');
        canvasRef.current.width = videoRef.current.videoWidth;
        canvasRef.current.height = videoRef.current.videoHeight;
        ctx.drawImage(videoRef.current, 0, 0);

        const imageData = canvasRef.current.toDataURL('image/jpeg', 0.8);

        try {
            const res = await axios.post(
                'http://127.0.0.1:5000/api/predict_frame',
                {
                    image: imageData,
                    session_id: sessionId
                }
            );

            if (res.data.prediction && res.data.confidence > 0.6) {
                const newPred = { label: res.data.prediction, confidence: res.data.confidence };
                setResult(newPred);
                setPredictions(prev => [...prev, newPred]);
            }
        } catch (err) {
            console.error("Frame prediction failed", err);
        }
    };


    // Upload Logic
    const handleFileChange = (e) => {
        const selectedinfo = e.target.files[0];
        if (selectedinfo) {
            setFile(selectedinfo);
            setVideoPreview(URL.createObjectURL(selectedinfo));
            setResult(null);
            setPredictions([]);
        }
    };

    const handleUploadDetection = async () => {
        if (!file) return;
        setLoading(true);
        setResult(null);
        setPredictions([]);

        const formData = new FormData();
        formData.append('file', file);

        try {
            const res = await axios.post('http://127.0.0.1:5000/api/predict_video', formData);
            if (res.data.result) {
                setResult({ label: res.data.result, confidence: res.data.confidence });
                if (res.data.all_predictions) {
                    setPredictions(res.data.all_predictions);
                } else {
                    setPredictions([{ label: res.data.result, confidence: res.data.confidence }]);
                }
            } else {
                alert("No signs detected with high confidence.");
            }
        } catch (err) {
            console.error(err);
            alert("Detection failed.");
        } finally {
            setLoading(false);
        }
    };

    // Cleanup
    useEffect(() => {
        return () => {
            stopCamera();
            if (intervalRef.current) clearInterval(intervalRef.current);
        };
    }, []);

    // Mode switch effect
    useEffect(() => {
        if (mode === 'camera') {
            startCamera();
        } else {
            stopCamera();
        }
        setResult(null);
        setPredictions([]);
    }, [mode]);


    return (
        <div className="min-h-screen py-10 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 h-full">

                {/* LEFT COLUMN: CONTROLS */}
                <div className="space-y-6">
                    <div className="bg-white/5 p-6 rounded-2xl border border-white/10 backdrop-blur-sm">
                        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
                            <Upload className="text-blue-400" /> Input Source
                        </h2>

                        <div className="flex gap-4 mb-6">
                            <button
                                onClick={() => setMode('upload')}
                                className={clsx("flex-1 py-3 px-4 rounded-xl flex items-center justify-center gap-2 transition-all",
                                    mode === 'upload' ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/25' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                                )}
                            >
                                <Upload size={20} /> Upload Video
                            </button>
                            <button
                                onClick={() => setMode('camera')}
                                className={clsx("flex-1 py-3 px-4 rounded-xl flex items-center justify-center gap-2 transition-all",
                                    mode === 'camera' ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/25' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                                )}
                            >
                                <Camera size={20} /> Use Camera
                            </button>
                        </div>

                        <div className="h-64 border-2 border-dashed border-gray-700 rounded-2xl flex items-center justify-center bg-black/20 overflow-hidden relative">
                            {mode === 'upload' ? (
                                <div className="text-center w-full h-full relative">
                                    {videoPreview ? (
                                        <video src={videoPreview} controls className="w-full h-full object-contain" />
                                    ) : (
                                        <label className="cursor-pointer w-full h-full flex flex-col items-center justify-center hover:bg-white/5 transition-colors">
                                            <Upload size={48} className="mb-4 text-gray-500" />
                                            <span className="text-gray-400 font-medium">Click to upload video</span>
                                            <input type="file" accept="video/*" onChange={handleFileChange} className="hidden" />
                                        </label>
                                    )}
                                </div>
                            ) : (
                                <div className="text-center text-gray-400">
                                    <Camera size={48} className="mx-auto mb-4" />
                                    <p>Camera is active on the right panel</p>
                                </div>
                            )}
                        </div>
                    </div>

                    <div className="flex gap-4">
                        {mode === 'upload' ? (
                            <button
                                onClick={handleUploadDetection}
                                disabled={!file || loading}
                                className={clsx("w-full py-4 rounded-xl font-bold text-lg flex items-center justify-center gap-2 transition-all shadow-lg",
                                    loading
                                        ? "bg-gray-700 text-gray-400 cursor-not-allowed"
                                        : "bg-green-600 hover:bg-green-500 text-white shadow-green-500/25"
                                )}
                            >
                                {loading ? 'Detecting...' : 'Start Detection'}
                                {!loading && <Play size={20} fill="currentColor" />}
                            </button>
                        ) : (
                            <button
                                onClick={isDetecting ? stopLiveDetection : startLiveDetection}
                                className={clsx("w-full py-4 rounded-xl font-bold text-lg flex items-center justify-center gap-2 transition-all shadow-lg",
                                    isDetecting
                                        ? "bg-red-600 hover:bg-red-500 text-white shadow-red-500/25"
                                        : "bg-green-600 hover:bg-green-500 text-white shadow-green-500/25"
                                )}
                            >
                                {isDetecting ? 'Stop Detection' : 'Start Real-time Detection'}
                                {isDetecting ? <StopCircle size={20} /> : <Play size={20} fill="currentColor" />}
                            </button>
                        )}
                    </div>
                </div>

                {/* RIGHT COLUMN: ANALYSIS & RESULTS */}
                <div className="space-y-6">
                    {/* Viewport */}
                    <div className="bg-black/40 p-1 rounded-2xl border border-white/10 shadow-2xl relative overflow-hidden aspect-video flex items-center justify-center">
                        <div className="absolute top-4 left-4 bg-black/60 px-3 py-1 rounded-full text-xs font-mono text-green-400 flex items-center gap-2">
                            <div className={`w-2 h-2 rounded-full ${isDetecting ? 'bg-red-500 animate-pulse' : 'bg-gray-500'}`} />
                            ANALYSIS VIEWPORT
                        </div>

                        {mode === 'camera' ? (
                            <>
                                <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover transform -scale-x-100" />
                                <canvas ref={canvasRef} className="hidden" />
                            </>
                        ) : (
                            <div className="text-gray-500 flex flex-col items-center">
                                <AlertTriangle size={48} className="mb-2" />
                                <span>Waiting for input...</span>
                            </div>
                        )}

                        {/* Overlay Results */}
                        <AnimatePresence>
                            {result && (
                                <motion.div
                                    initial={{ y: 50, opacity: 0 }}
                                    animate={{ y: 0, opacity: 1 }}
                                    exit={{ y: 50, opacity: 0 }}
                                    className="absolute bottom-0 w-full bg-black/80 backdrop-blur-md p-4 border-t border-white/10"
                                >
                                    <div className="flex justify-between items-end">
                                        <div>
                                            <p className="text-gray-400 text-sm mb-1">Top Prediction</p>
                                            <h3 className="text-3xl font-black text-white tracking-wide uppercase">{result.label}</h3>
                                        </div>
                                        <div className="text-right">
                                            <div className="text-2xl font-bold text-green-400">{(result.confidence * 100).toFixed(1)}%</div>
                                            <p className="text-xs text-gray-500">Confidence Score</p>
                                        </div>
                                    </div>
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </div>

                    {/* Results Box */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        {/* Summary Box */}
                        <div className="bentogrid-item">
                            <h3 className="text-lg font-bold mb-4 text-gray-300 border-b border-gray-700 pb-2">Detection Summary</h3>
                            {result ? (
                                <div className="space-y-4">
                                    <div className="flex justify-between items-center">
                                        <span className="text-gray-400">Predicted Sign:</span>
                                        <span className="font-bold text-lg text-blue-400 capitalize">{result.label}</span>
                                    </div>
                                    <div className="w-full bg-gray-700 rounded-full h-2.5">
                                        <div className="bg-blue-500 h-2.5 rounded-full" style={{ width: `${result.confidence * 100}%` }}></div>
                                    </div>
                                    <div className="flex justify-between text-xs text-gray-500">
                                        <span>0%</span>
                                        <span>Confidence</span>
                                        <span>100%</span>
                                    </div>
                                </div>
                            ) : (
                                <p className="text-gray-500 text-sm italic">No active detection results.</p>
                            )}
                        </div>

                        {/* Speech Box */}
                        <div className="bentogrid-item relative">
                            <h3 className="text-lg font-bold mb-4 text-gray-300 border-b border-gray-700 pb-2">Speech Output</h3>

                            <div className="h-32 overflow-y-auto space-y-2 pr-2 custom-scrollbar">
                                {predictions.length > 0 ? (
                                    predictions.slice().reverse().map((pred, idx) => (
                                        <div key={idx} className="flex items-center justify-between p-2 bg-white/5 rounded-lg">
                                            <span className="font-medium text-gray-200 capitalize">{pred.label}</span>
                                            <button
                                                onClick={() => speak(pred.label)}
                                                className="p-2 hover:bg-white/10 rounded-full text-blue-400 transition-colors"
                                                title="Speak"
                                            >
                                                <Volume2 size={16} />
                                            </button>
                                        </div>
                                    ))
                                ) : (
                                    <div className="h-full flex items-center justify-center text-gray-500 text-sm">
                                        Waiting for predictions...
                                    </div>
                                )}
                            </div>

                            {result && (
                                <button
                                    onClick={() => speak(result.label)}
                                    className="absolute top-6 right-6 p-2 bg-blue-600 rounded-full text-white shadow-lg hover:scale-110 transition-transform"
                                >
                                    <Volume2 size={20} />
                                </button>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Detection;
