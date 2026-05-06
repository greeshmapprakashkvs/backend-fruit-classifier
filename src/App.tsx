import React, { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Apple, Camera, Database, Upload, Wand2, Info, ChevronRight, RefreshCcw, ScanLine } from 'lucide-react';
import { GoogleGenAI } from "@google/genai";

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

type Mode = 'features' | 'camera';

export default function App() {
  const [mode, setMode] = useState<Mode>('features');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{ prediction: string; confidence: number } | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Feature Input State
  const [features, setFeatures] = useState({
    mass: 150,
    width: 7.5,
    height: 7.5,
    color_score: 0.6
  });

  // Camera State
  const videoRef = useRef<HTMLVideoElement>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(features)
      });
      if (!response.ok) throw new Error('Prediction failed');
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError('Could not connect to the ML backend.');
    } finally {
      setLoading(false);
    }
  };

  const startCamera = async () => {
    setMode('camera');
    setResult(null);
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
      setStream(mediaStream);
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
    } catch (err) {
      setError('Camera access denied or not available.');
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
  };

  const captureAndIdentify = async () => {
    if (!videoRef.current) return;
    setLoading(true);
    setError(null);
    try {
      const canvas = document.createElement('canvas');
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx?.drawImage(videoRef.current, 0, 0);
      const base64 = canvas.toDataURL('image/jpeg').split(',')[1];

      const response = await ai.models.generateContent({
        model: "gemini-3-flash-preview",
        contents: [
          {
            parts: [
              { text: "Identify the fruit in this image. Return JSON with 'prediction' (string) and 'confidence' (number between 0 and 1)." },
              { inlineData: { data: base64, mimeType: "image/jpeg" } }
            ]
          }
        ],
        config: { responseMimeType: "application/json" }
      });

      const text = response.text || "";
      const data = JSON.parse(text);
      setResult(data);
    } catch (err) {
      setError('AI Identification failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#FDFCFB] text-[#1D1D1F]">
      {/* Header */}
      <header className="border-b border-gray-100 bg-white/80 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-5xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-orange-500 rounded-xl flex items-center justify-center text-white shadow-lg shadow-orange-200">
              <Apple size={24} />
            </div>
            <h1 className="text-xl font-semibold tracking-tight">Fruitify AI</h1>
          </div>
          <nav className="flex gap-1 p-1 bg-gray-100 rounded-lg">
            <button 
              onClick={() => { stopCamera(); setMode('features'); setResult(null); }}
              className={`px-4 py-1.5 rounded-md text-sm font-medium transition-all ${mode === 'features' ? 'bg-white shadow-sm text-orange-600' : 'text-gray-500 hover:text-gray-700'}`}
            >
              ML Features
            </button>
            <button 
              onClick={startCamera}
              className={`px-4 py-1.5 rounded-md text-sm font-medium transition-all ${mode === 'camera' ? 'bg-white shadow-sm text-blue-600' : 'text-gray-500 hover:text-gray-700'}`}
            >
              AI Vision
            </button>
          </nav>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-6 py-12">
        <div className="grid lg:grid-cols-2 gap-12 items-start">
          
          {/* Left Column: UI Controls */}
          <section className="space-y-8">
            <div className="space-y-4">
              <h2 className="text-4xl font-bold tracking-tighter sm:text-5xl">
                Identify any fruit in <span className={mode === 'features' ? 'text-orange-500' : 'text-blue-500'}>seconds.</span>
              </h2>
              <p className="text-gray-500 text-lg leading-relaxed">
                {mode === 'features' 
                  ? "Input physical features like mass and color score to use our scikit-learn model trained on classic fruit datasets."
                  : "Use Gemini AI Vision to instantly recognize fruits from your camera or uploaded images."}
              </p>
            </div>

            <div className="bg-white p-8 rounded-3xl border border-gray-100 shadow-xl shadow-gray-100/50">
              {mode === 'features' ? (
                <div className="space-y-6">
                  <div className="grid grid-cols-2 gap-4">
                    <FeatureControl 
                      label="Mass (g)" 
                      value={features.mass} 
                      min={50} max={300} step={1}
                      onChange={(v) => setFeatures({...features, mass: v})} 
                    />
                    <FeatureControl 
                      label="Width (cm)" 
                      value={features.width} 
                      min={5} max={10} step={0.1}
                      onChange={(v) => setFeatures({...features, width: v})} 
                    />
                    <FeatureControl 
                      label="Height (cm)" 
                      value={features.height} 
                      min={4} max={12} step={0.1}
                      onChange={(v) => setFeatures({...features, height: v})} 
                    />
                    <FeatureControl 
                      label="Color Score" 
                      value={features.color_score} 
                      min={0} max={1} step={0.01}
                      onChange={(v) => setFeatures({...features, color_score: v})} 
                    />
                  </div>
                  <button 
                    disabled={loading}
                    onClick={handlePredict}
                    className="w-full py-4 bg-orange-500 hover:bg-orange-600 disabled:opacity-50 text-white rounded-2xl font-semibold flex items-center justify-center gap-2 transition-all shadow-lg shadow-orange-100"
                  >
                    {loading ? <RefreshCcw className="animate-spin" /> : <Database size={20} />}
                    Identify Fruit
                  </button>
                </div>
              ) : (
                <div className="space-y-6">
                  <div className="relative aspect-video bg-gray-900 rounded-2xl overflow-hidden shadow-inner group">
                    <video 
                      ref={videoRef} 
                      autoPlay 
                      playsInline 
                      className="w-full h-full object-cover"
                    />
                    <div className="absolute inset-0 border-2 border-dashed border-white/30 rounded-2xl m-4 pointer-events-none group-hover:border-blue-400/50 transition-colors" />
                    <div className="absolute top-4 right-4 flex gap-2">
                       <span className="flex items-center gap-1.5 px-3 py-1 bg-black/40 backdrop-blur-md rounded-full text-[10px] text-white font-mono uppercase tracking-widest">
                         <div className="w-1.5 h-1.5 bg-red-500 rounded-full animate-pulse" />
                         Live
                       </span>
                    </div>
                  </div>
                  <button 
                    disabled={loading}
                    onClick={captureAndIdentify}
                    className="w-full py-4 bg-blue-500 hover:bg-blue-600 disabled:opacity-50 text-white rounded-2xl font-semibold flex items-center justify-center gap-2 transition-all shadow-lg shadow-blue-100"
                  >
                    {loading ? <RefreshCcw className="animate-spin" /> : <Camera size={20} />}
                    Capture & Identify
                  </button>
                </div>
              )}
            </div>
            
            {error && (
              <motion.div 
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="p-4 bg-red-50 border border-red-100 text-red-600 rounded-2xl flex items-center gap-3 text-sm"
              >
                <Info size={16} />
                {error}
              </motion.div>
            )}
          </section>

          {/* Right Column: Results & Insights */}
          <section className="relative lg:sticky lg:top-28">
            <AnimatePresence mode="wait">
              {result ? (
                <motion.div
                  key="result"
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0 }}
                  className="bg-white rounded-[40px] p-10 border border-gray-100 shadow-2xl space-y-8"
                >
                  <div className="flex justify-between items-start">
                    <div>
                      <p className="text-gray-400 font-mono text-xs uppercase tracking-widest mb-1">Result Found</p>
                      <h3 className="text-5xl font-black">{result.prediction}</h3>
                    </div>
                    <div className="w-16 h-16 bg-gray-50 rounded-full flex items-center justify-center border border-gray-100">
                       <ScanLine className="text-gray-300" />
                    </div>
                  </div>

                  <div className="space-y-2">
                    <div className="flex justify-between text-sm font-semibold">
                      <span>Confidence</span>
                      <span className="text-blue-500">{(result.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div className="h-3 w-full bg-gray-100 rounded-full overflow-hidden">
                      <motion.div 
                        initial={{ width: 0 }}
                        animate={{ width: `${result.confidence * 100}%` }}
                        className={`h-full ${mode === 'features' ? 'bg-orange-500' : 'bg-blue-500'}`}
                      />
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <InsightCard 
                      icon={<Info size={18} />} 
                      title="Classification" 
                      value="Fruit" 
                    />
                    <InsightCard 
                      icon={<Wand2 size={18} />} 
                      title="Model Used" 
                      value={mode === 'features' ? "scikit-learn KNN" : "Gemini 3 Flash"} 
                    />
                  </div>

                  <button 
                    onClick={() => setResult(null)}
                    className="w-full py-4 border border-gray-100 hover:bg-gray-50 rounded-2xl font-medium text-gray-500 transition-colors flex items-center justify-center gap-2"
                  >
                    Reset Analysis
                  </button>
                </motion.div>
              ) : (
                <motion.div
                  key="placeholder"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="h-[400px] bg-gray-50/50 rounded-[40px] border-2 border-dashed border-gray-100 flex flex-col items-center justify-center text-center p-8"
                >
                  <div className="w-20 h-20 bg-white rounded-[24px] shadow-sm flex items-center justify-center mb-6">
                    <Upload className="text-gray-300" />
                  </div>
                  <p className="text-xl font-medium text-gray-400">Perform an analysis to see insights here.</p>
                  <p className="text-gray-400 text-sm mt-2 max-w-[200px]">Data is processed in real-time using selected ML model.</p>
                </motion.div>
              )}
            </AnimatePresence>
          </section>
        </div>
      </main>
    </div>
  );
}

function FeatureControl({ label, value, onChange, min, max, step }: { label: string, value: number, onChange: (v: number) => void, min: number, max: number, step: number }) {
  return (
    <div className="p-4 bg-gray-50 rounded-2xl space-y-3">
      <div className="flex justify-between items-center text-xs font-bold uppercase tracking-wider text-gray-400">
        <span>{label}</span>
        <span className="text-orange-600 font-mono">{value}</span>
      </div>
      <input 
        type="range" 
        min={min} max={max} step={step} 
        value={value} 
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full accent-orange-500 h-1"
      />
    </div>
  );
}

function InsightCard({ icon, title, value }: { icon: React.ReactNode, title: string, value: string }) {
  return (
    <div className="p-5 bg-gray-50 rounded-3xl border border-gray-100 flex items-start gap-4">
      <div className="p-2.5 bg-white rounded-xl shadow-sm text-gray-400">
        {icon}
      </div>
      <div>
        <p className="text-[10px] font-bold uppercase tracking-wider text-gray-400">{title}</p>
        <p className="text-sm font-semibold">{value}</p>
      </div>
    </div>
  );
}
