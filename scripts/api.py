#!/usr/bin/env python3
"""
FastAPI web application for Diabetic Retinopathy Detection
Access at: http://localhost:8000
"""

import os
import sys
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import torch
import uvicorn
import logging
from typing import List
import json
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessor import ImagePreprocessor
from src.models.architecture import RetinopathyDetectionModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Diabetic Retinopathy Detection",
    description="Real-time ML pipeline for eye scan analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/checkpoints/best_model.pt"
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

SEVERITY_LEVELS = {
    0: "No Diabetic Retinopathy",
    1: "Mild Diabetic Retinopathy",
    2: "Moderate Diabetic Retinopathy",
    3: "Severe Diabetic Retinopathy",
    4: "Proliferative Diabetic Retinopathy"
}

COLORS = {
    0: "#28a745",  # Green
    1: "#17a2b8",  # Blue
    2: "#ffc107",  # Yellow
    3: "#fd7e14",  # Orange
    4: "#dc3545"   # Red
}


class RetinopathyPredictor:
    """Model predictor"""
    def __init__(self, model_path: str):
        self.device = DEVICE
        self.model = RetinopathyDetectionModel(
            model_name="efficientnet_b4",
            num_classes=5,
            pretrained=False
        ).to(self.device)
        
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"✓ Model loaded from {model_path}")
        else:
            logger.warning(f"Model not found at {model_path}. Using untrained model.")
        
        self.model.eval()
        self.preprocessor = ImagePreprocessor(image_size=512, normalize=True)
    
    def predict(self, image_path: str) -> dict:
        """Make prediction"""
        try:
            image = self.preprocessor.preprocess(image_path)
            image_tensor = torch.from_numpy(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(image_tensor)
                probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
                predicted_class = probabilities.argmax()
                confidence = probabilities[predicted_class]
            
            return {
                'grade': int(predicted_class),
                'severity': SEVERITY_LEVELS[int(predicted_class)],
                'confidence': float(confidence),
                'color': COLORS[int(predicted_class)],
                'probabilities': {
                    i: {
                        'label': SEVERITY_LEVELS[i],
                        'probability': float(prob),
                        'percentage': f"{float(prob)*100:.1f}%"
                    }
                    for i, prob in enumerate(probabilities)
                }
            }
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise


# Initialize predictor
try:
    predictor = RetinopathyPredictor(MODEL_PATH)
    model_ready = True
except Exception as e:
    logger.warning(f"Model initialization failed: {str(e)}. API will work in demo mode.")
    predictor = None
    model_ready = False


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve main HTML page"""
    return get_html_page()


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": DEVICE,
        "model_ready": model_ready,
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Single image prediction"""
    if not model_ready:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Make prediction
        result = predictor.predict(str(file_path))
        result['filename'] = file.filename
        
        return result
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/batch-predict")
async def batch_predict(files: List[UploadFile] = File(...)):
    """Batch predictions"""
    if not model_ready:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    for file in files:
        try:
            file_path = UPLOAD_DIR / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            result = predictor.predict(str(file_path))
            result['filename'] = file.filename
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {str(e)}")
            results.append({
                'filename': file.filename,
                'error': str(e)
            })
    
    return {'results': results, 'total': len(files), 'processed': len([r for r in results if 'error' not in r])}


@app.get("/model-info")
async def model_info():
    """Get model information"""
    return {
        "model_name": "EfficientNet B4",
        "num_classes": 5,
        "classes": SEVERITY_LEVELS,
        "input_size": 512,
        "model_ready": model_ready,
        "device": DEVICE,
        "framework": "PyTorch"
    }


@app.get("/severity-info")
async def severity_info():
    """Get severity level information"""
    return {
        "levels": [
            {
                "grade": i,
                "name": SEVERITY_LEVELS[i],
                "color": COLORS[i],
                "description": get_severity_description(i)
            }
            for i in range(5)
        ]
    }


def get_severity_description(grade: int) -> str:
    """Get description for each severity level"""
    descriptions = {
        0: "No signs of diabetic retinopathy detected",
        1: "Microaneurysms may be present, but no other signs of retinopathy",
        2: "Microaneurysms, hard exudates, cotton wool spots, retinal hemorrhages",
        3: "Severe hemorrhages, venous beading, prominent cotton wool spots, intraretinal microvascular abnormalities",
        4: "Vitreous hemorrhage or retinal detachment"
    }
    return descriptions.get(grade, "Unknown")


def get_html_page() -> str:
    """Generate HTML page"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Diabetic Retinopathy Detection</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"></script>
        <style>
            .severity-color {
                transition: all 0.3s ease;
            }
            .loading {
                animation: spin 1s linear infinite;
            }
            @keyframes spin {
                from { transform: rotate(0deg); }
                to { transform: rotate(360deg); }
            }
            .pulse {
                animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
            }
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: .5; }
            }
        </style>
    </head>
    <body class="bg-gradient-to-br from-blue-50 to-indigo-100 min-h-screen">
        <!-- Navigation -->
        <nav class="bg-white shadow-lg">
            <div class="max-w-7xl mx-auto px-4 py-4">
                <div class="flex justify-between items-center">
                    <h1 class="text-2xl font-bold text-indigo-600">🔬 Retinopathy Detection</h1>
                    <div class="flex gap-4">
                        <span id="status" class="text-sm text-gray-600">Status: Loading...</span>
                        <span id="device" class="text-sm text-gray-600">Device: -</span>
                    </div>
                </div>
            </div>
        </nav>

        <!-- Main Container -->
        <div class="max-w-7xl mx-auto px-4 py-8">
            <!-- Tabs -->
            <div class="flex gap-2 mb-8">
                <button onclick="switchTab('upload')" class="tab-btn active px-6 py-2 rounded-lg font-semibold transition" id="tab-upload">
                    📤 Upload & Predict
                </button>
                <button onclick="switchTab('batch')" class="tab-btn px-6 py-2 rounded-lg font-semibold transition" id="tab-batch">
                    📁 Batch Processing
                </button>
                <button onclick="switchTab('info')" class="tab-btn px-6 py-2 rounded-lg font-semibold transition" id="tab-info">
                    ℹ️ Information
                </button>
            </div>

            <!-- Upload Tab -->
            <div id="upload-tab" class="tab-content">
                <div class="grid md:grid-cols-2 gap-8">
                    <!-- Upload Section -->
                    <div class="bg-white rounded-lg shadow-lg p-8">
                        <h2 class="text-2xl font-bold text-gray-800 mb-6">Upload Eye Scan</h2>
                        
                        <div class="mb-6">
                            <div id="dropzone" class="border-2 border-dashed border-indigo-300 rounded-lg p-8 text-center cursor-pointer hover:border-indigo-500 hover:bg-indigo-50 transition">
                                <div class="text-4xl mb-4">📸</div>
                                <p class="text-gray-600 font-semibold">Drag & drop image here</p>
                                <p class="text-sm text-gray-500 mt-2">or click to browse</p>
                                <input type="file" id="imageInput" accept="image/*" style="display:none;" onchange="handleFileUpload(event)">
                            </div>
                        </div>

                        <img id="preview" style="display:none;" class="w-full rounded-lg mb-4 max-h-80 object-cover">
                        
                        <button onclick="uploadImage()" class="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-3 rounded-lg transition" id="uploadBtn">
                            🚀 Analyze Image
                        </button>
                    </div>

                    <!-- Results Section -->
                    <div class="bg-white rounded-lg shadow-lg p-8">
                        <h2 class="text-2xl font-bold text-gray-800 mb-6">Analysis Results</h2>
                        
                        <div id="results" style="display:none;">
                            <!-- Severity Badge -->
                            <div id="severityBadge" class="rounded-lg p-6 mb-6 text-white text-center">
                                <div id="severityGrade" class="text-5xl font-bold mb-2">-</div>
                                <div id="severityName" class="text-xl font-semibold">-</div>
                            </div>

                            <!-- Confidence -->
                            <div class="mb-6">
                                <p class="text-gray-700 font-semibold mb-2">Confidence Score</p>
                                <div class="w-full bg-gray-200 rounded-full h-3">
                                    <div id="confidenceBar" class="bg-green-500 h-3 rounded-full transition-all"></div>
                                </div>
                                <p id="confidenceText" class="text-sm text-gray-600 mt-2">-</p>
                            </div>

                            <!-- Probabilities Chart -->
                            <div class="mb-6">
                                <p class="text-gray-700 font-semibold mb-4">Class Probabilities</p>
                                <div id="probabilities" class="space-y-3"></div>
                            </div>
                        </div>

                        <div id="loading" style="display:none;" class="text-center py-12">
                            <div class="text-4xl loading mb-4">⚙️</div>
                            <p class="text-gray-600 font-semibold">Analyzing image...</p>
                        </div>

                        <div id="noResults" class="text-center py-12 text-gray-500">
                            <div class="text-6xl mb-4">👆</div>
                            <p>Upload an image to see results</p>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Batch Tab -->
            <div id="batch-tab" class="tab-content" style="display:none;">
                <div class="bg-white rounded-lg shadow-lg p-8">
                    <h2 class="text-2xl font-bold text-gray-800 mb-6">Batch Processing</h2>
                    
                    <div class="mb-6">
                        <div id="batchDropzone" class="border-2 border-dashed border-indigo-300 rounded-lg p-8 text-center cursor-pointer hover:border-indigo-500 hover:bg-indigo-50 transition">
                            <div class="text-4xl mb-4">📁</div>
                            <p class="text-gray-600 font-semibold">Drag & drop multiple images</p>
                            <p class="text-sm text-gray-500 mt-2">or click to browse</p>
                            <input type="file" id="batchInput" accept="image/*" multiple style="display:none;" onchange="handleBatchUpload(event)">
                        </div>
                    </div>

                    <div id="fileList" class="mb-6 space-y-2"></div>
                    
                    <button onclick="uploadBatch()" class="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-3 rounded-lg transition" id="batchBtn">
                        🚀 Process All Images
                    </button>

                    <div id="batchResults" style="display:none;" class="mt-8">
                        <h3 class="text-xl font-bold mb-4">Results Summary</h3>
                        <div id="resultsList" class="space-y-2"></div>
                    </div>
                </div>
            </div>

            <!-- Info Tab -->
            <div id="info-tab" class="tab-content" style="display:none;">
                <div class="grid md:grid-cols-2 gap-8">
                    <!-- Model Info -->
                    <div class="bg-white rounded-lg shadow-lg p-8">
                        <h2 class="text-2xl font-bold text-gray-800 mb-6">Model Information</h2>
                        <div id="modelInfo" class="space-y-4 text-gray-700">
                            <p>Loading...</p>
                        </div>
                    </div>

                    <!-- Severity Guide -->
                    <div class="bg-white rounded-lg shadow-lg p-8">
                        <h2 class="text-2xl font-bold text-gray-800 mb-6">Severity Levels</h2>
                        <div id="severityGuide" class="space-y-4"></div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            const API_URL = window.location.origin;

            // Setup drag and drop
            const setupDragDrop = (elementId, inputId, callback) => {
                const element = document.getElementById(elementId);
                const input = document.getElementById(inputId);
                
                element.addEventListener('click', () => input.click());
                
                element.addEventListener('dragover', (e) => {
                    e.preventDefault();
                    element.classList.add('border-indigo-500', 'bg-indigo-50');
                });
                
                element.addEventListener('dragleave', () => {
                    element.classList.remove('border-indigo-500', 'bg-indigo-50');
                });
                
                element.addEventListener('drop', (e) => {
                    e.preventDefault();
                    element.classList.remove('border-indigo-500', 'bg-indigo-50');
                    callback(e.dataTransfer.files);
                });
            };

            setupDragDrop('dropzone', 'imageInput', (files) => {
                document.getElementById('imageInput').files = files;
                handleFileUpload({target: {files: files}});
            });

            setupDragDrop('batchDropzone', 'batchInput', (files) => {
                document.getElementById('batchInput').files = files;
                handleBatchUpload({target: {files: files}});
            });

            function handleFileUpload(e) {
                const file = e.target.files[0];
                if (!file) return;
                
                const reader = new FileReader();
                reader.onload = (event) => {
                    document.getElementById('preview').src = event.target.result;
                    document.getElementById('preview').style.display = 'block';
                };
                reader.readAsDataURL(file);
            }

            function handleBatchUpload(e) {
                const files = e.target.files;
                const list = document.getElementById('fileList');
                list.innerHTML = '';
                
                for (let file of files) {
                    const div = document.createElement('div');
                    div.className = 'flex items-center justify-between bg-gray-100 p-3 rounded';
                    div.innerHTML = `<span>${file.name}</span><span class="text-sm text-gray-500">${(file.size/1024).toFixed(1)}KB</span>`;
                    list.appendChild(div);
                }
            }

            async function uploadImage() {
                const file = document.getElementById('imageInput').files[0];
                if (!file) {
                    alert('Please select an image');
                    return;
                }
                
                const formData = new FormData();
                formData.append('file', file);
                
                showLoading(true);
                try {
                    const response = await axios.post(`${API_URL}/predict`, formData);
                    displayResults(response.data);
                } catch (error) {
                    alert('Error: ' + error.response.data.detail);
                } finally {
                    showLoading(false);
                }
            }

            async function uploadBatch() {
                const files = document.getElementById('batchInput').files;
                if (files.length === 0) {
                    alert('Please select images');
                    return;
                }
                
                const formData = new FormData();
                for (let file of files) {
                    formData.append('files', file);
                }
                
                try {
                    const response = await axios.post(`${API_URL}/batch-predict`, formData);
                    displayBatchResults(response.data.results);
                } catch (error) {
                    alert('Error: ' + error.response.data.detail);
                }
            }

            function displayResults(data) {
                document.getElementById('results').style.display = 'block';
                document.getElementById('noResults').style.display = 'none';
                
                const badge = document.getElementById('severityBadge');
                badge.style.backgroundColor = data.color;
                document.getElementById('severityGrade').textContent = data.grade;
                document.getElementById('severityName').textContent = data.severity;
                
                const confidence = data.confidence * 100;
                document.getElementById('confidenceBar').style.width = confidence + '%';
                document.getElementById('confidenceText').textContent = `${confidence.toFixed(1)}% confident`;
                
                const probDiv = document.getElementById('probabilities');
                probDiv.innerHTML = '';
                for (let [key, prob] of Object.entries(data.probabilities)) {
                    const div = document.createElement('div');
                    div.innerHTML = `
                        <div class="flex justify-between mb-1">
                            <span class="text-sm font-semibold">${prob.label}</span>
                            <span class="text-sm text-gray-600">${prob.percentage}</span>
                        </div>
                        <div class="w-full bg-gray-200 rounded-full h-2">
                            <div class="bg-indigo-600 h-2 rounded-full" style="width: ${prob.probability * 100}%"></div>
                        </div>
                    `;
                    probDiv.appendChild(div);
                }
            }

            function displayBatchResults(results) {
                document.getElementById('batchResults').style.display = 'block';
                const list = document.getElementById('resultsList');
                list.innerHTML = '';
                
                results.forEach(result => {
                    const div = document.createElement('div');
                    if (result.error) {
                        div.className = 'bg-red-100 border-l-4 border-red-500 p-4';
                        div.innerHTML = `<p class="font-semibold text-red-700">${result.filename}: Error</p>`;
                    } else {
                        div.className = 'bg-green-100 border-l-4 border-green-500 p-4';
                        div.innerHTML = `
                            <p class="font-semibold text-green-700">${result.filename}</p>
                            <p class="text-sm text-gray-700">${result.severity} (${(result.confidence*100).toFixed(1)}%)</p>
                        `;
                    }
                    list.appendChild(div);
                });
            }

            function showLoading(show) {
                document.getElementById('loading').style.display = show ? 'block' : 'none';
                document.getElementById('results').style.display = show ? 'none' : 'block';
            }

            function switchTab(tab) {
                document.querySelectorAll('.tab-content').forEach(el => el.style.display = 'none');
                document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active', 'bg-indigo-600', 'text-white'));
                
                document.getElementById(tab + '-tab').style.display = 'block';
                document.getElementById('tab-' + tab).classList.add('active', 'bg-indigo-600', 'text-white');
                
                if (tab === 'info') loadInfo();
            }

            async function loadInfo() {
                try {
                    const health = await axios.get(`${API_URL}/health`);
                    const model = await axios.get(`${API_URL}/model-info`);
                    const severity = await axios.get(`${API_URL}/severity-info`);
                    
                    // Display health
                    document.getElementById('device').textContent = `Device: ${health.data.device.toUpperCase()}`;
                    if (health.data.gpu_available) {
                        document.getElementById('device').textContent += ` (${health.data.gpu_name})`;
                    }
                    document.getElementById('status').textContent = `Status: ${health.data.model_ready ? '✓ Ready' : '✗ Loading'}`;
                    
                    // Display model info
                    const modelDiv = document.getElementById('modelInfo');
                    modelDiv.innerHTML = `
                        <div><strong>Model:</strong> ${model.data.model_name}</div>
                        <div><strong>Classes:</strong> ${model.data.num_classes}</div>
                        <div><strong>Input Size:</strong> ${model.data.input_size}x${model.data.input_size}</div>
                        <div><strong>Framework:</strong> ${model.data.framework}</div>
                        <div><strong>Device:</strong> ${health.data.device.toUpperCase()}</div>
                    `;
                    
                    // Display severity guide
                    const severityDiv = document.getElementById('severityGuide');
                    severityDiv.innerHTML = '';
                    severity.data.levels.forEach(level => {
                        const div = document.createElement('div');
                        div.className = 'p-4 rounded-lg';
                        div.style.borderLeft = `4px solid ${level.color}`;
                        div.innerHTML = `
                            <div class="font-bold" style="color: ${level.color}">Grade ${level.grade}: ${level.name}</div>
                            <div class="text-sm text-gray-700 mt-1">${level.description}</div>
                        `;
                        severityDiv.appendChild(div);
                    });
                } catch (error) {
                    console.error('Error loading info:', error);
                }
            }

            // Load health on startup
            window.addEventListener('load', loadInfo);
        </script>
    </body>
    </html>
    """


if __name__ == "__main__":
    logger.info("""
    ╔════════════════════════════════════════════════════════════════╗
    ║                                                                ║
    ║   🔬 Diabetic Retinopathy Detection - Web API                  ║
    ║                                                                ║
    ║   🌐 Open your browser: http://localhost:8000                  ║
    ║   📚 API Docs: http://localhost:8000/docs                     ║
    ║   🔧 ReDoc: http://localhost:8000/redoc                       ║
    ║                                                                ║
    ║   Press CTRL+C to stop                                         ║
    ║                                                                ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
