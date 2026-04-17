"""
Step 2: Browser-Based Rooftop Labeling Tool
=============================================
A Flask-based web app for quickly labeling rooftop crops
as one of: Gable, Hip, Flat, or Skip (unclear/bad crop).

Usage:
    python labeling_tool.py

Then open http://localhost:5555 in your browser.
Press keyboard shortcuts: 1=Gable, 2=Hip, 3=Flat, 4=Skip
"""

import os
import json
from flask import Flask, render_template_string, jsonify, request, send_from_directory

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CROPS_DIR = os.path.join(BASE_DIR, "Rooftop_Crops")
METADATA_FILE = os.path.join(CROPS_DIR, "crop_metadata.json")
LABELS_FILE = os.path.join(CROPS_DIR, "labels.json")

app = Flask(__name__)

# Load metadata
with open(METADATA_FILE, 'r') as f:
    metadata = json.load(f)

# Load existing labels if they exist
if os.path.exists(LABELS_FILE):
    with open(LABELS_FILE, 'r') as f:
        labels = json.load(f)
else:
    labels = {}

# Get sorted list of all crop filenames
all_crops = sorted(metadata.keys())

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rooftop Labeling Tool</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Inter', sans-serif;
            background: #0f0f23;
            color: #e0e0e0;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        
        .header {
            width: 100%;
            background: linear-gradient(135deg, #1a1a3e 0%, #2d1b69 100%);
            padding: 16px 32px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        
        .header h1 {
            font-size: 20px;
            font-weight: 700;
            background: linear-gradient(135deg, #a78bfa, #60a5fa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .progress-info {
            display: flex;
            gap: 20px;
            align-items: center;
            font-size: 14px;
        }
        
        .progress-bar-container {
            width: 200px;
            height: 8px;
            background: rgba(255,255,255,0.1);
            border-radius: 4px;
            overflow: hidden;
        }
        
        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, #a78bfa, #60a5fa);
            transition: width 0.3s ease;
            border-radius: 4px;
        }
        
        .stats {
            display: flex;
            gap: 16px;
            font-size: 13px;
        }
        
        .stat { 
            padding: 4px 12px; 
            border-radius: 20px; 
            font-weight: 500;
        }
        
        .stat-gable { background: rgba(239, 68, 68, 0.2); color: #fca5a5; }
        .stat-hip { background: rgba(59, 130, 246, 0.2); color: #93c5fd; }
        .stat-flat { background: rgba(34, 197, 94, 0.2); color: #86efac; }
        .stat-skip { background: rgba(156, 163, 175, 0.2); color: #d1d5db; }
        
        .main-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 24px;
            gap: 24px;
            max-width: 900px;
            width: 100%;
        }
        
        .image-container {
            background: #1a1a2e;
            border-radius: 16px;
            padding: 16px;
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 12px;
        }
        
        .image-container img {
            max-width: 100%;
            max-height: 400px;
            border-radius: 8px;
            image-rendering: auto;
        }
        
        .source-info {
            font-size: 12px;
            color: #888;
        }
        
        .crop-info {
            font-size: 13px;
            color: #aaa;
        }
        
        .buttons {
            display: flex;
            gap: 16px;
            flex-wrap: wrap;
            justify-content: center;
        }
        
        .btn {
            padding: 14px 32px;
            border: none;
            border-radius: 12px;
            font-family: 'Inter', sans-serif;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .btn:hover { transform: translateY(-2px); box-shadow: 0 4px 16px rgba(0,0,0,0.3); }
        .btn:active { transform: translateY(0); }
        
        .btn-gable { 
            background: linear-gradient(135deg, #ef4444, #dc2626); 
            color: white; 
        }
        .btn-hip { 
            background: linear-gradient(135deg, #3b82f6, #2563eb); 
            color: white; 
        }
        .btn-flat { 
            background: linear-gradient(135deg, #22c55e, #16a34a); 
            color: white; 
        }
        .btn-skip { 
            background: linear-gradient(135deg, #6b7280, #4b5563); 
            color: white; 
        }
        
        .btn-nav {
            padding: 10px 20px;
            background: rgba(255,255,255,0.1);
            color: #ccc;
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 8px;
            cursor: pointer;
            font-family: 'Inter', sans-serif;
            font-size: 13px;
            transition: all 0.2s;
        }
        
        .btn-nav:hover { background: rgba(255,255,255,0.2); }
        
        .nav-row {
            display: flex;
            gap: 12px;
            align-items: center;
        }
        
        .keyboard-hint {
            font-size: 11px;
            color: #666;
            text-align: center;
            margin-top: 8px;
        }
        
        kbd {
            background: rgba(255,255,255,0.1);
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 12px;
            border: 1px solid rgba(255,255,255,0.15);
        }
        
        .done-message {
            text-align: center;
            padding: 60px;
        }
        
        .done-message h2 {
            font-size: 28px;
            margin-bottom: 16px;
            background: linear-gradient(135deg, #a78bfa, #60a5fa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🏠 Rooftop Type Labeler</h1>
        <div class="progress-info">
            <div class="stats">
                <span class="stat stat-gable" id="stat-gable">Gable: 0</span>
                <span class="stat stat-hip" id="stat-hip">Hip: 0</span>
                <span class="stat stat-flat" id="stat-flat">Flat: 0</span>
                <span class="stat stat-skip" id="stat-skip">Skip: 0</span>
            </div>
            <span id="progress-text">0 / 0</span>
            <div class="progress-bar-container">
                <div class="progress-bar" id="progress-bar" style="width: 0%"></div>
            </div>
        </div>
    </div>
    
    <div class="main-container">
        <div class="image-container" id="image-container">
            <img id="crop-image" src="" alt="Rooftop crop">
            <div class="source-info" id="source-info"></div>
            <div class="crop-info" id="crop-info"></div>
        </div>
        
        <div class="buttons">
            <button class="btn btn-gable" onclick="label('gable')">
                <kbd>1</kbd> Gable (V-shape)
            </button>
            <button class="btn btn-hip" onclick="label('hip')">
                <kbd>2</kbd> Hip (Prism)
            </button>
            <button class="btn btn-flat" onclick="label('flat')">
                <kbd>3</kbd> Flat
            </button>
            <button class="btn btn-skip" onclick="label('skip')">
                <kbd>4</kbd> Skip / Unclear
            </button>
        </div>
        
        <div class="nav-row">
            <button class="btn-nav" onclick="navigate(-1)">← Previous</button>
            <button class="btn-nav" onclick="jumpToUnlabeled()">Next Unlabeled</button>
            <button class="btn-nav" onclick="navigate(1)">Next →</button>
            <button class="btn-nav" onclick="saveLabels()">💾 Save Progress</button>
        </div>
        
        <div class="keyboard-hint">
            Keyboard: <kbd>1</kbd> Gable  <kbd>2</kbd> Hip  <kbd>3</kbd> Flat  <kbd>4</kbd> Skip  
            <kbd>←</kbd> Prev  <kbd>→</kbd> Next  <kbd>S</kbd> Save
        </div>
    </div>

    <script>
        let currentIndex = 0;
        let crops = [];
        let labels = {};
        let stats = { gable: 0, hip: 0, flat: 0, skip: 0 };
        
        async function init() {
            const resp = await fetch('/api/data');
            const data = await resp.json();
            crops = data.crops;
            labels = data.labels;
            
            // Count existing labels
            stats = { gable: 0, hip: 0, flat: 0, skip: 0 };
            for (const [k, v] of Object.entries(labels)) {
                if (stats.hasOwnProperty(v)) stats[v]++;
            }
            
            // Find first unlabeled
            currentIndex = crops.findIndex(c => !labels[c]);
            if (currentIndex === -1) currentIndex = 0;
            
            updateDisplay();
        }
        
        function updateDisplay() {
            if (crops.length === 0) return;
            
            const crop = crops[currentIndex];
            document.getElementById('crop-image').src = '/crops/' + crop;
            document.getElementById('source-info').textContent = 
                'Source: ' + crop + (labels[crop] ? ' | Current label: ' + labels[crop].toUpperCase() : ' | UNLABELED');
            document.getElementById('crop-info').textContent = 
                'Crop ' + (currentIndex + 1) + ' of ' + crops.length;
            
            const labeled = Object.keys(labels).length;
            document.getElementById('progress-text').textContent = labeled + ' / ' + crops.length;
            document.getElementById('progress-bar').style.width = 
                (labeled / crops.length * 100) + '%';
            
            document.getElementById('stat-gable').textContent = 'Gable: ' + stats.gable;
            document.getElementById('stat-hip').textContent = 'Hip: ' + stats.hip;
            document.getElementById('stat-flat').textContent = 'Flat: ' + stats.flat;
            document.getElementById('stat-skip').textContent = 'Skip: ' + stats.skip;
        }
        
        async function label(type) {
            const crop = crops[currentIndex];
            
            // Update stats
            if (labels[crop] && stats.hasOwnProperty(labels[crop])) {
                stats[labels[crop]]--;
            }
            labels[crop] = type;
            stats[type]++;
            
            // Save to server
            await fetch('/api/label', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ crop: crop, label: type })
            });
            
            // Advance to next unlabeled
            let nextIdx = currentIndex + 1;
            while (nextIdx < crops.length && labels[crops[nextIdx]]) {
                nextIdx++;
            }
            if (nextIdx < crops.length) {
                currentIndex = nextIdx;
            } else {
                currentIndex = Math.min(currentIndex + 1, crops.length - 1);
            }
            
            updateDisplay();
        }
        
        function navigate(delta) {
            currentIndex = Math.max(0, Math.min(crops.length - 1, currentIndex + delta));
            updateDisplay();
        }
        
        function jumpToUnlabeled() {
            const idx = crops.findIndex((c, i) => i > currentIndex && !labels[c]);
            if (idx !== -1) {
                currentIndex = idx;
            } else {
                // Wrap around
                const idx2 = crops.findIndex(c => !labels[c]);
                if (idx2 !== -1) currentIndex = idx2;
            }
            updateDisplay();
        }
        
        async function saveLabels() {
            const resp = await fetch('/api/save', { method: 'POST' });
            const data = await resp.json();
            alert('Saved! ' + data.count + ' labels saved to disk.');
        }
        
        document.addEventListener('keydown', (e) => {
            if (e.key === '1') label('gable');
            else if (e.key === '2') label('hip');
            else if (e.key === '3') label('flat');
            else if (e.key === '4') label('skip');
            else if (e.key === 'ArrowLeft') navigate(-1);
            else if (e.key === 'ArrowRight') navigate(1);
            else if (e.key === 's' || e.key === 'S') saveLabels();
        });
        
        init();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/crops/<path:filename>')
def serve_crop(filename):
    return send_from_directory(CROPS_DIR, filename)

@app.route('/api/data')
def get_data():
    return jsonify({
        'crops': all_crops,
        'labels': labels
    })

@app.route('/api/label', methods=['POST'])
def set_label():
    data = request.get_json()
    crop = data['crop']
    label = data['label']
    labels[crop] = label
    
    # Auto-save every 20 labels
    if len(labels) % 20 == 0:
        with open(LABELS_FILE, 'w') as f:
            json.dump(labels, f, indent=2)
    
    return jsonify({'status': 'ok'})

@app.route('/api/save', methods=['POST'])
def save_labels():
    with open(LABELS_FILE, 'w') as f:
        json.dump(labels, f, indent=2)
    return jsonify({'status': 'saved', 'count': len(labels)})

if __name__ == '__main__':
    print(f"\n{'='*50}")
    print(f"ROOFTOP LABELING TOOL")
    print(f"{'='*50}")
    print(f"Total crops to label: {len(all_crops)}")
    print(f"Already labeled: {len(labels)}")
    print(f"Remaining: {len(all_crops) - len(labels)}")
    print(f"\nOpen http://localhost:5555 in your browser")
    print(f"Use keyboard shortcuts: 1=Gable, 2=Hip, 3=Flat, 4=Skip")
    print(f"{'='*50}\n")
    app.run(host='0.0.0.0', port=5555, debug=False)
