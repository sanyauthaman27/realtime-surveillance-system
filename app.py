# app.py
import os
import io
import time
import uuid
import math
from flask import Flask, request, jsonify, render_template, send_file, url_for
from ultralytics import YOLO
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from utils.tracker import CentroidTracker
from utils.detectors import SuspicionEngine
from werkzeug.utils import secure_filename
from pathlib import Path
import nltk
nltk.download('punkt', quiet=True)

UPLOAD_FOLDER = "uploads"
RESULTS_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

MODEL_NAME = "yolov8n.pt"  
yolo = YOLO(MODEL_NAME)

COCO_PERSON_CLASS_NAMES = {'person','car','truck','motorcycle','bicycle','bus'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ['mp4','mov','avi','mkv','webm','mpeg']

@app.route('/')
def index():
    return render_template('index.html')

def detect_frame_yolo(frame):

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = yolo.predict(img, imgsz=640, conf=0.35, verbose=False, device=0 if yolo.device.type!='cpu' else 'cpu')
    detections = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls = int(box.cls.cpu().numpy()) if hasattr(box, 'cls') else None
            conf = float(box.conf.cpu().numpy()) if hasattr(box, 'conf') else float(box.conf)
            x1,y1,x2,y2 = map(float, box.xyxy.cpu().numpy()[0]) if hasattr(box, 'xyxy') else box.xyxy[0].tolist()
            cls_name = r.names[cls] if cls is not None and cls < len(r.names) else "obj"
            detections.append({'bbox':[int(x1),int(y1),int(x2),int(y2)], 'class': cls_name, 'conf': conf})
    return detections

def generate_pdf_report(path, events, descriptions, video_path, heatmap_path):
    c = canvas.Canvas(path, pagesize=A4)
    width, height = A4
    c.setFont("Helvetica-Bold", 16)
    c.drawString(30, height-40, "Suspicious Activity Report")
    c.setFont("Helvetica", 10)
    c.drawString(30, height-60, f"Video analyzed: {os.path.basename(video_path)}")
    c.drawString(30, height-75, f"Generated: {time.ctime()}")
    y = height-110
    for i,ev in enumerate(events):
        tstr = f"{ev['type'].upper()} — from {ev['start_time']:.2f}s to {ev['end_time']:.2f}s"
        c.setFont("Helvetica-Bold", 12)
        c.drawString(30, y, tstr)
        y -= 16
        c.setFont("Helvetica", 10)
        c.drawString(40, y, f"Detected objects: {', '.join({d['class'] for d in ev['dets']})}")
        y -= 14
        desc = next((d['text'] for d in descriptions if abs(d['time'] - ev['start_time']) < 1.0), "")
        if desc:
            lines = nltk.tokenize.sent_tokenize(desc)
            for line in lines:
                c.drawString(40, y, line[:120])
                y -= 12
        else:
            c.drawString(40, y, "(No auto-description available)")
            y -= 12
        y -= 6
        if y < 120:
            c.showPage()
            y = height-80
    if heatmap_path and os.path.exists(heatmap_path):
        c.showPage()
        c.drawString(30, height-40, "Heatmap")
        try:
            from reportlab.lib.utils import ImageReader
            img = ImageReader(heatmap_path)
            iw, ih = img.getSize()
            maxw = width - 60
            scale = maxw / iw
            c.drawImage(img, 30, height - 60 - ih*scale, width=iw*scale, height=ih*scale)
        except Exception as e:
            print("Failed to add heatmap:", e)
    c.save()

def save_clip_from_times(input_video_path, start_t, end_t, out_path):
    clip = VideoFileClip(input_video_path).subclip(start_t, end_t)
    clip.write_videofile(out_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)

@app.route('/analyze_upload', methods=['POST'])
def analyze_upload():
    file = request.files.get('video')
    if not file:
        return jsonify({'error':'no file'}), 400
    filename = secure_filename(file.filename)
    if not allowed_file(filename):
        return jsonify({'error':'unsupported file type'}), 400
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4().hex}_{filename}")
    file.save(save_path)

    # Process video
    vidcap = cv2.VideoCapture(save_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    tracker = CentroidTracker(max_disappeared=30, max_distance=70)
    engine = SuspicionEngine((height,width,3), fps)

    results_events = []
    frame_idx = 0
    detections_buffer = []  
    print("Starting frame processing...")
    progress_last = time.time()
    while True:
        ret, frame = vidcap.read()
        if not ret:
            break
        frame_idx += 1
        timestamp = frame_idx / fps
        dets = detect_frame_yolo(frame)
        rects = [d['bbox'] for d in dets if d['class'] in COCO_PERSON_CLASS_NAMES]
        id_map = tracker.update(rects)
        dets_for_engine = []
        for d in dets:
            assigned_id = None
            if d['class'] in COCO_PERSON_CLASS_NAMES:
                bx1,by1,bx2,by2 = d['bbox']
                cx,cy = int((bx1+bx2)/2), int((by1+by2)/2)
                for oid,cent in tracker.objects.items():
                    if np.hypot(cent[0]-cx, cent[1]-cy) < 80:
                        assigned_id = oid
                        break
            dets_for_engine.append({'bbox':d['bbox'], 'class':d['class'], 'conf':d['conf'], 'id': assigned_id})
        engine.detect(frame_idx, timestamp, dets_for_engine, tracker, fps)
        detections_buffer.append({'frame_idx':frame_idx, 'timestamp':timestamp, 'detections':dets_for_engine})
        if time.time() - progress_last > 5:
            progress_last = time.time()
            print(f"Processed frame {frame_idx}/{total_frames}")
    vidcap.release()
    engine.finalize_events()

    artifacts = {'clips':[], 'descriptions':[]}
    hm = engine.generate_heatmap_overlay()
    heatmap_path = os.path.join(app.config['RESULTS_FOLDER'], f"heatmap_{uuid.uuid4().hex}.jpg")
    cv2.imwrite(heatmap_path, hm)
    for ev in engine.events:
        start_t = max(0.0, ev['start_time'] - 2.0)
        end_t = ev['end_time'] + 2.0
        clip_name = f"clip_{ev['type']}_{uuid.uuid4().hex}.mp4"
        clip_path = os.path.join(app.config['RESULTS_FOLDER'], clip_name)
        try:
            save_clip_from_times(save_path, start_t, end_t, clip_path)
            artifacts['clips'].append({'type': ev['type'], 'start_time': start_t, 'end_time': end_t, 'path': clip_path})
            txt = generate_description_for_event(ev)
            artifacts['descriptions'].append({'time': start_t, 'text': txt})
        except Exception as e:
            print("Clip creation failed:", e)

    pdf_path = os.path.join(app.config['RESULTS_FOLDER'], f"report_{uuid.uuid4().hex}.pdf")
    generate_pdf_report(pdf_path, engine.events, artifacts['descriptions'], save_path, heatmap_path)

    base = request.host_url.rstrip('/')
    clip_entries = []
    for c in artifacts['clips']:
        url = url_for('download_file', filename=os.path.basename(c['path']), _external=True)
        clip_entries.append({'type': c['type'], 'start_time': c['start_time'], 'end_time': c['end_time'], 'url': url})

    heatmap_url = url_for('download_file', filename=os.path.basename(heatmap_path), _external=True)
    report_url = url_for('download_file', filename=os.path.basename(pdf_path), _external=True)

    return jsonify({'clips': clip_entries, 'report_url': report_url, 'heatmap_url': heatmap_url, 'descriptions': artifacts['descriptions']})

@app.route('/download/<filename>')
def download_file(filename):
    path = os.path.join(app.config['RESULTS_FOLDER'], filename)
    if not os.path.exists(path):
        return "Not found", 404
    return send_file(path, as_attachment=True)

def generate_description_for_event(ev):
    typ = ev['type']
    start = ev['start_time']
    end = ev['end_time']
    if typ == 'loitering':
        return f"A person loitered at approximately {start:.1f} to {end:.1f} seconds. They remained in the same area for an unusually long time."
    if typ == 'fighting':
        return f"Two or more people were observed in close contact with erratic movement patterns between {start:.1f}–{end:.1f} seconds; behavior consistent with a physical altercation."
    if typ == 'crowd':
        return f"A sudden crowd formation of multiple people was recorded at {start:.1f}–{end:.1f} seconds. High density may indicate an incident."
    if typ == 'collapse':
        return f"A person appears to have collapsed or is lying on the ground between {start:.1f}–{end:.1f} seconds. Medical attention may be required."
    if typ == 'accident':
        return f"An abrupt change in motion suggests a possible accident involving a vehicle and a person around {start:.1f}–{end:.1f} seconds."
    return f"A suspicious event of type {typ} was detected from {start:.1f} to {end:.1f} seconds."

@app.route('/live_detect', methods=['POST'])
def live_detect():
    """
    Lightweight endpoint to accept a single frame (from webcam) and return a short summary.
    Not intended to run full event pipeline, but to provide real-time feedback.
    """
    file = request.files.get('frame')
    if not file:
        return jsonify({'error':'no frame'}), 400
    data = file.read()
    arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    dets = detect_frame_yolo(frame)
    classes = set(d['class'] for d in dets)
    summary = ', '.join(list(classes)[:5]) if classes else 'none'
    return jsonify({'summary': summary})

@app.route('/frame_detect', methods=['POST'])
def frame_detect():
    file = request.files.get('frame')
    if not file:
        return jsonify({'error':'no frame'}), 400
    data = file.read()
    arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    dets = detect_frame_yolo(frame)
    classes = set(d['class'] for d in dets)
    summary = ' and '.join(list(classes)[:3]) if classes else 'none'
    return jsonify({'summary': summary, 'detections': [{'class':d['class'], 'conf':d['conf']} for d in dets]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
