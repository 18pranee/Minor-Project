import os
from flask import Flask, request, jsonify, render_template

# Local imports
from predict import ensemble_predict
from utils.db import init_db, get_history, get_analytics_data, log_prediction, delete_prediction
from utils.recommendation import get_recommendation

app = Flask(__name__)

# Configure upload folders
UPLOAD_FOLDER = 'static/uploads'
GRADCAM_FOLDER = 'static/gradcam'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GRADCAM_FOLDER'] = GRADCAM_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)

# Initialize SQLite database
init_db()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ================================
# VIEW ROUTES (Dashboard UI)
# ================================

@app.route('/')
def index():
    # Route for the main dashboard display
    return render_template('dashboard.html')

@app.route('/upload')
def upload_page():
    # Route for the upload and predict page
    return render_template('upload.html')

@app.route('/history')
def history_page():
    # Retrieve past predictions from the database
    history = get_history()
    return render_template('history.html', history=history)

@app.route('/analytics')
def analytics_page():
    # Retrieve aggregated stats for charts
    stats = get_analytics_data()
    return render_template('analytics.html', stats=stats)


# ================================
# API ROUTES (Backend Logic)
# ================================

@app.route('/delete/<int:record_id>', methods=['DELETE'])
def delete_record(record_id):
    success = delete_prediction(record_id)
    if success:
        return jsonify({"message": "Record deleted successfully"}), 200
    else:
        return jsonify({"error": "Record not found"}), 404


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if file and allowed_file(file.filename):
        # Generate safe filenames
        import uuid
        ext = file.filename.rsplit('.', 1)[1].lower()
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{unique_id}.{ext}"
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Run the prediction pipeline with Grad-CAM++ capabilities
            # It will save the Grad-CAM++ to the configured directory
            results = ensemble_predict(filepath, save_cam_dir=app.config['GRADCAM_FOLDER'])
            
            disease_name = results.get("disease", "Unknown")
            confidence = results.get("confidence", 0.0)
            
            # 1. Look up recommendation
            rec = get_recommendation(disease_name)
            
            # Format relative paths for DB and Frontend consumption
            rel_original = f"/{UPLOAD_FOLDER}/{filename}"
            # Extract relative path if generated successfully
            cam_path = results.get('cam_cnn')
            rel_cam = f"/{cam_path}" if cam_path else None
            
            # 2. Write to SQLite database
            log_prediction(
                image_path=rel_original,
                predicted_disease=disease_name,
                confidence=confidence,
                rec_dict=rec,
                gradcam_image=rel_cam
            )
            
            # 3. Construct Final JSON Response for UI Upload Page
            response_data = {
                "original_image": rel_original,
                "disease": disease_name,
                "confidence": confidence,
                "probabilities": results.get("probabilities", {}),
                "cam_image": rel_cam,
                "recommendation": rec
            }
            
            return jsonify(response_data)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500
            
    return jsonify({"error": "Invalid file format. Please upload JPG or PNG."}), 400

if __name__ == '__main__':
    print("Serving AI Plant Disease Dashboard on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
