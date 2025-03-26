from flask import Flask, request, send_from_directory, jsonify
import os
import fitz  # PyMuPDF for thumbnail generation
from werkzeug.utils import secure_filename
from flask_cors import CORS  
from video_processor import process_video  

app = Flask(__name__)
CORS(app)  

# Folder paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))  
UPLOAD_FOLDER = os.path.join(BASE_DIR, "backend/uploads")
COMIC_FOLDER = os.path.join(BASE_DIR, "backend/generated_comics")
SAMPLE_COMIC_FOLDER = os.path.join(BASE_DIR, "backend/static/sample_comics")
THUMBNAIL_FOLDER = os.path.join(BASE_DIR, "backend/static/thumbnails")

# Ensure folders exist
for folder in [UPLOAD_FOLDER, COMIC_FOLDER, SAMPLE_COMIC_FOLDER, THUMBNAIL_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["COMIC_FOLDER"] = COMIC_FOLDER
app.config["SAMPLE_COMIC_FOLDER"] = SAMPLE_COMIC_FOLDER
app.config["THUMBNAIL_FOLDER"] = THUMBNAIL_FOLDER

def generate_thumbnail(pdf_path, thumb_path):
    """Generate a thumbnail from the first page of a PDF using PyMuPDF (fitz)."""
    try:
        if not os.path.exists(pdf_path):
            print(f"PDF not found: {pdf_path}")
            return
        
        doc = fitz.open(pdf_path)
        page = doc[0]
        pix = page.get_pixmap(dpi=100)
        pix.save(thumb_path)
        print(f"Thumbnail saved: {thumb_path}")
    
    except Exception as e:
        print(f"Error generating thumbnail for {pdf_path}: {e}")

@app.route("/api/upload_video", methods=["POST"])
def upload_video():
    """Handles video upload and processes it into a comic PDF."""
    if "video" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["video"]
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    # Process the video and generate the comic
    pdf_name = process_video(file_path)
    pdf_path = os.path.join(COMIC_FOLDER, pdf_name)

    # Generate thumbnail for the new comic
    thumb_filename = pdf_name.replace(".pdf", ".jpg")
    thumb_path = os.path.join(THUMBNAIL_FOLDER, thumb_filename)
    generate_thumbnail(pdf_path, thumb_path)

    return jsonify({
        "message": "Comic generated successfully",
        "pdf_url": f"/api/comics/download/{pdf_name}",
        "thumbnail": f"/api/comics/thumbnails/{thumb_filename}"
    })

@app.route("/api/comics/samples", methods=["GET"])
def list_sample_comics():
    """List all sample comics with thumbnails."""
    try:
        comics = []
        print(f"Checking Sample Comics Directory: {SAMPLE_COMIC_FOLDER}")

        if not os.path.exists(SAMPLE_COMIC_FOLDER):
            print("Sample comics folder does not exist!")
            return jsonify({"error": "Sample comics folder missing"}), 500

        files = os.listdir(SAMPLE_COMIC_FOLDER)
        print(f"Found Files: {files}")  

        for filename in files:
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(SAMPLE_COMIC_FOLDER, filename)
                thumb_filename = filename.replace(".pdf", ".jpg")
                thumb_path = os.path.join(THUMBNAIL_FOLDER, thumb_filename)

                if not os.path.exists(thumb_path):
                    print(f"Generating thumbnail for {filename}")
                    generate_thumbnail(pdf_path, thumb_path)

                comics.append({
                    "name": filename.replace(".pdf", ""),
                    "url": f"/api/comics/download/sample/{filename}",  
                    "thumbnail": f"/api/comics/thumbnails/{thumb_filename}"
                })

        print("Final Comics List:", comics)
        return jsonify(comics)
    
    except Exception as e:
        print(f"Error fetching comics: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/comics/download/sample/<filename>", methods=["GET"])
def download_sample_comic(filename):
    """Download a sample comic PDF."""
    try:
        pdf_path = os.path.join(SAMPLE_COMIC_FOLDER, filename)
        if not os.path.exists(pdf_path):
            return jsonify({"error": "File not found"}), 404

        return send_from_directory(SAMPLE_COMIC_FOLDER, filename, as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/comics/download/<filename>", methods=["GET"])
def download_comic(filename):
    """Download a comic PDF (handles both generated and sample comics)."""
    try:
        generated_comic_path = os.path.join(COMIC_FOLDER, filename)
        sample_comic_path = os.path.join(SAMPLE_COMIC_FOLDER, filename)

        if os.path.exists(generated_comic_path):
            return send_from_directory(COMIC_FOLDER, filename, as_attachment=True)
        elif os.path.exists(sample_comic_path):
            return send_from_directory(SAMPLE_COMIC_FOLDER, filename, as_attachment=True)
        else:
            return jsonify({"error": "File not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/comics/thumbnails/<filename>", methods=["GET"])
def get_thumbnail(filename):
    """Serve the generated thumbnail images."""
    try:
        thumb_path = os.path.join(THUMBNAIL_FOLDER, filename)
        if not os.path.exists(thumb_path):
            return jsonify({"error": "Thumbnail not found"}), 404
        
        return send_from_directory(THUMBNAIL_FOLDER, filename)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
