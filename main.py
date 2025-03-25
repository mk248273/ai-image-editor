import os
import base64
import mimetypes
import logging
from typing import Optional, Dict, Any

from flask import Flask, request, jsonify, render_template, current_app
from werkzeug.utils import secure_filename
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImageEditor:
    def __init__(self, api_key: str, model: str):
        """
        Initialize the Google AI client for image editing
        
        Args:
            api_key (str): Google AI API key
            model (str): Model to use for image generation
        """
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def _validate_input(self, image_data: bytes, query: str) -> bool:
        """
        Validate input before processing
        
        Args:
            image_data (bytes): Image data
            query (str): Edit description
        
        Returns:
            bool: Validation result
        """
        if not image_data:
            logger.error("No image data provided")
            return False
        
        if not query or len(query.strip()) < 3:
            logger.error("Invalid or too short query")
            return False
        
        return True

    def process_image(self, image_data: bytes, query: str) -> Optional[Dict[str, Any]]:
        """
        Process image with AI editing
        
        Args:
            image_data (bytes): Image data to edit
            query (str): Edit description
        
        Returns:
            Optional[Dict[str, Any]]: Processed image details or None
        """
        if not self._validate_input(image_data, query):
            return None

        try:
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=query),
                        types.Part(inline_data=types.Blob(mime_type="image/png", data=image_data)),
                    ],
                ),
            ]
            
            generate_config = types.GenerateContentConfig(
                temperature=0.8,
                top_p=0.9,
                top_k=30,
                max_output_tokens=8192,
                response_modalities=["image", "text"],
                response_mime_type="image/png",
            )
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=generate_config,
            )

            # Check and process response
            if response.candidates and response.candidates[0].content.parts:
                inline_data = response.candidates[0].content.parts[0].inline_data
                
                # Generate secure filename
                file_extension = mimetypes.guess_extension(inline_data.mime_type or 'image/png')
                filename = secure_filename(f"edited_image_{hash(query)}{file_extension}")
                filepath = os.path.join('static', 'uploads', filename)
                
                # Ensure upload directory exists
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                
                # Save file
                with open(filepath, "wb") as f:
                    f.write(inline_data.data)
                
                logger.info(f"Image processed successfully: {filename}")
                
                return {
                    "filename": filename,
                    "path": filepath
                }
            
            logger.warning("No valid image generated")
            return None

        except Exception as e:
            logger.error(f"Image processing error: {e}")
            return None

def create_app():
    """
    Application factory for Flask app
    
    Returns:
        Flask: Configured Flask application
    """
    app = Flask(__name__)
    
    # Configure app from environment
    app.config.update(
        SECRET_KEY=os.getenv('SECRET_KEY', 'development_secret_key'),
        MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB max upload
    )

    # Initialize image editor
    image_editor = ImageEditor(
        api_key=os.getenv('GOOGLE_AI_API_KEY', ''),
        model="gemini-pro-vision"
    )

    @app.route('/')
    def home():
        """Render the main page"""
        return render_template("index.html")

    @app.route('/edit-image', methods=['POST'])
    def edit_image():
        """
        Handle image editing requests
        
        Returns:
            JSON response with edited image or error
        """
        try:
            # Validate request
            if 'image' not in request.files:
                return jsonify({"error": "No image uploaded"}), 400
            
            image = request.files['image']
            query = request.form.get('query', '').strip()
            
            if not query:
                return jsonify({"error": "Edit description is required"}), 400

            # Read image data
            image_data = image.read()
            
            # Process image
            result = image_editor.process_image(image_data, query)
            
            if result:
                # Encode image for response
                with open(result['path'], "rb") as img_file:
                    encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
                
                return jsonify({
                    "image": encoded_image,
                    "filename": result['filename']
                })
            else:
                return jsonify({"error": "Failed to process image"}), 500

        except Exception as e:
            logger.error(f"Unexpected error in edit_image: {e}")
            return jsonify({"error": "Internal server error"}), 500

    # Error handlers
    @app.errorhandler(413)
    def request_entity_too_large(error):
        """Handle large file uploads"""
        return jsonify({"error": "File too large"}), 413

    return app

# Create and run the app
app = create_app()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)