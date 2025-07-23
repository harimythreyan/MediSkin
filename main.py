import os
import numpy as np
import json
import base64
import io
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import model_from_json, load_model
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from bson import ObjectId

# ===== APP INITIALIZATION =====
app = Flask(__name__)
CORS(app)

# Custom JSON encoder for MongoDB ObjectId
class MongoJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        return super().default(obj)

app.json_encoder = MongoJSONEncoder

# ===== DATABASE CONFIGURATION =====
# MongoDB connection
try:
    client = MongoClient("mongodb+srv://admin:admin_123@mediskin.cpheemi.mongodb.net/?retryWrites=true&w=majority&appName=mediskin")
    db = client["medicine_db"]
    medicine_collection = db["medicines"]
    disease_collection = db["diseases"]
    skin_conditions_collection = db["skin_conditions"]
    print("Connected to MongoDB successfully")
except Exception as e:
    print(f"Error connecting to MongoDB: {str(e)}")
    # Using a fallback approach is a good practice

# ===== SKIN DISEASE DATA =====
# Skin disease class labels mapping
SKIN_CLASS_LABELS = {
    "0": {"code": "akiec", "name": "Actinic keratoses and intraepithelial carcinomae"},
    "1": {"code": "bcc", "name": "Basal cell carcinoma"},
    "2": {"code": "bkl", "name": "Benign keratosis-like lesions"},
    "3": {"code": "df", "name": "Dermatofibroma"},
    "4": {"code": "nv", "name": "Melanocytic nevi"},
    "5": {"code": "vasc", "name": "Pyogenic granulomas and hemorrhage"},
    "6": {"code": "mel", "name": "Melanoma"}
}

# Detailed information about skin conditions
SKIN_CONDITIONS_DATA = {
    "skin_conditions": {
        "0": {
            "code": "akiec",
            "icd10": "L57.0",
            "name": "Actinic keratoses and intraepithelial carcinomae",
            "description": "Precancerous scaly lesions caused by chronic UV damage with 5-10% risk of progressing to squamous cell carcinoma.",
            "key_features": {
                "appearance": "Rough, erythematous or pigmented scaly patches",
                "size": "2-10mm",
                "texture": "Sandpaper-like"
            },
            "risk_score": 3,
            "progression_warning": "Increasing thickness, ulceration, or rapid growth may indicate malignant transformation",
            "diagnosis": {
                "primary": "Clinical examination with dermoscopy",
                "confirmation": "Biopsy if SCC suspected"
            },
            "treatment": {
                "first_line": ["Cryotherapy", "Topical 5-FU"],
                "second_line": ["Photodynamic therapy", "Laser ablation"],
                "surgical": "Excision for refractory cases"
            },
            "prevention": {
                "sun_protection": "SPF 50+ broad spectrum",
                "monitoring": "Annual skin exams for high-risk patients"
            }
        },
        "1": {
            "code": "bcc",
            "icd10": "C44.91",
            "name": "Basal cell carcinoma",
            "description": "Slow-growing, locally invasive skin cancer arising from basal cells with rare metastasis.",
            "key_features": {
                "subtypes": ["Nodular", "Superficial", "Morpheaform", "Pigmented"],
                "hallmark": "Pearly appearance with telangiectasia"
            },
            "risk_score": 2,
            "progression_warning": "Neglected lesions may cause significant tissue destruction",
            "diagnosis": {
                "primary": "Dermoscopic examination",
                "gold_standard": "Biopsy with histopathology"
            },
            "treatment": {
                "gold_standard": "Mohs micrographic surgery",
                "alternatives": ["Excisional surgery", "Electrodessication and curettage"],
                "nonsurgical": ["Topical imiquimod", "Radiation therapy"]
            },
            "prevention": {
                "high_risk": "Patients with Gorlin syndrome need regular screening",
                "general": "UV avoidance starting in childhood"
            }
        },
        "2": {
            "code": "bkl",
            "icd10": "L82.1",
            "name": "Benign keratosis-like lesions",
            "description": "Common benign epidermal proliferations including seborrheic keratoses and lichenoid keratoses.",
            "key_features": {
                "appearance": "Stuck-on, waxy plaques",
                "diagnostic_clue": "Horn pseudocysts on dermoscopy"
            },
            "risk_score": 1,
            "progression_warning": "None (benign condition)",
            "diagnosis": {
                "clinical": "Typical appearance",
                "atypical_cases": "Biopsy to rule out melanoma"
            },
            "treatment": {
                "cosmetic": ["Cryotherapy", "Curettage"],
                "medical": "None required"
            },
            "prevention": {
                "note": "No preventive measures needed"
            }
        },
        "3": {
            "code": "df",
            "icd10": "L72.3",
            "name": "Dermatofibroma",
            "description": "Common benign fibrous skin nodule often following minor trauma.",
            "key_features": {
                "pathognomonic": "Dimple sign on lateral compression",
                "color": "Pink-brown to violaceous"
            },
            "risk_score": 1,
            "progression_warning": "No malignant potential",
            "diagnosis": {
                "clinical": "Characteristic dimpling",
                "uncertain_cases": "Biopsy to exclude dermatofibrosarcoma protuberans"
            },
            "treatment": {
                "asymptomatic": "Observation",
                "symptomatic": "Surgical excision"
            },
            "prevention": {
                "note": "No known prevention"
            }
        },
        "4": {
            "code": "nv",
            "icd10": "D22.9",
            "name": "Melanocytic nevi",
            "description": "Benign proliferations of melanocytes ranging from congenital to acquired types.",
            "key_features": {
                "types": ["Junctional", "Compound", "Dermal"],
                "warning_signs": "ABCDE changes"
            },
            "risk_score": {
                "typical": 1,
                "dysplastic": 2
            },
            "progression_warning": "Atypical nevi may transform to melanoma",
            "diagnosis": {
                "routine": "Clinical monitoring",
                "suspicious": "Dermoscopy with possible excision"
            },
            "treatment": {
                "management": ["Serial photography", "Digital dermoscopy monitoring"],
                "intervention": "Excision of changing lesions"
            },
            "prevention": {
                "high_risk": "Total body photography for patients with >50 nevi"
            }
        },
        "5": {
            "code": "vasc",
            "icd10": "L98.0",
            "name": "Vascular lesions",
            "description": "Includes pyogenic granulomas (lobular capillary hemangiomas) and hemorrhagic lesions.",
            "key_features": {
                "pyogenic_granuloma": "Rapidly growing, friable vascular nodule",
                "hemorrhagic": "Blood-filled blister or bruise"
            },
            "risk_score": 1,
            "progression_warning": "Recurrence common after incomplete removal",
            "diagnosis": {
                "clinical": "Characteristic appearance",
                "atypical": "Biopsy to rule out amelanotic melanoma"
            },
            "treatment": {
                "pg": ["Shave excision with cautery", "Pulsed dye laser"],
                "hemorrhagic": "Pressure dressing"
            },
            "prevention": {
                "pg": "Careful handling of sharp instruments"
            }
        },
        "6": {
            "code": "mel",
            "icd10": "C43.9",
            "name": "Melanoma",
            "description": "Malignant tumor of melanocytes with significant metastatic potential.",
            "key_features": {
                "subtypes": ["Superficial spreading", "Nodular", "Lentigo maligna", "Acral lentiginous"],
                "mnemonic": "ABCDE criteria"
            },
            "risk_score": 5,
            "progression_warning": "Depth of invasion (Breslow thickness) predicts prognosis",
            "diagnosis": {
                "suspicion": "Clinical/dermoscopic examination",
                "confirmation": "Excisional biopsy with 1-2mm margins"
            },
            "treatment": {
                "early_stage": "Wide local excision",
                "advanced": ["Immunotherapy", "Targeted therapy", "Lymph node dissection"],
                "adjuvant": "Sentinel lymph node biopsy for tumors >0.8mm"
            },
            "prevention": {
                "primary": "Sun protection starting in infancy",
                "secondary": "Regular self-exams with partner assistance for hard-to-see areas"
            },
            "staging": {
                "TNM": "Based on tumor thickness, ulceration, nodal involvement, metastasis"
            }
        }
    }
}

# ===== ML MODEL CONFIGURATION =====
# Initialize model variables
skin_model = None

def load_models():
    """Load skin disease classification model from saved files"""
    global skin_model
    
    try:
        # Try different paths and loading methods
        model_paths = [
            ('exported_model/model.json', 'exported_model/weights.h5'),
            ('models/model.json', 'models/weights.h5'),
            'exported_model/full_model.h5',
            'models/full_model.h5'
        ]
        
        for path in model_paths:
            try:
                if isinstance(path, tuple):
                    # Load model from JSON architecture and weights
                    with open(path[0], 'r') as json_file:
                        loaded_model_json = json_file.read()
                    skin_model = model_from_json(loaded_model_json)
                    skin_model.load_weights(path[1])
                    print(f"Loaded skin disease model from {path[0]} and {path[1]}")
                    break
                else:
                    # Load full saved model
                    skin_model = load_model(path)
                    print(f"Loaded skin disease model from {path}")
                    break
            except Exception as e:
                print(f"Failed to load model from {path}: {str(e)}")
                continue
        
        # If no model loaded, create a placeholder
        if skin_model is None:
            print("Creating placeholder model for testing")
            skin_model = create_placeholder_model()
            
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error in model loading process: {str(e)}")
        print("Creating placeholder model for testing")
        skin_model = create_placeholder_model()

def create_placeholder_model():
    """Create a simple model for testing when real models are unavailable"""
    inputs = tf.keras.Input(shape=(28, 28, 3))
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(7, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def preprocess_image(image_data, target_size=(28, 28)):
    """Preprocess image data for model prediction
    
    Args:
        image_data: Raw image data (bytes or PIL Image)
        target_size: Target size for resizing (tuple)
        
    Returns:
        Preprocessed image array ready for model input
    """
    try:
        # Convert bytes to PIL Image if needed
        if isinstance(image_data, bytes):
            img = Image.open(io.BytesIO(image_data))
        else:
            img = image_data
        
        # Resize image to target size
        img = img.resize(target_size)
        img_array = np.array(img)
        
        # Handle different color channels
        if len(img_array.shape) == 2:  # Grayscale
            img_array = np.stack((img_array,) * 3, axis=-1)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = img_array[:, :, :3]  # Take only RGB channels
        
        # Normalize pixel values to [0, 1]
        img_normalized = img_array / 255.0
        
        # Add batch dimension
        img_reshaped = img_normalized.reshape(1, target_size[0], target_size[1], 3)
        
        return img_reshaped
    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}")

# ===== API ROUTES =====

@app.route('/')
def home():
    """Home route with API documentation"""
    return jsonify({
        "status": "running",
        "service": "MediSkin API",
        "version": "1.0.0",
        "endpoints": {
            "web_ui": "/ui",
            "skin_prediction": "/predict",
            "skin_prediction_api": "/api/predict",
            "skin_condition_details": "/skin-condition/<code>",
            "medicines_list": "/medicine",
            "medicine_search": "/medicine/search",
            "diseases_list": "/diseases",
            "disease_search": "/diseases/search"
        }
    })

@app.route('/ui')
def web_ui():
    """Web UI route"""
    return render_template('index.html')

# ===== SKIN DISEASE PREDICTION ROUTES =====

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload from web UI for skin disease prediction"""
    # Validate request
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Process file and make prediction
    file_path = None
    try:
        # Create uploads directory if it doesn't exist
        os.makedirs('uploads', exist_ok=True)
        
        # Save uploaded file temporarily
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        
        # Read file
        with open(file_path, 'rb') as f:
            image_data = f.read()
        
        # Process and get prediction
        result = process_and_predict(image_data)
        result = enrich_prediction_with_details(result)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    finally:
        # Clean up - remove uploaded file
        if file_path and os.path.exists(file_path):
            os.remove(file_path)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for prediction (for mobile apps)"""
    try:
        # Validate request
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Process base64 encoded image
        try:
            image_data = base64.b64decode(data['image'])
        except Exception as e:
            return jsonify({'error': f'Invalid base64 image data: {str(e)}'}), 400
        
        # Make prediction
        result = process_and_predict(image_data)
        result = enrich_prediction_with_details(result)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def process_and_predict(image_data):
    """Process image data and generate prediction
    
    Args:
        image_data: Raw image data bytes
        
    Returns:
        Dictionary containing prediction results
    """
    # Ensure model is loaded
    global skin_model
    if skin_model is None:
        load_models()
    
    # Preprocess the image and predict
    processed_image = preprocess_image(image_data)
    predictions = skin_model.predict(processed_image)
    
    # Get predicted class and confidence
    predicted_class_id = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]) * 100)
    
    # Get all probability scores
    all_probabilities = [float(prob * 100) for prob in predictions[0]]
    
    # Get class information
    class_info = SKIN_CLASS_LABELS[str(predicted_class_id)]
    
    # Build result dictionary
    result = {
        'class_id': predicted_class_id,
        'class_code': class_info['code'],
        'class_name': class_info['name'],
        'confidence': round(confidence, 2),  # Round to 2 decimal places
        'all_probabilities': [{
            'class_id': i,
            'class_code': SKIN_CLASS_LABELS[str(i)]['code'],
            'class_name': SKIN_CLASS_LABELS[str(i)]['name'],
            'probability': round(all_probabilities[i], 2)
        } for i in range(len(all_probabilities))]
    }
    
    return result

def enrich_prediction_with_details(prediction_result):
    """Add detailed information about predicted skin condition
    
    Args:
        prediction_result: Basic prediction result dictionary
        
    Returns:
        Prediction result enhanced with condition details
    """
    try:
        class_code = prediction_result['class_code']
        
        # Try to get condition details from database
        try:
            condition = skin_conditions_collection.find_one({"code": class_code})
            if condition:
                # Remove MongoDB _id field which is not JSON serializable
                if '_id' in condition:
                    condition.pop('_id')
                prediction_result['details'] = condition
                return prediction_result
        except Exception as db_error:
            print(f"Database lookup failed: {str(db_error)}")
        
        # Fall back to hard-coded data if database failed
        for condition_id, condition_data in SKIN_CONDITIONS_DATA["skin_conditions"].items():
            if condition_data["code"] == class_code:
                prediction_result['details'] = condition_data
                return prediction_result
        
        # If still no details found
        prediction_result['details'] = {"note": "Detailed information unavailable"}
        return prediction_result
        
    except Exception as e:
        prediction_result['details'] = {"error": f"Failed to retrieve details: {str(e)}"}
        return prediction_result

@app.route('/skin-condition/<code>', methods=['GET'])
def get_skin_condition(code):
    """Get detailed information about a specific skin condition
    
    Args:
        code: Skin condition code (e.g., 'mel', 'bcc')
    """
    try:
        # Try database first
        try:
            condition = skin_conditions_collection.find_one({"code": code})
            if condition:
                if '_id' in condition:
                    condition.pop('_id')
                return jsonify(condition), 200
        except Exception as db_error:
            print(f"Database lookup failed: {str(db_error)}")
        
        # Fall back to hard-coded data
        for condition_id, data in SKIN_CONDITIONS_DATA["skin_conditions"].items():
            if data["code"] == code:
                return jsonify(data), 200
        
        # If not found anywhere
        return jsonify({"error": f"Skin condition with code '{code}' not found"}), 404
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===== MEDICINE DATABASE ROUTES =====

@app.route('/medicine', methods=['GET'])
def get_medicines_list():
    """Get paginated list of medicines"""
    # Parse query parameters
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 50))
    sort_by = request.args.get('sort', 'name')
    sort_direction = 1 if request.args.get('direction', 'asc').lower() == 'asc' else -1

    # Define required fields to include in response
    required_fields = {
        'name': 1,
        'price': 1,
        'manufacturer_name': 1,
        'type': 1,
        'pack_size_label': 1,
        'short_composition1': 1,
        'salt_composition': 1,
        'medicine_desc': 1
    }

    try:
        # Build query with field existence requirements
        query = {
            "$and": [
                {field: {"$exists": True, "$ne": ""}} 
                for field in required_fields.keys()
            ]
        }

        # Get total count for pagination
        total_count = medicine_collection.count_documents(query)
        skip = (page - 1) * per_page

        # Get paginated and sorted data
        cursor = medicine_collection.find(query, required_fields)\
            .sort([(sort_by, sort_direction)])\
            .skip(skip)\
            .limit(per_page)

        # Format response
        medicines = list(cursor)
        formatted = {
            "data": [
                {
                    "name": m.get("name"),
                    "type": m.get("type"),
                    "manufacturer": m.get("manufacturer_name"),
                    "price": m.get("price")
                }
                for m in medicines
                if all(m.get(field) for field in required_fields)
            ],
            "pagination": {
                "current_page": page,
                "items_per_page": per_page,
                "total_items": total_count,
                "total_pages": (total_count + per_page - 1) // per_page
            }
        }

        return jsonify(formatted), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/medicine/search', methods=['GET'])
def search_medicine():
    """Search for medicines by name"""
    # Parse query parameters
    query = request.args.get('query', '').strip()
    mode = request.args.get('mode', 'auto')  # 'auto', 'suggestions', or 'details'
    
    if not query:
        return jsonify({"error": "Search query is required"}), 400
    
    try:
        # Default to suggestions for very short queries in auto mode
        if len(query) < 3 and mode == 'auto':
            mode = 'suggestions'
        
        # Try exact match first
        exact_match = medicine_collection.find_one({"name": {"$regex": f"^{query}$", "$options": "i"}})
        
        # Handle exact match for details or auto mode
        if (exact_match and mode == 'details') or (exact_match and mode == 'auto'):
            formatted = format_medicine_details(exact_match)
            return jsonify(formatted), 200
        
        # For suggestions or when no exact match in auto mode
        # Find medicines containing the query string
        results = medicine_collection.find(
            {"name": {"$regex": f".*{query}.*", "$options": "i"}},
            {"name": 1, "type": 1, "manufacturer_name": 1, "_id": 1}
        ).limit(10)
        
        suggestions = list(results)
        
        # Handle no results case
        if not suggestions:
            return jsonify({
                "result_type": "no_results",
                "message": "No matches found for your search term",
                "suggestions": []
            }), 404
        
        # If only one suggestion in auto mode, return its details
        if len(suggestions) == 1 and mode == 'auto':
            medicine = medicine_collection.find_one({"_id": suggestions[0].get("_id")})
            formatted = format_medicine_details(medicine)
            return jsonify(formatted), 200
        
        # Otherwise return suggestions list
        formatted_suggestions = [
            {
                "name": s.get("name"),
                "type": s.get("type"),
                "manufacturer": s.get("manufacturer_name")
            } for s in suggestions
        ]
        
        return jsonify({
            "data": formatted_suggestions
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def format_medicine_details(medicine):
    """Format medicine document for detailed response"""
    return {
        "result_type": "details",
        "details": {
            "name": medicine.get("name", "Unknown Medicine"),
            "price": medicine.get("price", 0.0),
            "is_discontinued": medicine.get("is_discontinued", False),
            "manufacturer_name": medicine.get("manufacturer_name", "Unknown"),
            "type": medicine.get("type", "Unknown"),
            "pack_size_label": medicine.get("pack_size_label", "Unknown"),
            "short_composition1": medicine.get("short_composition1", ""),
            "short_composition2": medicine.get("short_composition2", ""),
            "salt_composition": medicine.get("salt_composition", ""),
            "medicine_desc": medicine.get("medicine_desc", ""),
            "side_effects": medicine.get("side_effects", ""),
            "drug_interactions": medicine.get("drug_interactions", {})
        }
    }

# ===== DISEASE DATABASE ROUTES =====

@app.route('/diseases', methods=['GET'])
def get_diseases_list():
    """Get paginated list of diseases"""
    try:
        # Parse pagination parameters
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 50))
        
        # Calculate pagination values
        skip = (page - 1) * per_page
        total_count = disease_collection.count_documents({})
        
        # Get paginated diseases
        diseases = list(disease_collection.find({}, {"Disease": 1}).skip(skip).limit(per_page))
        
        # Format response
        formatted = [{"disease": d.get("Disease", "Unknown Disease")} for d in diseases]

        return jsonify({
            "data": formatted,
            "pagination": {
                "current_page": page,
                "items_per_page": per_page,
                "total_items": total_count,
                "total_pages": (total_count + per_page - 1) // per_page
            }
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/diseases/search', methods=['GET'])
def search_disease():
    """Search for diseases by name"""
    # Parse query parameters
    query = request.args.get('query', '').strip()
    mode = request.args.get('mode', 'auto')  # 'auto', 'suggestions', or 'details'
    
    if not query:
        return jsonify({"error": "Search query is required"}), 400
    
    try:
        # Default to suggestions for very short queries in auto mode
        if len(query) < 3 and mode == 'auto':
            mode = 'suggestions'
        
        # Try exact match first
        exact_match = disease_collection.find_one({"Disease": {"$regex": f"^{query}$", "$options": "i"}})
        
        # Handle exact match for details or auto mode
        if (exact_match and mode == 'details') or (exact_match and mode == 'auto'):
            formatted = format_disease_details(exact_match)
            return jsonify(formatted), 200
        
        # For suggestions or when no exact match in auto mode
        # Find diseases containing the query string
        results = disease_collection.find(
            {"Disease": {"$regex": f".*{query}.*", "$options": "i"}},
            {"Disease": 1, "_id": 1}
        ).limit(10)
        
        suggestions = list(results)
        
        # Handle no results case
        if not suggestions:
            return jsonify({
                "result_type": "no_results",
                "message": "No matches found for your search term",
                "suggestions": []
            }), 404
        
        # If only one suggestion in auto mode, return its details
        if len(suggestions) == 1 and mode == 'auto':
            disease = disease_collection.find_one({"_id": suggestions[0].get("_id")})
            formatted = format_disease_details(disease)
            return jsonify(formatted), 200
        
        # Otherwise return suggestions list
        formatted_suggestions = [{"name": s.get("Disease")} for s in suggestions]
        
        return jsonify({
            "data": formatted_suggestions
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def format_disease_details(disease):
    """Format disease document for detailed response"""
    # Clean empty strings from symptoms and precautions
    symptoms = [
        disease.get(f"Symptom_{i}", "") 
        for i in range(1, 7)
        if disease.get(f"Symptom_{i}", "").strip()
    ]
    
    precautions = [
        disease.get(f"Precaution_{i}", "")
        for i in range(1, 5)
        if disease.get(f"Precaution_{i}", "").strip()
    ]
    
    return {
        "result_type": "details",
        "details": {
            "disease": disease.get("Disease", "Unknown Disease"),
            "description": disease.get("Description", ""),
            "symptoms": symptoms,
            "precautions": precautions
        }
    }

# ===== ERROR HANDLERS =====

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Resource not found"}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({"error": "Internal server error"}), 500

# ===== APPLICATION ENTRY POINT =====

if __name__ == '__main__':
    # Load models at startup
    load_models()
    # Run app on local network
    app.run(host='0.0.0.0', port=5000, debug=True)