
from flask import Flask, request, jsonify
import pickle
import os
import sys
import numpy as np

# Try to import catboost
try:
    import catboost
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not installed.")

app = Flask(__name__)

# --- Configuration & Data ---

# Feature Names (Target Order from Model)
MODEL_FEATURES = ['R', 'I', 'A', 'S', 'E', 'C', 
'Current Qualification (please be specific here) For eg : (B.Com, B.Sc C.S, B.Tech)_12Th Passout Yr 2019', 
'Current Qualification (please be specific here) For eg : (B.Com, B.Sc C.S, B.Tech)_B Pharm', 
'Current Qualification (please be specific here) For eg : (B.Com, B.Sc C.S, B.Tech)_B Pharmacy Third Year', 
'Current Qualification (please be specific here) For eg : (B.Com, B.Sc C.S, B.Tech)_B Tech Mechanical',
'Current Qualification (please be specific here) For eg : (B.Com, B.Sc C.S, B.Tech)_B. A',
'Current Qualification (please be specific here) For eg : (B.Com, B.Sc C.S, B.Tech)_B. Architecture', 
'Current Qualification (please be specific here) For eg : (B.Com, B.Sc C.S, B.Tech)_B. E', 
'Current Qualification (please be specific here) For eg : (B.Com, B.Sc C.S, B.Tech)_B. Sc C.S',
'Current Qualification (please be specific here) For eg : (B.Com, B.Sc C.S, B.Tech)_B.A',
'Current Qualification (please be specific here) For eg : (B.Com, B.Sc C.S, B.Tech)_B.B.A', 
'Current Qualification (please be specific here) For eg : (B.Com, B.Sc C.S, B.Tech)_B.Com', 
'Current Qualification (please be specific here) For eg : (B.Com, B.Sc C.S, B.Tech)_B.E', 
'Current Qualification (please be specific here) For eg : (B.Com, B.Sc C.S, B.Tech)_B.E Computer Science',
'Current Qualification (please be specific here) For eg : (B.Com, B.Sc C.S, B.Tech)_B.E Mechanical', 
'Current Qualification (please be specific here) For eg : (B.Com, B.Sc C.S, B.Tech)_B.Pharm', 
'Current Qualification (please be specific here) For eg : (B.Com, B.Sc C.S, B.Tech)_B.Sc', 
'Current Qualification (please be specific here) For eg : (B.Com, B.Sc C.S, B.Tech)_B.Sc Biotechnology',
'Current Qualification (please be specific here) For eg : (B.Com, B.Sc C.S, B.Tech)_B.Sc C.S', 
'Current Qualification (please be specific here) For eg : (B.Com, B.Sc C.S, B.Tech)_B.Sc Computer Science', 
'Current Qualification (please be specific here) For eg : (B.Com, B.Sc C.S, B.Tech)_B.Sc IT', 
'Current Qualification (please be specific here) For eg : (B.Com, B.Sc C.S, B.Tech)_B.Tech',
'Current Qualification (please be specific here) For eg : (B.Com, B.Sc C.S, B.Tech)_B.Tech Mechanical',
'Current Qualification (please be specific here) For eg : (B.Com, B.Sc C.S, B.Tech)_B.Tech Mechatronics Engineering', 
'Current Qualification (please be specific here) For eg : (B.Com, B.Sc C.S, B.Tech)_Ba Psychology',
'Current Qualification (please be specific here) For eg : (B.Com, B.Sc C.S, B.Tech)_Bachelor Of Engineering',
'Current Qualification (please be specific here) For eg : (B.Com, B.Sc C.S, B.Tech)_Bca',
'Current Qualification (please be specific here) For eg : (B.Com, B.Sc C.S, B.Tech)_Bcs', 
'Current Qualification (please be specific here) For eg : (B.Com, B.Sc C.S, B.Tech)_Bhms',
'Current Qualification (please be specific here) For eg : (B.Com, B.Sc C.S, B.Tech)_Bms', 
'Current Qualification (please be specific here) For eg : (B.Com, B.Sc C.S, B.Tech)_Bpt',
'Current Qualification (please be specific here) For eg : (B.Com, B.Sc C.S, B.Tech)_Bsc Computer Science', 
'Current Qualification (please be specific here) For eg : (B.Com, B.Sc C.S, B.Tech)_Bsc In Computer Science',
'Current Qualification (please be specific here) For eg : (B.Com, B.Sc C.S, B.Tech)_Bsc It', 
'Current Qualification (please be specific here) For eg : (B.Com, B.Sc C.S, B.Tech)_Bsw', 
'Current Qualification (please be specific here) For eg : (B.Com, B.Sc C.S, B.Tech)_Graduate', 
'Current Qualification (please be specific here) For eg : (B.Com, B.Sc C.S, B.Tech)_Intermediate',
'Current Qualification (please be specific here) For eg : (B.Com, B.Sc C.S, B.Tech)_Llb', 
'Current Qualification (please be specific here) For eg : (B.Com, B.Sc C.S, B.Tech)_M.Sc',
'Current Qualification (please be specific here) For eg : (B.Com, B.Sc C.S, B.Tech)_M.Sc C.S',
'Current Qualification (please be specific here) For eg : (B.Com, B.Sc C.S, B.Tech)_M.Sc Data Science & Big Data Analytics', 
'Current Qualification (please be specific here) For eg : (B.Com, B.Sc C.S, B.Tech)_M.Sc Dsbda', 
'Current Qualification (please be specific here) For eg : (B.Com, B.Sc C.S, B.Tech)_M.Sc. Data Science And Big Data Analytics', 
'Current Qualification (please be specific here) For eg : (B.Com, B.Sc C.S, B.Tech)_MBA', 
'Current Qualification (please be specific here) For eg : (B.Com, B.Sc C.S, B.Tech)_MSW',
'Current Qualification (please be specific here) For eg : (B.Com, B.Sc C.S, B.Tech)_Masters In Media And Journalism', 
'Current Qualification (please be specific here) For eg : (B.Com, B.Sc C.S, B.Tech)_Mbbs', 
'Current Qualification (please be specific here) For eg : (B.Com, B.Sc C.S, B.Tech)_Msc Biotech',
'Current Qualification (please be specific here) For eg : (B.Com, B.Sc C.S, B.Tech)_Msc Computer Science', 
'Current Qualification (please be specific here) For eg : (B.Com, B.Sc C.S, B.Tech)_Msc Data Science And Big Data Analytics',
'Current Qualification (please be specific here) For eg : (B.Com, B.Sc C.S, B.Tech)_Msc Dsbda',
'Current Qualification (please be specific here) For eg : (B.Com, B.Sc C.S, B.Tech)_Msc Stats', 
'Current Qualification (please be specific here) For eg : (B.Com, B.Sc C.S, B.Tech)_PGDM',
'Current Qualification (please be specific here) For eg : (B.Com, B.Sc C.S, B.Tech)_Pursuing Pgdm', 
'Current Qualification (please be specific here) For eg : (B.Com, B.Sc C.S, B.Tech)_Textile Digener', 
'Current Qualification (please be specific here) For eg : (B.Com, B.Sc C.S, B.Tech)_Tyba',
'I like to work on cars_yes', 'I like to build things_yes', 'I like to take care of animals_yes',
'I like putting things together or assembling things_yes', 'I like to cook_yes', 'I am a practical person_yes',
'I like working outdoors_yes', 'I like to do puzzles_yes', 'I like to do experiments_yes', 'I enjoy science_yes',
'I enjoy trying to figure out how things work_yes', 'I like to analyze things (problems/situations)_yes',
'I like working with numbers or charts_yes', "I'm good at math_yes", 'I am good at working independently_yes', 
'I like to read about art and music_yes', 'I enjoy creative writing_yes', 'I am a creative person_yes', 'I like to play instruments or sing_yes',
'I like acting in plays_yes', 'I like to draw_yes', 'I like to work in teams_yes', 'I like to teach or train people_yes',
'I like trying to help people solve their problems_yes', 'I am interested in healing people_yes', 'I enjoy learning about other cultures_yes',
'I like to get into discussions about issues_yes', 'I like helping people_yes', 'I am an ambitious person,I set goals for myself_yes', 
'I like to try to influence or persuade people_yes', 'I like selling things_yes', 'I am quick to take on new responsibilities_yes',
'I would like to start my own business_yes', 'I like to lead_yes', 'I like to give speeches_yes', 'I like to organize things,(files, desks/offices)_yes',
'I like to have clear instructions to follow_yes', "I wouldn't mind working 8 hours per day in an office_yes", 'I pay attention to details_yes',
'I like to do filing or typing_yes', 'I am good at keeping records of my work_yes', 'I would like to work in an office_yes',
'class_C', 'class_E', 'class_I', 'class_R', 'class_S']

# Inverted mapping: Category -> List of Track IDs
CATEGORY_TO_TRACK_IDS = {
    "AI/ML": [5, 6, 20, 21],
    "Data": [3, 18, 19],
    "Software": [1, 2, 4, 17, 15, 13, 16, 9, 22, 14],
    "Security": [8],
    "Game": [10, 23],
    "Mobile": [11, 12],
    "UX Design": [24],
    "Blockchain": [7]
}

# --- Model Loading ---

model_cb = None

def load_models():
    global model_cb
    try:
        if os.path.exists("CatBoost_Best_Model.cbm") and CATBOOST_AVAILABLE:
            model_cb = catboost.CatBoostClassifier()
            model_cb.load_model("CatBoost_Best_Model.cbm")
            print("Loaded CatBoost_Best_Model.cbm")
    except Exception as e:
        print(f"Error loading models: {e}")

load_models()

# --- Helpers ---

# Question ID -> (Feature Name in Model, RIASEC Category)
QUESTION_MAPPING = {
    # Page 1
    "page1_q1": ("I like to work on cars_yes", "R"),
    "page1_q2": ("I like to do puzzles_yes", "I"),
    "page1_q3": ("I am good at working independently_yes", "I"), # Assumed I
    "page1_q4": ("I like to work in teams_yes", "S"), # Assumed S
    "page1_q5": ("I am an ambitious person,I set goals for myself_yes", "E"), # Note: Comma no space in model
    "page1_q6": ("I like to organize things,(files, desks/offices)_yes", "C"), # Note: Comma in model
    "page1_q7": ("I like to build things_yes", "R"),
    "page1_q8": ("I like to read about art and music_yes", "A"),
    "page1_q9": ("I like to have clear instructions to follow_yes", "C"),
    "page1_q10": ("I like to try to influence or persuade people_yes", "E"),
    "page1_q11": ("I like to do experiments_yes", "I"),
    "page1_q12": ("I like to teach or train people_yes", "S"),
    "page1_q13": ("I like trying to help people solve their problems_yes", "S"),
    "page1_q14": ("I like to take care of animals_yes", "R"), # Often R or S. Putting in R based on "Nature" aspect common in R

    # Page 2
    "page2_q1": ("I wouldn't mind working 8 hours per day in an office_yes", "C"), # smart quote
    "page2_q2": ("I like selling things_yes", "E"),
    "page2_q3": ("I enjoy creative writing_yes", "A"),
    "page2_q4": ("I enjoy science_yes", "I"),
    "page2_q5": ("I am quick to take on new responsibilities_yes", "E"),
    "page2_q6": ("I am interested in healing people_yes", "S"),
    "page2_q7": ("I enjoy trying to figure out how things work_yes", "I"),
    "page2_q8": ("I like putting things together or assembling things_yes", "R"),
    "page2_q9": ("I am a creative person_yes", "A"),
    "page2_q10": ("I pay attention to details_yes", "C"),
    "page2_q11": ("I like to do filing or typing_yes", "C"),
    "page2_q12": ("I like to analyze things (problems/situations)_yes", "I"),
    "page2_q13": ("I like to play instruments or sing_yes", "A"),
    "page2_q14": ("I enjoy learning about other cultures_yes", "S"), # Assumed S

    # Page 3
    "page3_q1": ("I would like to start my own business_yes", "E"),
    "page3_q2": ("I like to cook_yes", "R"), # often R
    "page3_q3": ("I like acting in plays_yes", "A"),
    "page3_q4": ("I am a practical person_yes", "R"),
    "page3_q5": ("I like working with numbers or charts_yes", "I"), # or C? usually I/C.
    "page3_q6": ("I like to get into discussions about issues_yes", "S"), # Assumed S
    "page3_q7": ("I am good at keeping records of my work_yes", "C"),
    "page3_q8": ("I like to lead_yes", "E"),
    "page3_q9": ("I like working outdoors_yes", "R"),
    "page3_q10": ("I would like to work in an office_yes", "C"),
    "page3_q11": ("I'm good at math_yes", "I"), # smart quote
    "page3_q12": ("I like helping people_yes", "S"),
    "page3_q13": ("I like to draw_yes", "A"),
    "page3_q14": ("I like to give speeches_yes", "E"),
}

MODEL_FEATURES_SET = set(MODEL_FEATURES) # for faster lookup
FEATURE_INDEX = {f: i for i, f in enumerate(MODEL_FEATURES)} # O(1) lookup

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        response = jsonify({"status": "ok"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,ngrok-skip-browser-warning')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response

    try:
        data = request.json
        answers = data.get('answers', {})
        
        # Initialize input vector with 0s
        input_vector = [0] * len(MODEL_FEATURES)
        
        # Initialize RIASEC counters
        riasec_counts = {"R": 0, "I": 0, "A": 0, "S": 0, "E": 0, "C": 0}
        
        # Process each answer
        for q_id, val_str in answers.items():
            if q_id not in QUESTION_MAPPING:
                continue
                
            feature_name, riasec_cat = QUESTION_MAPPING[q_id]
            
            # 1. Map Answer to Binary (agree = 1, disagree = 0)
            val_norm = str(val_str).lower().strip()
            if val_norm == 'agree':
                is_positive = 1
            else:
                is_positive = 0
            
            # 2. Update Feature Vector
            idx = FEATURE_INDEX.get(feature_name)
            if idx is not None:
                input_vector[idx] = is_positive
            
            # 3. Update RIASEC Score
            if is_positive:
                if riasec_cat in riasec_counts:
                    riasec_counts[riasec_cat] += 1
        
        # 4. Fill RIASEC Features (Indices 0-5 assumed: R, I, A, S, E, C)
        # Verify indices: 
        # MODEL_FEATURES[0] is 'R'
        # MODEL_FEATURES[1] is 'I' ...
        input_vector[0] = riasec_counts["R"]
        input_vector[1] = riasec_counts["I"]
        input_vector[2] = riasec_counts["A"]
        input_vector[3] = riasec_counts["S"]
        input_vector[4] = riasec_counts["E"]
        input_vector[5] = riasec_counts["C"]
        
        print("----- NEW REQUEST -----")
        print(f"Answers Keys: {list(answers.keys())}")
        print(f"Sample Answer Values: {list(answers.values())[:5]}") 
        print(f"Received {len(answers)} answers.")
        print(f"Calculated RIASEC: {riasec_counts}")
        
        recommended_ids = []
        
        if model_cb:
            # Predict Probabilities (CatBoost needs 2D input)
            proba = model_cb.predict_proba([input_vector])[0]  # 2D input, extract first row
            
            # Classes order
            classes = model_cb.classes_
            
            # Get Top 1 Category
            top_idx = np.argmax(proba)  # Simple and clean
            top_category = classes[top_idx]
            
            print(f"Probabilities: {proba}")
            print(f"Top Category: {top_category}")
            
            # Map Category to Tracks
            cat_clean = str(top_category).strip()
            recommended_ids = CATEGORY_TO_TRACK_IDS.get(cat_clean, [])
            
            if not recommended_ids:
                print(f"Warning: No tracks found for category '{cat_clean}'")
            else:
                print(f"Final Recommended IDs: {recommended_ids}")
            print("----- END REQUEST -----\n")
            
            # Deduplicate and sort
            recommended_ids = sorted(list(set(recommended_ids)))
            
        else:
            print("Model not loaded, using Mock")
            recommended_ids = [1, 2, 5]

        # Return formatted response
        return jsonify({
            "recommendedTracks": recommended_ids
        })

    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def health():
    return jsonify({"status": "running", "models_loaded": (model_cb is not None)})

if __name__ == '__main__':
    port = 5000
    print(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port)
