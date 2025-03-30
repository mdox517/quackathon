from flask import Flask, render_template, request, abort
import google.generativeai as genai
from food_detector import analyze_food 
from dotenv import load_dotenv
import os
from datetime import datetime
import csv

load_dotenv()

genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))  
health_model = genai.GenerativeModel('gemini-2.0-flash')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024 

def get_food_analysis(foods):
    print("Foods passed to get_food_analysis:", foods)

    food_names = []
    for f in foods:
        if isinstance(f, dict) and "food" in f:
            food_names.append(f["food"])
        else:
            print("Invalid entry in foods:", f)

    if not food_names:
        return "No valid food names found in detection."

    prompt = f"""
    Analyze these foods as a meal: {', '.join(food_names)}. Provide:
    - 3 health benefits (bullet points)
    - 2 potential risks (bullet points)
    - 1 healthier alternative
    - Overall nutrition score (1-5)
    Format with 🦆 Attila's rating at the top.
    """
    
    try:
        response = health_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Couldn't analyze: {str(e)}"

def save_to_history(foods, analysis):
    """Save results to CSV"""
    with open('meal_history.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        food_names = [food['food'] for food in foods]
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            ", ".join(food_names),
            analysis[:200] 
        ])

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No file uploaded", 400
            
        uploaded_file = request.files['image']
        if uploaded_file.filename == '':
            return "No file selected", 400


        image_path = os.path.join('static', 'upload.jpg')
        uploaded_file.save(image_path)

        try:
            detected_foods = analyze_food(image_path)
            
            analysis = get_food_analysis(detected_foods)
            
            save_to_history(detected_foods, analysis)
            
            return render_template('index.html',
                               foods=detected_foods,
                               analysis=analysis,
                               image_path=image_path)
            
        except Exception as e:
            return f"Error processing image: {str(e)}", 500

    return render_template('index.html')

@app.errorhandler(413)
def too_large(e):
    return "File too large (max 8MB)", 413

if __name__ == '__main__':
    if not os.path.exists('meal_history.csv'):
        with open('meal_history.csv', 'w') as f:
            f.write("timestamp,foods,analysis\n")
    
    app.run(debug=True)