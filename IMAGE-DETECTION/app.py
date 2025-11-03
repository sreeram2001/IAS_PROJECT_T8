import os
import json
import base64
from io import BytesIO
from flask import Flask, render_template, request, jsonify
from PIL import Image
from google import genai
from google.genai import types
from google.genai.errors import APIError


API_KEY = "AIzaSyAAza7WAkrUIqwfZSEckja4tCU_GCe4W1E" 
MODEL_NAME = "gemini-2.5-flash"

SYSTEM_INSTRUCTION_PROMPT = """
You are a top-tier forensic linguist and stylistic analysis expert specializing in machine-generated text. Your task is to analyze the text and layout within the provided image (a letter or document) to determine the likelihood it was drafted by a Large Language Model (LLM) like ChatGPT, GPT-4, or other writing bots.

**Your primary objective is to determine the definitive verdict of origin: 'AI Generated' or 'Likely Human'.**

Your analysis must critically inspect the image for the following linguistic and rhetorical artifacts:

1. **Rhetorical Patterns:** Look for generic, corporate, or overly formal phrases that lack specific human voice or context (e.g., "drove results," "synergistic," "demonstrated strong skills" without detail).
2. **Lexical Predictability:** Identify high-frequency or predictable vocabulary (low perplexity) and a lack of 'burstiness' (sentence length and structure variation).
3. **Structure & Flow:** Check for repetitive paragraph structure, overly perfect grammar, or a mechanical, predictable organization that LLMs favor.
4. **Content Factual Consistency:** Note any dates, names, or titles that appear generic, fictional, or inconsistent (e.g., a "Software Engineer" using vague, non-technical language).
5. **Visual Clutter (If Applicable):** Assess if the visual layout (fonts, spacing, borders) is overly simplistic or generated with minimal stylistic flair.

Provide your final assessment in a JSON object using the following schema. After the JSON, provide a detailed, line-by-line explanation of your linguistic findings in a clear, narrative markdown format. Do not use the JSON in the explanation, only refer to its contents.
"""

RESPONSE_SCHEMA = {
  "type": "OBJECT",
  "properties": {
    "verdict": {
      "type": "STRING",
      "description": "The final verdict: 'AI Generated', 'Likely Human', or 'Inconclusive'."
    },
    "likelihoodScore": {
      "type": "NUMBER",
      "description": "A score from 0.0 to 1.0 indicating the likelihood of AI generation (1.0 is 100% AI)."
    },
    "keyFindings": {
      "type": "ARRAY",
      "description": "A list of 2-3 specific visual or textural artifacts found.",
      "items": { "type": "STRING" }
    }
  },
  "required": ["verdict", "likelihoodScore", "keyFindings"]
}

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyzeImageArtifact():
    
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided."}), 400
    
    image_file = request.files['image']
    
    if image_file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    try:
        image_bytes = image_file.read()
        image = Image.open(BytesIO(image_bytes))

        client = genai.Client(api_key=API_KEY)
        user_prompt = "Analyze this image for generative AI artifacts and provide a verdict as requested in the system instruction. Be extremely critical and detailed in your visual analysis."

        generation_config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
        )
        
        system_content_part = types.Content(parts=[types.Part(text=SYSTEM_INSTRUCTION_PROMPT)])
        
        api_contents = [
            system_content_part,
            image,
            user_prompt
        ]

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=api_contents,
            config=generation_config,
        )

        if not response.text:
            return jsonify({
                "error": "Analysis Failed",
                "message": "The Gemini API returned an empty response."
            }), 500

        json_start = response.text.find('{')
        json_end = response.text.rfind('}') + 1
        
        if json_start != -1 and json_end != -1 and json_end > json_start:
            json_string = response.text[json_start:json_end]
            json_string = json_string.replace('```json', '').replace('```', '').strip()
            
            result_data = json.loads(json_string)
            
            raw_explanation = response.text[json_end:].strip()
        else:
            result_data = {}
            raw_explanation = response.text

        report = {
            "verdict": result_data.get("verdict", "Inconclusive"),
            "likelihoodScore": result_data.get("likelihoodScore", 0.0),
            "keyFindings": result_data.get("keyFindings", ["Could not extract structured findings."]),
            "detailedExplanation": raw_explanation,
            "success": True
        }
        
        return jsonify(report)

    except APIError as e:
        return jsonify({
            "error": "Gemini API Error",
            "message": str(e),
            "detail": "Please ensure your API key is valid and the model is accessible."
        }), 500
    except Exception as e:
        return jsonify({
            "error": "Internal Server Error",
            "message": f"An unexpected error occurred: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
