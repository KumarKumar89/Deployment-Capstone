import os
from dotenv import load_dotenv
import google.generativeai as genai

def test_gemini_api_connection():
    """Tests the connection to the Google Gemini API."""

    # Load environment variables
    load_dotenv()

    # Configure Google Gemini
    api_key = os.getenv("GOOGLE_API_KEY")

    assert api_key, "GOOGLE_API_KEY environment variable not found."

    try:
        genai.configure(api_key=api_key)

        # List available models
        models = [m.name for m in genai.list_models()]
        assert models, "No models found in the Gemini API."
        print("\nAvailable models:")
        for model_name in models:
            print(f"- {model_name}")

        # Try to use the model
        model = genai.GenerativeModel('gemini-2.0-flash-lite-001')
        response = model.generate_content("Say hello!")

        assert response.text, "Gemini API did not return a response."
        print("\nGemini Response:", response.text)
        print("\nAPI is working correctly!")

    except Exception as e:
        assert False, f"Error occurred: {str(e)}"

# Example of how to run the function as a test
if __name__ == "__main__":
    test_gemini_api_connection()