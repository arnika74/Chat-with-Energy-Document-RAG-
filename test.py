from google import genai
# import google.generativeai as genai

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
# client = genai.Client(api_key= "AIzaSyA7Fp6Ywuqrazyx1cr0RgUMbGTKCtq_lgA")
client = genai.Client(api_key= "AIzaSyAx8w87lKy6psIf54RykhCAWyWvMe5H2ms")


response = client.models.generate_content(
    model="gemini-2.5-flash", contents="Explain how AI works in a few words"
)
print(response.text)
