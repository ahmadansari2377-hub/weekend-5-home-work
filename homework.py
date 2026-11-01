import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# load_dotenv() reads the .env file and puts every KEY=value line into the
# environment, so the API key never has to be written inside the code.
load_dotenv()

# os.getenv() reads the key back out of the environment
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise SystemExit("GOOGLE_API_KEY not found. Add it to the .env file.")

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # free-tier model; gemini-3-flash-preview also works
    google_api_key=api_key,
)

response = model.invoke("Hello how are you?")

print("Full response:")
print(response)
print("\nAnswer only:")
print(response.text)
