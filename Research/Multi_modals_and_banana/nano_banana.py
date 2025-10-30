
from google import genai
from google.genai import types

client = genai.Client()

response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents='high',
    config=types.GenerateContentConfig(
        system_instruction='I say high, you say',
        max_output_tokens=33333,
        temperature=0.7,
    ),
)
print(response.text)


from google import genai
from PIL import Image
from io import BytesIO

prompt = "A high-resolution, photorealistic image of the DoDo bird flying over a calm jungle"

response = client.models.generate_content(
    model="gemini-2.5-flash-image-preview",
    contents=[ prompt ],
    config=None  # You may omit or pass a config object if required
)

# Process parts
for part in response.candidates[0].content.parts:
    if part.text is not None:
        print("Text part:", part.text)
    elif part.inline_data is not None:
        image_bytes = part.inline_data.data
        img = Image.open(BytesIO(image_bytes))
        filename = "dodo_bird.png"
        img.save(filename)
        print(f"Saved image as {filename}")
