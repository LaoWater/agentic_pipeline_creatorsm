from google import genai
from PIL import Image
from io import BytesIO
import os

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_API_KEY_HERE")
MODEL_ID = "gemini-2.5-flash-image-preview"   # Nano Banana preview model
STARTING_IMAGE_PATH = "cm_quantum_chip.png"
OUTPUT_DIR = "output"

PROMPT = (
    "Enhance this photo with a cinematic effect, "
    "Put text `Creators Multiverse` in a beautiful caption with thin, clear letters (as this is the entity we are designing for.)"
    "Make more strings coming out/radiant out of the chip"


)

# -----------------------------------------------------------------------------
# SETUP
# -----------------------------------------------------------------------------
if not os.path.exists(STARTING_IMAGE_PATH):
    raise FileNotFoundError(f"Starting image not found: {STARTING_IMAGE_PATH}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

client = genai.Client(api_key=API_KEY)

# Load the input image
starting_image = Image.open(STARTING_IMAGE_PATH)

print("[*] Sending prompt to Nano Banana...")
print(f"    Model: {MODEL_ID}")
print(f"    Prompt: {PROMPT}")

# -----------------------------------------------------------------------------
# REQUEST
# -----------------------------------------------------------------------------
response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        PROMPT,
        starting_image,  # You can also send multiple images if needed
    ],
)

# -----------------------------------------------------------------------------
# PROCESS RESPONSE (safe version)
# -----------------------------------------------------------------------------
image_count = 0

for candidate in response.candidates:
    for part in candidate.content.parts:
        # --- DEBUG print to understand what we got back ---
        # (you can comment this out later)
        print(f"[DEBUG] Part type: {type(part)}")
        if hasattr(part, "text") and part.text:
            print(f"[TEXT] {part.text[:200]}")
        elif hasattr(part, "inline_data") and part.inline_data:
            if part.inline_data.mime_type and part.inline_data.mime_type.startswith("image/"):
                image_count += 1
                image_bytes = part.inline_data.data
                img = Image.open(BytesIO(image_bytes))
                output_path = os.path.join(OUTPUT_DIR, f"output_{image_count}.png")
                img.save(output_path)
                print(f"[+] Saved: {output_path}")
            else:
                print(f"[WARN] inline_data exists but not image, mime: {part.inline_data.mime_type}")
        else:
            print("[WARN] Part had neither text nor inline_data.")

if image_count == 0:
    print("[!] No image data found in response. "
          "Your prompt might have produced text only or model access may be limited.")

print(f"[*] Done. Generated {image_count} image(s).")
