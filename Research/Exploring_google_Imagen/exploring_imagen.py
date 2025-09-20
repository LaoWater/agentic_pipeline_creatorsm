from google import genai
from google.genai import types
from PIL import Image
# PIL is implicitly used by the library, but good to know
import os

# --- Configuration ---
# Ensure your GOOGLE_API_KEY is set as an environment variable.
# The client will find it automatically.
# e.g., os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY"
# genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


# Create client
print("[*] Initializing Google Generative AI client...")
client = genai.Client()

# Generate image
print("[*] Sending prompt to Imagen model...")
try:
    # Using the preview model from your example.
    # If this model is unavailable to you, switch to 'imagen-3.0-generate-06-06'
    response = client.models.generate_images(
        model='imagen-4.0-generate-preview-06-06',
        prompt='A high-resolution, photorealistic image of a friendly robot holding a red skateboard in a sunny skatepark.',
        config=types.GenerateImagesConfig(
            number_of_images=1,
            aspect_ratio="1:1" # Explicitly setting aspect ratio
        )
    )

    # Handle response
    if not response.generated_images:
        print("[!] No images were returned by the API!")
    else:
        for idx, generated_image in enumerate(response.generated_images):
            print(f"[+] Image {idx + 1} received. Processing...")

            # --- KEY CORRECTION ---
            # The `generated_image.image` attribute is ALREADY a PIL-compatible image object.
            # We do not need to decode bytes manually. We can use it directly.
            img = generated_image.image

            # You can show the image directly (optional)
            # This will open in your default image viewer.
            # print("[*] Displaying image...")
            # img.show()

            # Save the image directly using its .save() method
            filename = f"output_robot_{idx + 1}.png"
            img.save(filename)
            print(f"[+] Image saved successfully as {filename}")

except Exception as e:
    # This will catch API errors (e.g., model not available, invalid key)
    # and any other runtime errors.
    print(f"[!] An error occurred: {e}")