from google import genai
from google.genai import types
from PIL import Image # PIL is implicitly used by the library, but good to know
import os


# Create client
print("[*] Initializing Google Generative AI client...")
client = genai.Client()

# Generate image
print("[*] Sending prompt to Imagen model...")
try:
    # Using the preview model from your example.
    # If this model is unavailable to you, switch to 'imagen-3.0-generate-06-06'
    response = client.models.generate_images(
        model='imagen-4.0-ultra-generate-001',
        prompt='A high-resolution, photorealistic image of the DoDo bird flying over a calm jungle',
        config=types.GenerateImagesConfig(
            number_of_images=1,
            aspect_ratio="9:16"
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
            filename = f"imagen4ultra_dodo_{idx + 1}.png"
            img.save(filename)
            print(f"[+] Image saved successfully as {filename}")

except Exception as e:
    # This will catch API errors (e.g., model not available, invalid key)
    # and any other runtime errors.
    print(f"[!] An error occurred: {e}")