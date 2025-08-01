from vertexai.preview.vision_models import ImageGenerationModel
import vertexai

vertexai.init(project="creators-multi-verse", location="us-central1")

generation_model = ImageGenerationModel.from_pretrained("imagen-4.0-fast-generate-preview-06-06")

images = generation_model.generate_images(
    prompt="A beautiful scenery of people finding their Mental Support in Terapie Acasa Ecosystem",
    number_of_images=1,
    aspect_ratio="1:1",
    negative_prompt="",
    person_generation="allow_all",
    safety_filter_level="block_few",
    add_watermark=True,
)

# Access and save image (image is a PIL.Image.Image object)
image = images[0].image  # This is a PIL image
image.save("terapie_acasa_image.jpg")