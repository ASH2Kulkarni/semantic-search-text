from sentence_transformers import SentenceTransformer, util
from PIL import Image

# Load CLIP model
model = SentenceTransformer("clip-ViT-B-32")

# Encode an image:
img_emb = model.encode(Image.open("C:/Users/ASHWINI/Desktop/jijuML/two_dogs_in_snow.jpg"))

# Encode text descriptions
text_embs = model.encode(
    ["Two dogs in the snow", "A cat on a table", "A picture of London at night"]
)

# Compute similarities
similarity_scores = util.pytorch_cos_sim(img_emb, text_embs)
print(similarity_scores)
