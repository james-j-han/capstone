from flask import Flask, jsonify, render_template, request
from sklearn.utils import shuffle
import torch
import clip
from torchvision import datasets, transforms
from PIL import Image
from io import BytesIO
from base64 import b64encode
import numpy as np
import joblib
import umap
import time
from tqdm import tqdm

app = Flask(__name__)

dataset_path = './images'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

image_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

# Use a subset if needed
num_images = 28000
num_images = min(num_images, len(image_dataset))  # Limit to the first 100 images
subset = torch.utils.data.Subset(image_dataset, range(num_images))

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Use DataLoader for batching
batch_size = 32
data_loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False)

# Extract features
# def extract_clip_features(data_loader):
#     all_features = []
#     with torch.no_grad():
#         for images, _ in data_loader:
#             images = images.to(device)
#             features = model.encode_image(images)  # Encode images
#             features /= features.norm(dim=-1, keepdim=True)  # Normalize
#             all_features.append(features.cpu().numpy())  # Move to CPU
#     return np.vstack(all_features)  # Stack all batches into one array

def extract_clip_features_with_tqdm(data_loader, model, device):
    all_features = []
    total_batches = len(data_loader)

    print("Starting feature extraction...")
    start_time = time.time()

    with torch.no_grad():
        for images, _ in tqdm(data_loader, total=total_batches, desc="Extracting Features"):
            images = images.to(device)
            features = model.encode_image(images)
            features /= features.norm(dim=-1, keepdim=True)  # Normalize
            all_features.append(features.cpu().numpy())  # Move to CPU

    total_time = time.time() - start_time  # Total time
    print(f"Feature extraction completed in {total_time:.2f} seconds.")

    return np.vstack(all_features)  # Combine all features into one array

def train_umap_with_progress(data, n_components=2, n_epochs=200, random_state=42):
    """
    Train a UMAP model with a manually controlled progress bar for epochs.

    Args:
        data (np.ndarray): Input data for UMAP training.
        n_components (int): Number of dimensions for UMAP embeddings.
        n_epochs (int): Total number of epochs for UMAP training.
        random_state (int): Random seed for reproducibility.

    Returns:
        model (umap.UMAP): Trained UMAP model.
        embeddings (np.ndarray): Low-dimensional embeddings of the input data.
    """
    print(f"Training UMAP ({n_components}D) with {n_epochs} epochs...")

    # Initialize UMAP model with a subset of epochs
    model = umap.UMAP(n_components=n_components, n_epochs=n_epochs, random_state=random_state)

    # Manually track progress
    with tqdm(total=n_epochs, desc=f"Training UMAP {n_components}D") as pbar:
        for epoch in range(1, n_epochs + 1):
            model.n_epochs = epoch
            model.fit(data)  # Fit the model incrementally
            pbar.update(1)  # Update the progress bar by 1 epoch

    # Transform the entire dataset
    embeddings = model.transform(data)
    return model, embeddings

##############################################################################
# PREPROCESSING
##############################################################################

# Get image features
# print('Extracting features...')
# # image_features = extract_clip_features(data_loader)
# image_features = extract_clip_features_with_tqdm(data_loader, model, device)
# print(f'Extracted features: {image_features.shape}')

# # Save image features for reuse
# np.save("image_features.npy", image_features)
# print("Saved image features as 'image_features.npy'")

# # Train 2D UMAP with progress
# umap_model_2d, umap_2d = train_umap_with_progress(image_features, n_components=2, n_epochs=10)
# joblib.dump(umap_model_2d, "umap_model_2d.pkl")
# np.save("umap_2d.npy", umap_2d)

# # Train 3D UMAP with progress
# umap_model_3d, umap_3d = train_umap_with_progress(image_features, n_components=3, n_epochs=10)
# joblib.dump(umap_model_3d, "umap_model_3d.pkl")
# np.save("umap_3d.npy", umap_3d)
# print("UMAP models and embeddings saved.")

# def image_to_base64(img_tensor, size=(64, 64), quality=70):  # Adjust quality
#     img = transforms.ToPILImage()(img_tensor).resize(size)
#     buffer = BytesIO()
#     img.save(buffer, format="JPEG", quality=quality)  # Save as JPEG
#     return f"data:image/jpeg;base64,{b64encode(buffer.getvalue()).decode()}"

# print("Converting images to Base64...")
# image_b64_list = []
# for img, _ in tqdm(subset, desc="Encoding Images", total=len(subset)):
#     image_b64_list.append(image_to_base64(img))
# np.save("image_b64_list.npy", image_b64_list)  # Save the Base64 strings
# print("Base64 encoding complete.")

##############################################################################
# LOADING
##############################################################################

# Load precomputed data
image_features = np.load("image_features.npy")
image_b64_list = np.load("image_b64_list.npy", allow_pickle=True).tolist()
umap_2d = np.load("umap_2d.npy")
umap_3d = np.load("umap_3d.npy")

# Load UMAP models
umap_model_2d = joblib.load("umap_model_2d.pkl")
umap_model_3d = joblib.load("umap_model_3d.pkl")

@app.route('/query', methods=['POST'])
def query():
    try:
        query_type = request.form.get('type') or (request.json or {}).get('type')
        print(f"Received query type: {query_type}")

        if query_type not in ['image', 'text']:
            return jsonify({"error": "Invalid query type. Must be 'image' or 'text'"}), 400

        if query_type == 'text':
            # Handle text query
            data = request.get_json() or {}
            query_text = data.get('text', '')
            k = int(data.get('k', 5))

            if not query_text:
                return jsonify({"error": "Text query is required"}), 400

            print(f"Processing text query: {query_text}, Top-K: {k}")

            # Extract text features
            with torch.no_grad():
                text_tokens = clip.tokenize([query_text]).to(device)
                text_features = model.encode_text(text_tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)

            # Compute similarity scores
            similarity = torch.nn.functional.cosine_similarity(
                torch.tensor(image_features).to(device), text_features, dim=1
            )

            # Find the top-K most similar images
            top_k_indices = torch.topk(similarity, k, largest=True).indices.tolist()

            # Transform text features to t-SNE coordinates
            umap_2d_query = umap_model_2d.transform(text_features.cpu().numpy())[0]
            umap_3d_query = umap_model_3d.transform(text_features.cpu().numpy())[0]

            # Prepare the top-K results
            top_k_results = [
                {
                    "x": float(umap_2d[i, 0]),
                    "y": float(umap_2d[i, 1]),
                    "z": float(umap_3d[i, 2]),
                    "image": image_b64_list[i]
                }
                for i in top_k_indices
            ]

            response = {
                "query_point": {
                    "x_2d": float(umap_2d_query[0]),
                    "y_2d": float(umap_2d_query[1]),
                    "x_3d": float(umap_3d_query[0]),
                    "y_3d": float(umap_3d_query[1]),
                    "z_3d": float(umap_3d_query[2]),
                },
                "top_k_results": top_k_results,
                "query_type": query_type,
                "query_text": query_text,
            }
            return jsonify(response)

        elif query_type == 'image':
            # Handle image query
            if 'image' not in request.files:
                return jsonify({"error": "Image is required for image query"}), 400

            file = request.files['image']
            k = int(request.form.get('k', 5))

            print(f"Processing image query: File received: {file.filename}, Top-K: {k}")

            # Preprocess the uploaded image
            image = Image.open(file.stream).convert("RGB")
            img_tensor = preprocess(image).unsqueeze(0).to(device)

            # Extract features for the query image
            with torch.no_grad():
                query_features = model.encode_image(img_tensor)
                query_features /= query_features.norm(dim=-1, keepdim=True)

            # Compute similarity scores
            similarity = torch.nn.functional.cosine_similarity(
                torch.tensor(image_features).to(device), query_features, dim=1
            )

            # Find the top-K most similar images
            top_k_indices = torch.topk(similarity, k, largest=True).indices.tolist()

            # Transform image features to t-SNE coordinates
            umap_2d_query = umap_model_2d.transform(query_features.cpu().numpy())[0]
            umap_3d_query = umap_model_3d.transform(query_features.cpu().numpy())[0]

            # Prepare the top-K results
            top_k_results = [
                {
                    "x": float(umap_2d[i, 0]),
                    "y": float(umap_2d[i, 1]),
                    "z": float(umap_3d[i, 2]),
                    "image": image_b64_list[i]
                }
                for i in top_k_indices
            ]

            response = {
                "query_point": {
                    "x_2d": float(umap_2d_query[0]),
                    "y_2d": float(umap_2d_query[1]),
                    "x_3d": float(umap_3d_query[0]),
                    "y_3d": float(umap_3d_query[1]),
                    "z_3d": float(umap_3d_query[2]),
                },
                "top_k_results": top_k_results,
                "query_type": query_type,
                "query_text": query_text,
            }
            return jsonify(response)

    except Exception as e:
        print(f"Error processing query: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/plot-data', methods=['GET'])
def get_plot_data():
    # Send dot positions and image data
    dots_data = [
        {
            "x": float(umap_2d[i, 0]),
            "y": float(umap_2d[i, 1]),
            "z": float(umap_3d[i, 2]),
            "image": image_b64_list[i]
        } for i in range(len(subset))
    ]
    return jsonify(dots_data)

@app.route('/')
def index():
    return render_template('index.html')  # Serve the HTML template

if __name__ == '__main__':
    app.run(debug=True)