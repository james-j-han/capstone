from flask import Flask, render_template, redirect, request
import torch
import clip
import plotly.express as px
from torchvision import datasets, transforms
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import plotly.io as pio
import numpy as np
from base64 import b64encode
from io import BytesIO
from PIL import Image
import plotly.graph_objects as go
import openai
import requests
from dotenv import load_dotenv
import os

# load_dotenv()

app = Flask(__name__)

# openai_api_key = os.getenv("OPENAI_API_KEY")

# client = openai.Client(openai_api_key)

# def generate_image(prompt):
#     try:
#         response = client.images.generate(
#             model='dall-e-3',
#             prompt=prompt,
#             size='256x256',
#             n=1,
#             quality='standard'
#         )
#         image_url = response.data[0].url
#         return image_url
#     except Exception as e:
#         print(f"Error: {e}")
#         return None

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Precompute CIFAR-10 features
cifar_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transforms.Resize((224, 224)))

# global_dataset = None
# global_features = None

# def load_and_preprocess_dataset():
#     global global_dataset, global_features
#     # cifar_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transforms.Resize((224, 224)))
#     global_dataset = load_cifar10_images(num_images=200)
#     global_features = extract_clip_features(global_dataset)

# Function to load limited CIFAR-10 images for visualization (without ToTensor)
def load_cifar10_images(num_images=100):
    # transform = transforms.Compose([
    #     transforms.Resize((224, 224))  # Only resize for visualization, no tensor conversion
    # ])
    # full_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    return torch.utils.data.Subset(cifar_dataset, range(num_images))

# Extract features from dataset using CLIP
def extract_clip_features(dataset):
    image_features = []
    for img, _ in dataset:
        img = preprocess(img).unsqueeze(0).to(device)  # Apply preprocess during feature extraction
        with torch.no_grad():
            feature = model.encode_image(img)
        feature /= feature.norm(dim=-1, keepdim=True)
        image_features.append(feature.cpu().numpy())
    return np.concatenate(image_features, axis=0)

# Apply t-SNE to reduce the dimensions of CLIP features
def apply_tsne(features, n_components=2):
    tsne = TSNE(n_components=n_components, random_state=42)
    return tsne.fit_transform(features)

# load_and_preprocess_dataset()

@app.route('/', methods=['GET', 'POST'])
def index():
    height = 600
    width = 600

    # Load CIFAR-10 images and extract features
    cifar_dataset = load_cifar10_images(num_images=1000)
    image_features = extract_clip_features(cifar_dataset)

    # Apply t-SNE for 2D and 3D plots
    tsne_2d = apply_tsne(image_features, n_components=2)
    tsne_3d = apply_tsne(image_features, n_components=3)

    # Prepare the image data for visualization
    image_list = []
    for img, _ in cifar_dataset:
        image_list.append(img)

    # Create a 2D scatter plot with images
    fig_2d = go.Figure()
    for i, img in enumerate(image_list):
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        img_b64 = f"data:image/png;base64,{b64encode(buffer.getvalue()).decode('utf-8')}"
        fig_2d.add_layout_image(
            dict(
                source=img_b64,
                xref="x", yref="y",
                x=tsne_2d[i, 0], y=tsne_2d[i, 1],
                sizex=0.1, sizey=0.1,
                xanchor="center", yanchor="middle"
            )
        )

    # Update layout for 2D plot
    fig_2d.update_layout(
        # title="2D t-SNE Visualization of CLIP Features with CIFAR-10 Images",
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        height=height,
        width=width
    )

    # 3D Scatter Plot with markers (images not supported in 3D)
    fig_3d = go.Figure(data=[go.Scatter3d(
        x=tsne_3d[:, 0],
        y=tsne_3d[:, 1],
        z=tsne_3d[:, 2],
        mode='markers',
        marker=dict(size=5, color=tsne_3d[:, 0], colorscale='Viridis', opacity=0.8)
    )])

    fig_3d.update_layout(
        # title="3D t-SNE Visualization",
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z')
        ),
        height=height,
        width=width
    )

    # Convert the plot to JSON to pass to the template
    graph_json_2d = fig_2d.to_json()
    graph_json_3d = fig_3d.to_json()

    if request.method == 'POST' and 'image' in request.files:
        file = request.files['image']
    
        if file:
            # Load the uploaded image and convert to RGB
            image = Image.open(file.stream).convert("RGB")

            # Preprocess the uploaded image for CLIP
            uploaded_img = preprocess(image).unsqueeze(0).to(device)

            # Extract the CLIP features of the uploaded image
            with torch.no_grad():
                uploaded_img_features = model.encode_image(uploaded_img)
            uploaded_img_features /= uploaded_img_features.norm(dim=-1, keepdim=True)

            # Convert features to numpy and ensure they are 2D
            uploaded_img_features = uploaded_img_features.cpu().numpy().squeeze()  # Shape: [n_features]

            # Calculate cosine similarity between the uploaded image and CIFAR-10 dataset
            # cifar_dataset = load_cifar10_images(num_images=100)
            # cifar_features = extract_clip_features(cifar_dataset)
            similarities = cosine_similarity(uploaded_img_features.reshape(1, -1), image_features)

            # Get top-K most similar images
            K = int(request.form.get('k', 5))  # Get 'k' from the form, default to 5
            top_k_indices = similarities.argsort()[0][-K:][::-1]

            # Get the top-K images from CIFAR-10
            top_k_images = [cifar_dataset[i][0] for i in top_k_indices]

            # Convert top-K images to base64 to display on the frontend
            top_k_b64_images = []
            for img in top_k_images:
                buffer = BytesIO()
                img.save(buffer, format="PNG")
                img_b64 = b64encode(buffer.getvalue()).decode('utf-8')
                top_k_b64_images.append(f"data:image/png;base64,{img_b64}")
            
            # Render the results page with top-K similar images
            return render_template('index.html', top_k=top_k_b64_images, graph_json_2d=graph_json_2d, graph_json_3d=graph_json_3d)

    return render_template('index.html', graph_json_2d=graph_json_2d, graph_json_3d=graph_json_3d)