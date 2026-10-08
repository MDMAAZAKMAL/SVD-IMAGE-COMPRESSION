import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import math

# Page Title
st.title("SVD Image Compression & Noise Reduction")

# Sidebar Controls
uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
energy_target = st.sidebar.slider("Target Energy Retention (%)", 50, 100, 95) / 100.0

if uploaded_file is not None:
    # 1. Load Image and convert to RGB
    image = Image.open(uploaded_file).convert("RGB")
    
    # Resize large images for super fast processing (Max width/height: 800px)
    image.thumbnail((800, 800))
    img_np = np.array(image)
    
    # Get image dimensions
    m, n, channels = img_np.shape
    
    compressed_channels = []
    ranks = []
    
    # 2. Fast SVD processing across color channels
    for i in range(3):
        channel = img_np[:, :, i].astype(np.float64)
        
        # Compute SVD: A = U * S * Vt
        U, S, Vt = np.linalg.svd(channel, full_matrices=False)
        
        # Calculate rank k based on energy retention
        total_energy = np.sum(S ** 2)
        cum_energy = np.cumsum(S ** 2) / total_energy
        
        # Determine k
        k = np.searchsorted(cum_energy, energy_target) + 1
        
        # Enforce a minimum rank (at least 10% of total rank) so fine details stay crisp
        min_k = max(10, int(len(S) * 0.10))
        k = max(k, min_k)
        k = min(k, len(S))
        ranks.append(k)
        
        # Reconstruct channel using top k components: A_k = U_k * S_k * Vt_k
        U_k = U[:, :k]
        S_k = S[:k]
        Vt_k = Vt[:k, :]
        
        reconstructed = np.dot(U_k, np.dot(np.diag(S_k), Vt_k))
        
        # Clip values to valid pixel range [0, 255]
        reconstructed = np.clip(reconstructed, 0, 255)
        compressed_channels.append(reconstructed.astype(np.uint8))
    
    # Stack channels back together
    compressed_img = np.stack(compressed_channels, axis=2)
    selected_k = int(np.mean(ranks))
    max_k = min(m, n)
    
    # 3. Calculate Performance Metrics
    mse = np.mean((img_np.astype(float) - compressed_img.astype(float)) ** 2)
    rmse = math.sqrt(mse)
    psnr = 20 * math.log10(255.0 / rmse) if rmse != 0 else 100
    
    orig_size = m * n * 3
    comp_size = selected_k * (m + n + 1) * 3
    ratio = orig_size / comp_size
    space_saved = (1 - (comp_size / orig_size)) * 100
    
    # 4. Display Metrics
    st.subheader("Performance Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Rank (k)", f"{selected_k} / {max_k}")
    col2.metric("Compression Ratio", f"{ratio:.2f}:1")
    col3.metric("Space Saved", f"{space_saved:.1f}%")
    col4.metric("RMSE", f"{rmse:.2f}")
    col5.metric("PSNR", f"{psnr:.2f} dB")
    
    # 5. Display Images Side by Side
    st.subheader("Visual Comparison")
    col_a, col_b = st.columns(2)
    with col_a:
        st.image(img_np, caption="Original Image", use_container_width=True)
    with col_b:
        st.image(compressed_img, caption=f"Compressed Image (Rank k={selected_k})", use_container_width=True)
        
    # 6. Display Top 100 Singular Values Plot
    st.subheader("Singular Value Decay Plot")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(S[:100], color='blue', linewidth=2)
    ax.set_title("Top 100 Singular Values Decay (Log Scale)")
    ax.set_yscale('log')
    ax.set_xlabel("Component Rank (i)")
    ax.set_ylabel("Singular Value Energy")
    ax.grid(True)
    
    st.pyplot(fig)

else:
    st.info("Please upload an image from the sidebar.")