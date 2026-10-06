import numpy as np
import cv2
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import math

# --- PAGE CONFIG ---
st.set_page_config(page_title="SVD Image Compression & Denoising", layout="wide")
st.title("SVD-Based Image Compression & Noise Reduction")

# --- CORE SVD FUNCTIONS ---
def compress_channel_svd(channel, energy_threshold=0.95):
    U, S, Vt = np.linalg.svd(channel, full_matrices=False)
    total_energy = np.sum(S ** 2)
    cumulative_energy = np.cumsum(S ** 2) / total_energy
    k = np.searchsorted(cumulative_energy, energy_threshold) + 1
    
    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]
    
    reconstructed = np.dot(U_k, np.dot(np.diag(S_k), Vt_k))
    reconstructed = np.clip(reconstructed, 0, 255)
    return reconstructed.astype(np.uint8), k, len(S), S, cumulative_energy

def process_image(image_np, energy_threshold=0.95, add_noise=False, noise_factor=20):
    if add_noise:
        noise = np.random.normal(0, noise_factor, image_np.shape)
        image_np = np.clip(image_np.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if len(image_np.shape) == 2:
        compressed, k, max_k, S, cum_energy = compress_channel_svd(image_np.astype(float), energy_threshold)
    else:
        channels = cv2.split(image_np)
        compressed_channels = []
        ranks = []
        for ch in channels:
            comp_ch, k, max_k, S, cum_energy = compress_channel_svd(ch.astype(float), energy_threshold)
            compressed_channels.append(comp_ch)
            ranks.append(k)
        compressed = cv2.merge(compressed_channels)
        k = int(np.mean(ranks))

    return image_np, compressed, k, max_k, S, cum_energy

def calculate_metrics(original, compressed, k):
    m, n = original.shape[:2]
    num_channels = 1 if len(original.shape) == 2 else original.shape[2]
    mse = np.mean((original.astype(float) - compressed.astype(float)) ** 2)
    rmse = math.sqrt(mse)
    psnr = 20 * math.log10(255.0 / rmse) if rmse != 0 else 100
    orig_elements = m * n * num_channels
    comp_elements = k * (m + n + 1) * num_channels
    compression_ratio = orig_elements / comp_elements
    space_saving = (1 - (comp_elements / orig_elements)) * 100
    return mse, rmse, psnr, compression_ratio, space_saving

# --- SIDEBAR & INTERFACE ---
st.sidebar.header("Control Panel")
uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
energy_target = st.sidebar.slider("Target Energy Retention (%)", 50, 100, 95) / 100.0
apply_noise = st.sidebar.checkbox("Simulate Noise Reduction Demo")
noise_level = st.sidebar.slider("Noise Level", 5, 50, 25) if apply_noise else 0

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img_np = np.array(img)
    orig_img, comp_img, selected_k, max_k, S_vals, cum_energy = process_image(img_np, energy_target, apply_noise, noise_level)
    mse, rmse, psnr, ratio, space_saving = calculate_metrics(orig_img, comp_img, selected_k)

    st.subheader("Performance & Quality Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Selected Rank (k)", f"{selected_k} / {max_k}")
    col2.metric("Compression Ratio", f"{ratio:.2f}:1")
    col3.metric("Space Saved", f"{space_saving:.1f}%")
    col4.metric("RMSE", f"{rmse:.2f}")
    col5.metric("PSNR", f"{psnr:.2f} dB")

    st.subheader("Visual Comparison")
    col_a, col_b = st.columns(2)
    with col_a:
        st.image(orig_img, caption="Original / Noisy Image", use_column_width=True)
    with col_b:
        st.image(comp_img, caption=f"Compressed / Denoised (Rank k={selected_k})", use_column_width=True)

    st.subheader("SVD Energy & Quality Trade-Off Analysis")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(S_vals[:100], color='blue', linewidth=2)
    ax1.set_title("Top 100 Singular Values Decay (Log-Scale)")
    ax1.set_yscale('log')
    ax1.set_xlabel("Component Rank (i)")
    ax1.set_ylabel("Singular Value Energy")
    ax1.grid(True)

    ax2.plot(cum_energy * 100, color='green', linewidth=2)
    ax2.axhline(y=energy_target*100, color='r', linestyle='--', label=f'{energy_target*100:.0f}% Threshold')
    ax2.axvline(x=selected_k, color='black', linestyle=':', label=f'Auto k={selected_k}')
    ax2.set_title("Cumulative Energy vs Rank k")
    ax2.set_xlabel("Rank (k)")
    ax2.set_ylabel("Retained Energy (%)")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig)
else:
    st.info("Please upload an image from the sidebar to test the system.")