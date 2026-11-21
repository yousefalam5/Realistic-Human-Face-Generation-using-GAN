import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torchvision.utils as vutils
from PIL import Image

from dataset import denormalize, get_celeba_dataloader
from models import Generator
from train import train_dcgan

PROJECT_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
SAMPLES_DIR = PROJECT_DIR / "samples"
DATA_DIR = PROJECT_DIR / "data" / "celeba" / "img_align_celeba"

st.set_page_config(
    page_title="CelebA Face DCGAN Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.05rem; color: #4B5563; margin-bottom: 1.5rem; }
    .metric-badge { background-color: #EEF2FF; color: #3730A3; padding: 4px 10px; border-radius: 6px; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading DCGAN Generator Model...")
def load_generator(device: str = "cpu") -> Generator:
    netG = Generator(nz=100, ngf=64, nc=3).to(device)
    weights_path = CHECKPOINT_DIR / "generator.pth"
    if weights_path.exists():
        netG.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    netG.eval()
    return netG


device = "cuda" if torch.cuda.is_available() else "cpu"
generator = load_generator(device)

# --- SIDEBAR: CONTROLS & TRAINING ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
    st.title("DCGAN Control Center")
    st.caption("Deep Convolutional GAN on CelebA")

    st.markdown("---")
    st.subheader("1. Hardware & Model Status")
    st.markdown(f"**Compute Device:** `{device.upper()}`")
    weights_ready = (CHECKPOINT_DIR / "generator.pth").exists()
    if weights_ready:
        st.success("Trained Generator Loaded ✅")
    else:
        st.warning("Model weights not found. Click Train below.")

    st.markdown("---")
    st.subheader("2. Adversarial Training")
    num_epochs = st.slider("Training Epochs", min_value=1, max_value=10, value=3)
    batch_size = st.select_slider("Batch Size", options=[16, 32, 64], value=32)

    if st.button("Train / Fine-Tune DCGAN 🚀", type="primary", use_container_width=True):
        with st.spinner("Training DCGAN on CelebA dataset..."):
            train_report = train_dcgan(
                data_dir=DATA_DIR,
                output_dir=CHECKPOINT_DIR,
                samples_dir=SAMPLES_DIR,
                num_epochs=num_epochs,
                batch_size=batch_size,
                device=device,
            )
        st.success(f"Training completed in {train_report['total_time']}s!")
        st.cache_resource.clear()
        st.rerun()

    st.markdown("---")
    st.subheader("3. Architecture Specs")
    st.markdown(
        """
        - **Latent Vector ($z$):** 100-dim $\\mathcal{N}(0, 1)$
        - **Output Size:** $3 \\times 64 \\times 64$ RGB
        - **Activation:** Tanh (Generator) / LeakyReLU (Discriminator)
        - **Optimizer:** Adam (lr=0.0002, $\\beta_1=0.5$)
        """
    )


# --- MAIN INTERFACE ---
st.markdown('<div class="main-header">🎨 CelebA Face DCGAN Generator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Deep Convolutional Generative Adversarial Network that synthesizes '
    'realistic synthetic human face images from random noise vectors.</div>',
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4 = st.tabs([
    "🎲 Real-Time Face Generator",
    "🎛️ Latent Space Interpolation",
    "📈 Training Progression & Loss Curves",
    "👥 CelebA Real Dataset Explorer",
])

# ==========================================
# TAB 1: Face Generator
# ==========================================
with tab1:
    st.subheader("Generate Synthetic Faces from Latent Noise Vectors")
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        seed = st.number_input("Random Seed", value=42, min_value=0, max_value=999999)
    with c2:
        num_faces = st.slider("Number of Faces", min_value=1, max_value=16, value=8)
    with c3:
        randomize = st.button("🎲 Randomize Seed", use_container_width=True)
        if randomize:
            seed = int(np.random.randint(0, 999999))

    torch.manual_seed(seed)
    noise = torch.randn(num_faces, 100, 1, 1, device=device)

    with torch.no_grad():
        fakes = generator(noise).detach().cpu()
        denorm_fakes = denormalize(fakes)

    # Display grid
    cols = st.columns(min(4, num_faces))
    for i in range(num_faces):
        col = cols[i % len(cols)]
        img_np = denorm_fakes[i].permute(1, 2, 0).numpy()
        pil_img = Image.fromarray((img_np * 255).astype(np.uint8))
        with col:
            st.image(pil_img, caption=f"Synthetic Face #{i+1}", use_container_width=True)

    # Download All Grid
    grid_img = vutils.make_grid(denorm_fakes, nrow=4, padding=4)
    grid_pil = Image.fromarray((grid_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    buf = io.BytesIO()
    grid_pil.save(buf, format="PNG")
    st.download_button(
        "📥 Download Generated Face Grid (PNG)",
        data=buf.getvalue(),
        file_name="dcgan_synthetic_faces_grid.png",
        mime="image/png",
    )

# ==========================================
# TAB 2: Latent Space Interpolation
# ==========================================
with tab2:
    st.subheader("Latent Space Manifold Walk (Face Morphing)")
    st.caption("Demonstrates that the generator learned a continuous semantic manifold of facial features.")

    i_col1, i_col2, i_col3 = st.columns(3)
    with i_col1:
        seed_a = st.number_input("Face A Seed", value=100, min_value=0, max_value=99999)
    with i_col2:
        seed_b = st.number_input("Face B Seed", value=500, min_value=0, max_value=99999)
    with i_col3:
        steps = st.slider("Interpolation Steps", min_value=5, max_value=12, value=8)

    torch.manual_seed(seed_a)
    z_a = torch.randn(1, 100, 1, 1, device=device)
    torch.manual_seed(seed_b)
    z_b = torch.randn(1, 100, 1, 1, device=device)

    # Spherical / Linear interpolation
    alphas = np.linspace(0.0, 1.0, steps)
    interpolated_noise = torch.cat([((1 - a) * z_a + a * z_b) for a in alphas], dim=0)

    with torch.no_grad():
        morph_fakes = generator(interpolated_noise).detach().cpu()
        denorm_morph = denormalize(morph_fakes)

    morph_cols = st.columns(steps)
    for idx, col in enumerate(morph_cols):
        img_np = denorm_morph[idx].permute(1, 2, 0).numpy()
        pil_img = Image.fromarray((img_np * 255).astype(np.uint8))
        label = "Face A" if idx == 0 else ("Face B" if idx == steps - 1 else f"α = {alphas[idx]:.2f}")
        with col:
            st.image(pil_img, caption=label, use_container_width=True)

# ==========================================
# TAB 3: Training Progression & Loss Curves
# ==========================================
with tab3:
    st.subheader("Adversarial Loss Curves & Evolution Over Epochs")
    history_path = CHECKPOINT_DIR / "history.json"
    if history_path.exists():
        with open(history_path, "r") as f:
            hist = json.load(f)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(hist["G_losses"], label="Generator Loss (G)", color="#2563EB", lw=2)
        ax.plot(hist["D_losses"], label="Discriminator Loss (D)", color="#DC2626", lw=2)
        ax.set_title("DCGAN Minimax Training Loss Dynamics")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Binary Cross-Entropy Loss")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        st.pyplot(fig)

    # Display epoch sample images if available
    epoch_samples = sorted(list(SAMPLES_DIR.glob("epoch_*.png")))
    if epoch_samples:
        st.subheader("Visual Evolution from Noise to Realistic Faces")
        s_cols = st.columns(min(len(epoch_samples), 4))
        for i, s_file in enumerate(epoch_samples[:4]):
            with s_cols[i]:
                st.image(str(s_file), caption=f"Progression: {s_file.stem.replace('_', ' ').title()}", use_container_width=True)

# ==========================================
# TAB 4: Real CelebA Dataset Explorer
# ==========================================
with tab4:
    st.subheader("Real CelebA Human Face Dataset (Ground Truth)")
    dataloader = get_celeba_dataloader(DATA_DIR, batch_size=8, image_size=64)
    real_batch, _ = next(iter(dataloader))
    real_denorm = denormalize(real_batch)

    r_cols = st.columns(4)
    for i in range(min(8, real_denorm.size(0))):
        col = r_cols[i % 4]
        img_np = real_denorm[i].permute(1, 2, 0).numpy()
        pil_img = Image.fromarray((img_np * 255).astype(np.uint8))
        with col:
            st.image(pil_img, caption=f"CelebA Real #{i+1}", use_container_width=True)
