"""
RoadScan-MA — app.py
Dashboard Streamlit principal pour la détection de dégradations routières.
Modes : IMAGE | VIDÉO | DÉMO
"""

import os
import io
import cv2
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from PIL import Image
from streamlit_folium import st_folium

from inference import (load_model, predict_image, predict_video,
                       bgr_to_rgb, CLASS_LABELS_FR, CLASS_COLORS_HEX)
from utils import (format_summary, generate_image_coords,
                   generate_video_coords, export_csv)
from map_generator import generate_image_map, generate_video_map
from pdf_report import generate_report


# ── Config page ────────────────────────────────────────────
st.set_page_config(
    page_title="RoadScan-MA",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS custom ─────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stMetricValue"] { font-size: 1.8rem !important; }
  [data-testid="stMetricLabel"] { font-size: 0.85rem !important; }
  .block-container { padding-top: 1.2rem; }
  .stProgress > div > div { background-color: #E74C3C; }
  div[data-testid="stSidebarContent"] { background: #1A1D23; }
  .severity-badge {
    display: inline-block; padding: 4px 12px;
    border-radius: 20px; font-weight: bold; font-size: 0.9rem;
  }
</style>
""", unsafe_allow_html=True)


# ── Chargement modèle (1 seule fois) ──────────────────────
@st.cache_resource
def load_model_cached():
    try:
        return load_model("models/yolo_final_best.pt") # CHANGÉ ICI
    except FileNotFoundError:
        return None
# ══════════════════════════════════════════════════════════
#  MODE TEST MEKNES (Défaut)
# ══════════════════════════════════════════════════════════
def test_meknes_mode():
    st.markdown("## 🌍 Test Global : Terrain Réel Meknès")
    st.markdown("Analyse d'un parcours réel avec le modèle `yolo_final_best.pt`.")
    st.divider()

    t1, t2, t3 = st.tabs(["🎬 Vidéo Originale", "📸 10 Échantillons Annotés", "🗺️ Carte GPS Interactive"])

    # ── Tab 1 : Video ──────────────────────────────────
    # ── Tab 1 : Video ──────────────────────────────────
    with t1:
        video_path = os.path.join(os.path.dirname(__file__), 'Video2.MOV')
        if os.path.exists(video_path):
            st.video(video_path, muted=True) # <-- AJOUTE muted=True ICI
        else:
            st.warning("⚠️ Place le fichier Video2.MOV dans le même dossier que app.py")

    # ── Tab 2 : Frames annotees ────────────────────────
    with t2:
        ann_dir = os.path.join(os.path.dirname(__file__), 'annotated_frames')
        if os.path.exists(ann_dir):
            frames = sorted([f for f in os.listdir(ann_dir) if f.endswith(('.jpg', '.png'))])
            if frames:
                cols = st.columns(2)
                for i, fname in enumerate(frames[:10]): # Affiche les 10 premières
                    with cols[i % 2]:
                        st.image(
                            os.path.join(ann_dir, fname),
                            caption=f"Détection : {fname}",
                            use_container_width=True
                        )
            else:
                st.info("Aucune image trouvée dans le dossier annotated_frames.")
        else:
            st.warning("⚠️ Place le dossier annotated_frames/ dans le même dossier que app.py")

    # ── Tab 3 : Carte interactive ──────────────────────
    with t3:
        import streamlit.components.v1 as components
        html_path = os.path.join(os.path.dirname(__file__), 'carte_interactive.html')
        if os.path.exists(html_path):
            with open(html_path, 'r', encoding='utf-8') as f:
                # height=600 permet d'avoir une belle taille pour interagir
                components.html(f.read(), height=600, scrolling=True) 
        else:
            st.warning("⚠️ Place carte_interactive.html dans le même dossier que app.py")
# ══════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown("## 🛣️ RoadScan-MA")
        st.markdown("*Détection automatique des dégradations routières*")
        st.divider()

        mode = st.radio(
            "📋 Mode d'analyse",
            ["🌍 TEST MEKNES", "🖼️ IMAGE", "🎬 VIDÉO"], # DÉMO supprimé, TEST en premier
            index=0,
        )
        confidence = st.slider(
            "🎯 Seuil de confiance",
            min_value=0.10, max_value=0.90,
            value=0.25, step=0.05,
        )
        st.divider()

        # Infos modèle
        model_path = "models/yolo_final_best.pt" # CHANGÉ ICI
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            st.success(f"✅ Modèle chargé ({size_mb:.1f} MB)")
        else:
            st.warning("⚠️ yolo_final_best.pt introuvable\nCopier dans models/")

        st.markdown("**5 classes détectées :**")
        for cls, label in CLASS_LABELS_FR.items():
            color = CLASS_COLORS_HEX.get(
                list(CLASS_LABELS_FR.keys()).index(cls), "#888")
            st.markdown(
                f'<span style="color:{CLASS_COLORS_HEX[list(CLASS_LABELS_FR.keys()).index(cls)]}">●</span> {label}',
                unsafe_allow_html=True,
            )
        st.divider()
        st.caption("ENSAM Meknès — IATD | v1.0")
    return mode, confidence


# ══════════════════════════════════════════════════════════
#  MÉTRIQUES EN HAUT
# ══════════════════════════════════════════════════════════
def show_metrics(summary):
    level = summary.get("severity_level", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📍 Détections totales", summary.get("total", 0))
    c2.metric(
        "⚠️ Indice de sévérité",
        f"{summary.get('severity_index', 0)}/100",
        delta=level.get("emoji", "") + " " + level.get("label", ""),
    )
    c3.metric("🕳️ Nids-de-poule", summary.get("potholes", 0))
    c4.metric("〰️ Fissures", summary.get("cracks", 0))


# ══════════════════════════════════════════════════════════
#  GRAPHIQUE PLOTLY
# ══════════════════════════════════════════════════════════
def show_plotly_chart(summary):
    counts = summary.get("counts", {})
    CLASS_NAME_TO_FR = {
    "linear_crack":    "Fissure linéaire",
    "alligator_crack": "Fissure alligator",
    "minor_pothole":   "Nid-de-poule mineur",
    "medium_pothole":  "Nid-de-poule moyen",
    "major_pothole":   "Nid-de-poule majeur",
}
    labels = [CLASS_NAME_TO_FR.get(k, k) for k in counts]    
    values = list(counts.values())
    colors = [CLASS_COLORS_HEX[i] for i in range(len(counts))]

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=values,
        textposition="outside",
        hovertemplate="%{x}<br>%{y} détections<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FAFAFA"),
        margin=dict(t=20, b=40, l=10, r=10),
        height=260,
        showlegend=False,
        yaxis=dict(gridcolor="#333"),
        xaxis=dict(tickangle=-20),
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════
#  CARTE FOLIUM
# ══════════════════════════════════════════════════════════
def show_map(detections, lat, lon, mode="image"):
    if not detections:
        st.info("📍 Aucune détection à afficher sur la carte.")
        return

    tab1, tab2 = st.tabs(["📊 Carte analyse", "📍 Terrain Meknès (réel)"])

    with tab1:
        if mode == "image":
            m = generate_image_map(detections, lat, lon)
        else:
            m = generate_video_map(detections)
        st_folium(m, width=None, height=380, returned_objects=[])

    with tab2:
        import streamlit.components.v1 as components, os
        html_path = os.path.join(os.path.dirname(__file__), 'carte_interactive.html')
        if os.path.exists(html_path):
            with open(html_path, 'r', encoding='utf-8') as f:
                components.html(f.read(), height=550, scrolling=False)
        else:
            st.warning("⚠️ Place carte_interactive.html dans le même dossier que app.py")
# ══════════════════════════════════════════════════════════
#  BOUTONS TÉLÉCHARGEMENT
# ══════════════════════════════════════════════════════════
def show_downloads(detections, summary, zone="Meknès", annotated_img=None):
    st.markdown("### ⬇️ Téléchargements")
    col_a, col_b = st.columns(2)

    # CSV
    if detections:
        with col_a:
            df = pd.DataFrame([{
                "classe":     d.get("class_name", ""),
                "label":      d.get("label_fr", ""),
                "confiance":  f"{d.get('confidence', 0):.2%}",
                "latitude":   d.get("lat", ""),
                "longitude":  d.get("lon", ""),
                "frame":      d.get("frame", ""),
            } for d in detections])
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📊 Télécharger CSV",
                data=csv_bytes,
                file_name="roadscan_detections.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # PDF
    with col_b:
        if st.button("📄 Générer rapport PDF", use_container_width=True):
            with st.spinner("Génération du PDF..."):
                with tempfile.NamedTemporaryFile(
                    suffix=".pdf", delete=False
                ) as tmp:
                    img_path = None
                    if annotated_img is not None:
                        tmp_img = tempfile.NamedTemporaryFile(
                            suffix=".jpg", delete=False
                        )
                        cv2.imwrite(tmp_img.name,
                                    cv2.cvtColor(annotated_img,
                                                 cv2.COLOR_RGB2BGR))
                        img_path = tmp_img.name

                    generate_report(summary, tmp.name, zone, img_path)
                    with open(tmp.name, "rb") as f:
                        pdf_bytes = f.read()

                st.download_button(
                    "⬇️ Télécharger le PDF",
                    data=pdf_bytes,
                    file_name="rapport_roadscan.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )


# ══════════════════════════════════════════════════════════
#  MODE IMAGE
# ══════════════════════════════════════════════════════════
def image_mode(model, confidence):
    st.markdown("## 🖼️ Analyse d'image")
    uploaded = st.file_uploader(
        "Charger une image de route (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
    )
    if uploaded is None:
        st.info("👆 Charger une image pour lancer la détection.")
        return

    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image_bgr  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if model is None:
        st.error("❌ Modèle non chargé — copier best.pt dans models/")
        return

    with st.spinner("🔍 Détection en cours..."):
        detections, annotated_bgr = predict_image(model, image_bgr, confidence)

    lat, lon = generate_image_coords()
    for d in detections:
        d["lat"] = lat
        d["lon"] = lon

    summary = format_summary(detections)

    # Métriques
    show_metrics(summary)
    st.divider()

    # Layout 2 colonnes
    col_left, col_right = st.columns([1.1, 0.9])

    with col_left:
        st.markdown("#### 📸 Résultat de détection")
        tab1, tab2 = st.tabs(["✅ Image annotée", "🖼️ Image originale"])
        with tab1:
            st.image(bgr_to_rgb(annotated_bgr), use_container_width=True)
        with tab2:
            st.image(bgr_to_rgb(image_bgr), use_container_width=True)

        if not detections:
            st.success("✅ Aucune dégradation détectée sur cette image.")
        else:
            st.markdown(f"**{len(detections)} dégradation(s) détectée(s)**")

    with col_right:
        st.markdown("#### 📊 Statistiques")
        show_plotly_chart(summary)
        st.markdown("#### 🗺️ Localisation GPS")
        show_map(detections, lat, lon, mode="image")

    st.divider()
    show_downloads(detections, summary, zone="Meknès", annotated_img=bgr_to_rgb(annotated_bgr))


# ══════════════════════════════════════════════════════════
#  MODE VIDÉO
# ══════════════════════════════════════════════════════════
def video_mode(model, confidence):
    st.markdown("## 🎬 Analyse de vidéo dashcam")
    uploaded = st.file_uploader(
        "Charger une vidéo (MP4 / MOV)",
        type=["mp4", "mov", "avi"],
    )
    if uploaded is None:
        st.info("👆 Charger une vidéo pour lancer l'analyse.")
        return

    if model is None:
        st.error("❌ Modèle non chargé — copier best.pt dans models/")
        return

    col_info, col_btn = st.columns([3, 1])
    with col_info:
        st.video(uploaded)
    with col_btn:
        st.markdown("&nbsp;")
        run = st.button("🚀 Lancer l'analyse", use_container_width=True)

    if not run:
        return

    # Sauvegarder la vidéo dans un fichier temporaire
    with tempfile.NamedTemporaryFile(
        suffix=f".{uploaded.name.split('.')[-1]}", delete=False
    ) as tmp_video:
        tmp_video.write(uploaded.read())
        video_path = tmp_video.name

    progress_bar = st.progress(0, text="⏳ Analyse en cours...")
    status_text  = st.empty()

    def update_progress(current, total):
        if total > 0:
            pct = min(int(current / total * 100), 100)
            progress_bar.progress(pct, text=f"⏳ Frame {current}/{total}")
            status_text.text(f"Traitement : {pct}%")

    with st.spinner("🎬 Traitement vidéo frame par frame..."):
        all_detections, video_summary = predict_video(
            model, video_path, confidence,
            frame_interval=10,
            progress_callback=update_progress,
        )

    progress_bar.progress(100, text="✅ Analyse terminée !")
    status_text.empty()
    os.unlink(video_path)

    # Ajouter coordonnées GPS à chaque détection
    for det in all_detections:
        lat, lon = generate_video_coords(det.get("frame", 0))
        det["lat"] = lat
        det["lon"] = lon

    summary = format_summary(all_detections)

    # Métriques
    show_metrics(summary)
    st.divider()

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("#### 📊 Répartition des dégradations")
        show_plotly_chart(summary)

        # Tableau résumé
        counts = summary.get("counts", {})
        df = pd.DataFrame({
            "Type": [CLASS_LABELS_FR[k] for k in counts],
            "Détections": list(counts.values()),
        })
        st.dataframe(df, use_container_width=True, hide_index=True)

    with col_right:
        st.markdown("#### 🗺️ Carte du parcours")
        show_map(all_detections, 33.8935, -5.5473, mode="video")

    st.divider()
    show_downloads(all_detections, summary, zone="Meknès — Parcours dashcam")


# ══════════════════════════════════════════════════════════
#  MODE DÉMO
# ══════════════════════════════════════════════════════════
def demo_mode(model, confidence):
    st.markdown("## 🎯 Mode Démonstration")
    st.info("💡 Mode sans upload — utilise les images de test locales.")

    # Chercher images dans sample_data/test_images/
    demo_dir = "sample_data/test_images"
    images   = []
    if os.path.exists(demo_dir):
        images = [
            f for f in os.listdir(demo_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

    if not images:
        st.warning(
            "⚠️ Aucune image dans sample_data/test_images/\n\n"
            "→ Copier des images de route marocaine dans ce dossier."
        )
        return

    selected = st.selectbox(
        "🖼️ Choisir une image de démonstration",
        images,
        format_func=lambda x: f"📸 {x}",
    )
    img_path = os.path.join(demo_dir, selected)

    if model is None:
        st.error("❌ Modèle non chargé — copier best.pt dans models/")
        return

    image_bgr = cv2.imread(img_path)
    if image_bgr is None:
        st.error(f"Impossible de lire : {img_path}")
        return

    with st.spinner("🔍 Détection en cours..."):
        detections, annotated_bgr = predict_image(model, image_bgr, confidence)

    lat, lon = generate_image_coords()
    for d in detections:
        d["lat"] = lat
        d["lon"] = lon

    summary = format_summary(detections)
    show_metrics(summary)
    st.divider()

    col_left, col_right = st.columns([1.1, 0.9])

    with col_left:
        st.markdown("#### 📸 Résultat")
        tab1, tab2 = st.tabs(["✅ Annoté", "🖼️ Original"])
        with tab1:
            st.image(bgr_to_rgb(annotated_bgr), use_container_width=True)
        with tab2:
            st.image(bgr_to_rgb(image_bgr), use_container_width=True)

    with col_right:
        st.markdown("#### 📊 Statistiques")
        show_plotly_chart(summary)
        st.markdown("#### 🗺️ Carte")
        show_map(detections, lat, lon, mode="image")

    st.divider()
    show_downloads(
        detections, summary,
        zone=f"Meknès — Démo ({selected})",
        annotated_img=bgr_to_rgb(annotated_bgr),
    )

# ═══════════════════════════════════════════════════════
#  TEST GLOBAL TERRAIN REEL MEKNES
# ═══════════════════════════════════════════════════════
st.markdown("---")
with st.expander("TEST GLOBAL 1 - MEKNES TERRAIN REEL", expanded=False):

    t1, t2, t3 = st.tabs(["Video", "10 Echantillons", "Carte GPS"])

    # ── Tab 1 : Video ──────────────────────────────────
    with t1:
        video_path = os.path.join(os.path.dirname(__file__), 'Video2.MOV')
        if os.path.exists(video_path):
            st.video(video_path)
        else:
            st.info("Place Video2.MOV dans le dossier de app.py")

    # ── Tab 2 : Frames annotees ────────────────────────
    with t2:
        ann_dir = os.path.join(os.path.dirname(__file__), 'annotated_frames')
        if os.path.exists(ann_dir):
            frames = sorted([f for f in os.listdir(ann_dir) if f.endswith('.jpg')])
            if frames:
                cols = st.columns(2)
                for i, fname in enumerate(frames[:10]):
                    with cols[i % 2]:
                        st.image(
                            os.path.join(ann_dir, fname),
                            caption=fname.replace('ann_', ''),
                            use_container_width=True
                        )
        else:
            st.info("Place le dossier annotated_frames/ ici")

    # ── Tab 3 : Carte interactive ──────────────────────
    with t3:
        import streamlit.components.v1 as components
        html_path = os.path.join(os.path.dirname(__file__), 'carte_interactive.html')
        if os.path.exists(html_path):
            with open(html_path, 'r', encoding='utf-8') as f:
                components.html(f.read(), height=560, scrolling=False)
        else:
            st.warning("Place carte_interactive.html dans le dossier de app.py")
# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════
def main():
    # Titre principal
    st.markdown(
        "<h1 style='color:#E74C3C;margin-bottom:0'>🛣️ RoadScan-MA</h1>"
        "<p style='color:#888;margin-top:0'>Détection automatique des dégradations routières — Meknès</p>",
        unsafe_allow_html=True,
    )

    mode, confidence = render_sidebar()
    model = load_model_cached()

    # Changement de la logique ici :
    if "TEST MEKNES" in mode:
        test_meknes_mode()
    elif "IMAGE" in mode:
        image_mode(model, confidence)
    elif "VIDÉO" in mode:
        video_mode(model, confidence)

if __name__ == "__main__":
    main()