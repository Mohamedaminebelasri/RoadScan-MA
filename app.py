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
from utils import (format_summary, extract_gps_exif, generate_image_coords,
                   validate_image, validate_video, validate_gpx, sync_gpx_to_frames,
                   generate_video_coords, export_csv)
from map_generator import generate_image_map, generate_video_map
from pdf_report import generate_report
from rag_advisor import load_knowledge_base, generate_recommendations, calculate_severity

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
  /* ── Fonts & Base ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  * { font-family: 'Inter', sans-serif; }

  /* ── Page layout ── */
  .block-container { padding-top: 1rem; padding-bottom: 1rem; }
  .main { background: #0E1117; }

  /* ── Sidebar pro ── */
  div[data-testid="stSidebarContent"] {
    background: linear-gradient(180deg, #1A1D23 0%, #12151A 100%);
    border-right: 1px solid #2D3139;
  }

  /* ── Métriques cards ── */
  [data-testid="stMetricValue"] {
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #FF6B35 !important;
  }
  [data-testid="stMetricLabel"] {
    font-size: 0.8rem !important;
    color: #8B92A5 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  [data-testid="metric-container"] {
    background: linear-gradient(135deg, #1E2130 0%, #16192B 100%);
    border: 1px solid #2D3139;
    border-radius: 12px;
    padding: 1rem;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    transition: transform 0.2s ease;
  }
  [data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    border-color: #FF6B35;
  }

  /* ── Boutons pro ── */
  .stButton > button {
    background: linear-gradient(135deg, #FF6B35, #E74C3C) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.5rem !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(231, 76, 60, 0.3) !important;
  }
  .stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(231, 76, 60, 0.5) !important;
  }

  /* ── Progress bar ── */
  .stProgress > div > div {
    background: linear-gradient(90deg, #FF6B35, #E74C3C) !important;
    border-radius: 10px;
  }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px 8px 0 0;
    color: #8B92A5;
    font-weight: 500;
  }
  .stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1E2130, #16192B) !important;
    color: #FF6B35 !important;
    border-bottom: 2px solid #FF6B35 !important;
  }

  /* ── Expander ── */
  .streamlit-expanderHeader {
    background: #1E2130 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
  }

  /* ── Divider ── */
  hr { border-color: #2D3139 !important; }

  /* ── Severity badge ── */
  .severity-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.9rem;
    letter-spacing: 0.02em;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
  }

  /* ── Info/Warning boxes ── */
  [data-testid="stInfo"] {
    background: rgba(255, 107, 53, 0.1) !important;
    border-left: 3px solid #FF6B35 !important;
    border-radius: 8px !important;
  }
  [data-testid="stSuccess"] {
    border-radius: 8px !important;
  }

  /* ── Dataframe ── */
  [data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden;
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
        video_path = os.path.join(os.path.dirname(__file__), 'video_demo_meknes.mp4')
        if os.path.exists(video_path):
            st.video(video_path, muted=True) # <-- AJOUTE muted=True ICI
        else:
            st.warning("⚠️ Place le fichier video_demo_meknes.mp4 dans le même dossier que app.py")

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
        html_path = os.path.join(os.path.dirname(__file__), 'carte_double_trajectoire.html')
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
        st.caption("ENSAM Meknès — IATD | Mohamed Amine Belasri & Yahya Amajane | v1.0")
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
    # Vérification GPS
    if mode == "image" and (lat is None or lon is None or (lat == 0.0 and lon == 0.0)):
        st.info("📍 Aucune donnée GPS trouvée dans cette image. "
                "La détection est disponible mais la localisation cartographique "
                "n'est pas possible sans métadonnées GPS.")
        return
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
        html_path = os.path.join(os.path.dirname(__file__), 'carte_double_trajectoire.html')
        if os.path.exists(html_path):
            with open(html_path, 'r', encoding='utf-8') as f:
                components.html(f.read(), height=550, scrolling=False)
        else:
            st.warning("⚠️ Place carte_interactive.html dans le même dossier que app.py")
# ══════════════════════════════════════════════════════════
#  BOUTONS TÉLÉCHARGEMENT
# ══════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════
#  BOUTONS IA ET TÉLÉCHARGEMENT (Version Unifiée)
# ══════════════════════════════════════════════════════════
def show_downloads(detections, summary, zone="Meknès", annotated_img=None):
    # --- 1. SECTION IA ---
    st.divider()
    st.markdown("### 🤖 Recommandations IA (Expertise DRCR)")

    # Compter les détections pour l'IA
    counts = {}
    if detections:
        for det in detections:
            cls_name = det.get("class_name", "")
            if cls_name:
                counts[cls_name] = counts.get(cls_name, 0) + 1

    # Affichage de la sévérité
    severity_info = calculate_severity(counts)
    st.markdown(f"**Indice de sévérité détecté :** <span style='color:{severity_info['color']}; font-size: 16px; font-weight: bold;'>{severity_info['score']} ({severity_info['level']})</span>", unsafe_allow_html=True)
    
    # 🔴 VOICI LE BOUTON IA !
    if st.button("🧠 Générer les recommandations IA", use_container_width=True):
        with st.spinner("Analyse experte en cours..."):
            api_key = st.session_state.get("groq_api_key", "")
            # Chargement sécurisé de la base de connaissances
            chemin_kb = os.path.join(os.path.dirname(__file__), "knowledge_base", "road_maintenance_guide.txt")
            kb_text = load_knowledge_base(chemin_kb)
            
            # Appel de l'IA et sauvegarde en mémoire
            reco = generate_recommendations(counts, kb_text, api_key)
            st.session_state["recommendations"] = reco

    # Affichage du rapport si on vient de cliquer sur le bouton
    if st.session_state.get("recommendations"):
        st.success("✅ Rapport IA généré et prêt à être inclus dans le PDF.")
        with st.expander("Voir le rapport IA généré", expanded=True):
            st.markdown(st.session_state["recommendations"])

    # --- 2. SECTION TÉLÉCHARGEMENTS ---
    st.divider()
    st.markdown("### ⬇️ Téléchargements")
    col_a, col_b = st.columns(2)

    # Bouton CSV
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

    # Bouton PDF
    with col_b:
        # Étape 1 : Préparation
        if st.button("📄 Préparer le rapport PDF", use_container_width=True):
            with st.spinner("Génération du PDF en cours..."):
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    img_path = None
                    if annotated_img is not None:
                        tmp_img = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                        cv2.imwrite(tmp_img.name, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
                        img_path = tmp_img.name

                    rapport_ia = st.session_state.get("recommendations", None)
                    generate_report(summary, tmp.name, zone, img_path, rapport_ia)
                    
                    with open(tmp.name, "rb") as f:
                        st.session_state["pdf_bytes"] = f.read()

        # Étape 2 : Téléchargement final
        if "pdf_bytes" in st.session_state:
            st.download_button(
                "⬇️ TÉLÉCHARGER LE PDF FINAL",
                data=st.session_state["pdf_bytes"],
                file_name="rapport_roadscan_expert.pdf",
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
    ok, msg = validate_image(uploaded)
    if not ok:
        st.error(msg)
        return

    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image_bgr  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if model is None:
        st.error("❌ Modèle non chargé — copier best.pt dans models/")
        return

    with st.spinner("🔍 Détection en cours..."):
        detections, annotated_bgr = predict_image(model, image_bgr, confidence)

    uploaded.seek(0)
    lat, lon = extract_gps_exif(uploaded)
    gps_reel = lat is not None
    if not gps_reel:
        lat, lon = None, None
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
    # Upload GPX optionnel
    gpx_file = st.file_uploader(
        "📍 Fichier GPX (optionnel) — pour la géolocalisation réelle",
        type=["gpx"],
        help="Enregistrez un tracé GPS avec une app comme GPSLogger pendant le tournage"
    )

    if uploaded is None:
        st.info("👆 Charger une vidéo pour lancer l'analyse.")
        return
    ok_v, msg_v = validate_video(uploaded)
    if not ok_v:
        st.error(msg_v)
        return
    elif msg_v:
        st.warning(msg_v) if "⚠️" in msg_v else st.success(msg_v)

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
        # Afficher résultats stockés si disponibles
        if st.session_state.get("video_all_detections") is not None:
            all_detections = st.session_state["video_all_detections"]
            video_summary  = st.session_state["video_summary_data"]
        else:
            return

    # Sauvegarder la vidéo localement (fix Windows OpenCV)
    import uuid, os
    ext = uploaded.name.split(".")[-1]
    video_path = os.path.abspath(f"temp_video_{uuid.uuid4().hex[:8]}.{ext}")
    video_bytes = uploaded.read()
    with open(video_path, "wb") as f:
        f.write(video_bytes)
    # Vérification fichier écrit
    if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
        st.error("❌ Erreur écriture fichier vidéo temporaire.")
        return
    st.write(f"📁 Vidéo sauvegardée : {os.path.getsize(video_path)//1024} KB")

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
    st.session_state["video_all_detections"] = all_detections
    st.session_state["video_summary_data"] = video_summary
    status_text.empty()
    os.unlink(video_path)

    # GPS réel (GPX) ou simulé
    ok_g, msg_g, n_pts = validate_gpx(gpx_file)
    if not ok_g:
        st.error(msg_g)
        for det in all_detections:
            det["lat"] = None
            det["lon"] = None
    elif gpx_file is not None and ok_g:
        method = sync_gpx_to_frames(gpx_file, all_detections)
        st.success(f"✅ GPX synchronisé ({n_pts} points, méthode : {method})")
    else:
        st.info("📍 Aucun fichier GPX — détections sans géolocalisation.")
        for det in all_detections:
            det["lat"] = None
            det["lon"] = None

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
            "Type": [CLASS_LABELS_FR.get(k, k) for k in counts],
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

    uploaded.seek(0)
    lat, lon = extract_gps_exif(uploaded)
    gps_reel = lat is not None
    if not gps_reel:
        lat, lon = None, None
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
        video_path = os.path.join(os.path.dirname(__file__), 'video_demo_meknes.mp4')
        if os.path.exists(video_path):
            st.video(video_path)
        else:
            st.info("Place video_demo_meknes.mp4 dans le dossier de app.py")

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
        html_path = os.path.join(os.path.dirname(__file__), 'carte_double_trajectoire.html')
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