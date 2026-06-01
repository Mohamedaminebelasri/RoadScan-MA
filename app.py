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

  /* ── RAG Recommendation Cards ── */
  .rag-card {
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    cursor: default;
  }
  .rag-card:hover {
    transform: translateX(5px);
    box-shadow: 0 6px 24px rgba(0,0,0,0.4) !important;
  }
  .rag-severity-header {
    transition: transform 0.2s ease;
  }
  .rag-severity-header:hover {
    transform: translateY(-2px);
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
        "linear_crack":    "Fissure lineaire",
        "alligator_crack": "Fissure alligator",
        "minor_pothole":   "Nid-de-poule mineur",
        "medium_pothole":  "Nid-de-poule moyen",
        "major_pothole":   "Nid-de-poule majeur",
    }
    labels = [CLASS_NAME_TO_FR.get(k, k) for k in counts]
    values = list(counts.values())
    colors = ["#3498DB", "#E67E22", "#2ECC71", "#E74C3C", "#9B59B6"]
    total  = sum(values)

    col1, col2 = st.columns([3, 2])

    with col1:
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=labels, y=values,
            marker=dict(color=colors[:len(values)], opacity=0.9),
            text=[f"{v} ({v/total*100:.0f}%)" if total > 0 else "0" for v in values],
            textposition="outside",
            textfont=dict(size=11, color="#FAFAFA"),
            hovertemplate="<b>%{x}</b><br>Detections : %{y}<extra></extra>",
        ))
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(14,17,23,0.5)",
            font=dict(color="#FAFAFA"),
            margin=dict(t=30, b=50, l=10, r=10),
            height=280,
            showlegend=False,
            title=dict(text="Repartition par classe", font=dict(size=13, color="#8B92A5"), x=0),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False),
            xaxis=dict(tickangle=-15, tickfont=dict(size=10)),
            bargap=0.3,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        if total > 0:
            active_labels = [l for l, v in zip(labels, values) if v > 0]
            active_values = [v for v in values if v > 0]
            active_colors = [c for c, v in zip(colors, values) if v > 0]
            fig_pie = go.Figure(go.Pie(
                labels=active_labels, values=active_values,
                hole=0.6,
                marker=dict(colors=active_colors, line=dict(color="#0E1117", width=2)),
                textinfo="percent",
                hovertemplate="<b>%{label}</b><br>%{value} det.<extra></extra>",
            ))
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#FAFAFA"),
                margin=dict(t=30, b=10, l=10, r=10),
                height=280, showlegend=False,
                title=dict(text="Distribution", font=dict(size=13, color="#8B92A5"), x=0.1),
                annotations=[dict(
                    text=f"<b>{total}</b><br>total",
                    x=0.5, y=0.5,
                    font=dict(size=14, color="#FF6B35"),
                    showarrow=False
                )]
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Aucune detection")


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
#  AFFICHAGE RECOMMANDATIONS RAG
# ══════════════════════════════════════════════════════════

_SECTION_CONFIG = {
    "Diagnostic":                 {"icon": "🔍", "color": "#3498DB"},
    "Niveau d'urgence":           {"icon": "⏱️", "color": "#E74C3C"},
    "Interventions recommandees": {"icon": "🔧", "color": "#E67E22"},
    "Interventions recommandées": {"icon": "🔧", "color": "#E67E22"},
    "Calendrier d'intervention":  {"icon": "📅", "color": "#9B59B6"},
    "Mesures preventives":        {"icon": "🛡️", "color": "#27AE60"},
    "Mesures préventives":        {"icon": "🛡️", "color": "#27AE60"},
    "Bilan":                      {"icon": "📋", "color": "#8B92A5"},
}

_SEVERITY_ICONS = {
    "Aucune":   "✅",
    "Faible":   "⚠️",
    "Modere":   "🟠",
    "Eleve":    "🔴",
    "CRITIQUE": "🚨",
}


def _parse_sections(text):
    """Découpe le markdown en dict {titre: contenu}."""
    import re
    text = re.sub(r'^\*\(Genere hors-ligne\)\*\s*\n?', '', text.strip())
    parts = re.split(r'\n##\s+', '\n' + text)
    sections = {}
    for part in parts:
        if not part.strip():
            continue
        lines = part.strip().split('\n', 1)
        title = lines[0].strip().lstrip('#').strip()
        content = lines[1].strip() if len(lines) > 1 else ""
        if title:
            sections[title] = content
    return sections


def _md_to_html(text):
    """Convertit le markdown basique en HTML inline pour les cartes."""
    import re
    lines = text.split('\n')
    result = []
    in_list = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(('- ', '* ')):
            if not in_list:
                result.append('<ul style="margin:0.4rem 0;padding-left:1.3rem;">')
                in_list = True
            item = stripped[2:]
            item = re.sub(r'\*\*(.+?)\*\*', r'<strong style="color:#FAFAFA;">\1</strong>', item)
            result.append(f'<li style="margin-bottom:0.3rem;">{item}</li>')
        else:
            if in_list:
                result.append('</ul>')
                in_list = False
            if stripped:
                stripped = re.sub(r'\*\*(.+?)\*\*', r'<strong style="color:#FAFAFA;">\1</strong>', stripped)
                result.append(f'<p style="margin:0.25rem 0;">{stripped}</p>')
    if in_list:
        result.append('</ul>')
    return ''.join(result)


def _render_recommendation_cards(text, sev_color):
    """Affiche chaque section du rapport IA sous forme de carte colorée."""
    st.markdown(
        "<div style='display:flex;align-items:center;gap:0.6rem;margin:1.25rem 0 0.85rem;'>"
        "<span style='font-size:1.2rem;'>📋</span>"
        "<span style='color:#FAFAFA;font-size:1rem;font-weight:700;'>Rapport d'expertise IA</span>"
        "<span style='background:#27AE60;color:#fff;font-size:0.65rem;font-weight:700;"
        "padding:3px 10px;border-radius:20px;margin-left:0.5rem;letter-spacing:0.05em;'>✓ GÉNÉRÉ</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    sections = _parse_sections(text)
    if not sections:
        st.markdown(text)
        return

    for raw_title, content in sections.items():
        title_key = raw_title.split(':')[0].strip()
        cfg = _SECTION_CONFIG.get(title_key, {"icon": "📌", "color": sev_color})
        card_color = cfg["color"]
        card_icon  = cfg["icon"]
        html_content = _md_to_html(content) if content else "<em style='color:#8B92A5;'>—</em>"

        st.markdown(f"""
        <div class="rag-card" style="
            background: linear-gradient(135deg, {card_color}18 0%, {card_color}06 100%);
            border-left: 4px solid {card_color};
            border-radius: 0 12px 12px 0;
            padding: 1.1rem 1.5rem;
            margin-bottom: 0.85rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        ">
            <div style="display:flex;align-items:center;gap:0.65rem;margin-bottom:0.65rem;">
                <span style="
                    background:{card_color}28;border-radius:8px;
                    padding:4px 9px;font-size:1.1rem;
                ">{card_icon}</span>
                <span style="
                    color:{card_color};font-size:0.88rem;font-weight:700;
                    text-transform:uppercase;letter-spacing:0.07em;
                ">{raw_title}</span>
            </div>
            <div style="color:#C5CAD3;font-size:0.87rem;line-height:1.75;">
                {html_content}
            </div>
        </div>
        """, unsafe_allow_html=True)


def show_rag_recommendations(counts):
    """Affiche la section IA avec design professionnel et cartes colorées par sévérité."""
    severity_info = calculate_severity(counts)
    sev_level   = severity_info.get("level", "Aucune")
    sev_color   = severity_info.get("color", "#28a745")
    sev_score   = severity_info.get("score", 0)
    sev_urgency = severity_info.get("urgency", "")
    sev_icon    = _SEVERITY_ICONS.get(sev_level, "⚠️")

    # ── Carte de sévérité globale ───────────────────────────
    st.markdown(f"""
    <div class="rag-severity-header" style="
        background: linear-gradient(135deg, {sev_color}22 0%, {sev_color}08 100%);
        border: 2px solid {sev_color};
        border-radius: 16px;
        padding: 1.25rem 1.75rem;
        margin-bottom: 1.1rem;
        box-shadow: 0 4px 20px {sev_color}28;
        display: flex; align-items: center; gap: 1.25rem;
    ">
        <div style="font-size:2.8rem;line-height:1;flex-shrink:0;">{sev_icon}</div>
        <div style="flex:1;">
            <div style="color:#8B92A5;font-size:0.7rem;text-transform:uppercase;
                letter-spacing:0.12em;margin-bottom:0.2rem;">INDICE DE SÉVÉRITÉ</div>
            <div style="display:flex;align-items:baseline;gap:0.8rem;flex-wrap:wrap;">
                <span style="color:{sev_color};font-size:2.3rem;font-weight:800;line-height:1;">
                    {sev_score}
                </span>
                <span style="
                    background:{sev_color};color:#fff;
                    font-size:0.72rem;font-weight:700;
                    padding:3px 11px;border-radius:20px;
                    text-transform:uppercase;letter-spacing:0.09em;
                    align-self:center;
                ">{sev_level}</span>
            </div>
            <div style="color:#8B92A5;font-size:0.82rem;margin-top:0.4rem;">
                ⏱️ {sev_urgency}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Bouton de génération ────────────────────────────────
    if st.button("🧠 Générer les recommandations IA", use_container_width=True):
        with st.spinner("Analyse experte en cours..."):
            api_key   = st.session_state.get("groq_api_key", "")
            chemin_kb = os.path.join(os.path.dirname(__file__), "knowledge_base", "road_maintenance_guide.txt")
            kb_text   = load_knowledge_base(chemin_kb)
            reco = generate_recommendations(counts, kb_text, api_key)
            st.session_state["recommendations"] = reco

    # ── Rendu des cartes ────────────────────────────────────
    if st.session_state.get("recommendations"):
        _render_recommendation_cards(st.session_state["recommendations"], sev_color)


# ══════════════════════════════════════════════════════════
#  BOUTONS IA ET TÉLÉCHARGEMENT (Version Unifiée)
# ══════════════════════════════════════════════════════════
def show_downloads(detections, summary, zone="Meknès", annotated_img=None):
    # --- 1. SECTION IA ---
    st.divider()
    st.markdown("### 🤖 Recommandations IA (Expertise DRCR)")

    counts = {}
    if detections:
        for det in detections:
            cls_name = det.get("class_name", "")
            if cls_name:
                counts[cls_name] = counts.get(cls_name, 0) + 1

    show_rag_recommendations(counts)

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
                    st.session_state.pop("pdf_bytes", None)
                    generate_report(summary, tmp.name, zone, img_path, rapport_ia)
                    
                    with open(tmp.name, "rb") as f:
                        st.session_state["pdf_bytes"] = f.read()

        # Étape 2 : Téléchargement final
    col_reset, col_dl = st.columns([1, 3])
    with col_reset:
        if st.button("🔄 Nouveau PDF"):
            del st.session_state["pdf_bytes"]
            st.rerun()
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