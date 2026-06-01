"""
RoadScan-MA — map_generator.py
Génération de cartes Folium interactives avec :
  - Marqueurs colorés par classe
  - Heatmap des zones critiques
  - LayerControl (basculer entre vues)
  - MiniMap de navigation
  - Légende personnalisée
"""

import os
import base64
import folium
from folium.plugins import MarkerCluster, HeatMap, MiniMap, Fullscreen
from inference import CLASS_COLORS_HEX, CLASS_LABELS_FR


# ── Coordonnées centre Meknès ──────────────────────────────
MEKNES_LAT = 33.8935
MEKNES_LON = -5.5473

# Icônes Folium par classe
CLASS_ICONS = {
    0: ("info-sign",        "blue"),
    1: ("warning-sign",     "beige"),
    2: ("screenshot",       "green"),
    3: ("exclamation-sign", "orange"),
    4: ("remove-sign",      "red"),
}

# Poids heatmap par classe (plus critique = plus chaud)
HEATMAP_WEIGHTS = {0: 0.3, 1: 0.5, 2: 0.6, 3: 0.8, 4: 1.0}


# ── Créer la carte de base ─────────────────────────────────
def create_base_map(lat=MEKNES_LAT, lon=MEKNES_LON, zoom=14):
    """Carte Folium avec 3 couches de tuiles."""
    m = folium.Map(location=[lat, lon], zoom_start=zoom)

    # Couche 1 — OpenStreetMap (par défaut)
    folium.TileLayer("OpenStreetMap", name="🗺️ Carte standard").add_to(m)

    # Couche 2 — CartoDB sombre (style dashboard)
    folium.TileLayer(
        "CartoDB dark_matter",
        name="🌑 Carte sombre",
    ).add_to(m)

    # Couche 3 — CartoDB clair
    folium.TileLayer(
        "CartoDB positron",
        name="⬜ Carte claire",
    ).add_to(m)

    Fullscreen(
        position="topright",
        title="Plein écran",
        title_cancel="Quitter",
        force_separate_button=True,
    ).add_to(m)
    return m


# ── Groupe de marqueurs ────────────────────────────────────
def _add_markers_group(m, detections, default_lat, default_lon):
    """Groupe de marqueurs clusterisés — activé par défaut."""
    fg_markers = folium.FeatureGroup(name="📍 Marqueurs détections", show=True)
    cluster = MarkerCluster().add_to(fg_markers)

    for det in detections:
        lat = det.get("lat", default_lat)
        lon = det.get("lon", default_lon)
        cls_id = det["class_id"]
        icon_name, color = CLASS_ICONS.get(cls_id, ("info-sign", "gray"))

        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(_make_popup(det, lat, lon), max_width=250),
            tooltip=CLASS_LABELS_FR.get(cls_id, "Inconnu"),
            icon=folium.Icon(color=color, icon=icon_name, prefix="glyphicon"),
        ).add_to(cluster)

    fg_markers.add_to(m)
    return m


# ── Heatmap ────────────────────────────────────────────────
def _add_heatmap_group(m, detections, default_lat, default_lon):
    """Couche heatmap — zones critiques en rouge."""
    fg_heat = folium.FeatureGroup(name="🔥 Carte de chaleur", show=False)

    heat_data = []
    for det in detections:
        lat    = det.get("lat", default_lat)
        lon    = det.get("lon", default_lon)
        weight = HEATMAP_WEIGHTS.get(det["class_id"], 0.5)
        heat_data.append([lat, lon, weight])

    if heat_data:
        HeatMap(
            heat_data,
            min_opacity=0.4,
            radius=20,
            blur=15,
            gradient={0.3: "blue", 0.6: "orange", 1.0: "red"},
        ).add_to(fg_heat)

    fg_heat.add_to(m)
    return m


# ── MiniMap ────────────────────────────────────────────────
def _add_minimap(m):
    """Petite carte de navigation en bas à droite."""
    MiniMap(
        tile_layer="CartoDB positron",
        position="bottomright",
        width=150,
        height=120,
        collapsed_width=25,
        collapsed_height=25,
        zoom_level_offset=-5,
        toggle_display=True,
    ).add_to(m)
    return m


# ── Légende ────────────────────────────────────────────────
def _add_legend(m):
    """Légende colorée thème sombre."""
    legend_html = """
    <div style="
        position:fixed; bottom:30px; left:10px;z-index:1000;
        background:#1A1D23; padding:12px 16px; border-radius:8px;
        border:1px solid #444; font-family:sans-serif;
        font-size:12px; color:#FAFAFA; min-width:190px;">
      <b style="font-size:13px;">🛣️ RoadScan-MA</b><br><br>
      <span style="color:#3498DB;">●</span> Fissure linéaire<br>
      <span style="color:#8B4513;">●</span> Fissure alligator<br>
      <span style="color:#2ECC71;">●</span> Nid-de-poule mineur<br>
      <span style="color:#F39C12;">●</span> Nid-de-poule moyen<br>
      <span style="color:#E74C3C;">●</span> Nid-de-poule majeur<br>
      <br><i style="color:#888;font-size:11px;">
      🔥 Activer "Carte de chaleur"<br>
      pour voir les zones critiques</i>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


# ── Fonctions publiques ────────────────────────────────────
def generate_image_map(detections, lat, lon):
    """Carte complète pour le mode IMAGE."""
    m = create_base_map(lat, lon, zoom=15)
    m = _add_markers_group(m, detections, lat, lon)
    m = _add_heatmap_group(m, detections, lat, lon)
    m = _add_minimap(m)
    m = _add_legend(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def generate_video_map(detections):
    """Carte complète pour le mode VIDEO avec frames annotées dans les popups."""
    gps_pts = [(d["lat"], d["lon"]) for d in detections
               if d.get("lat") and d.get("lon")]
    if gps_pts:
        center_lat = sum(p[0] for p in gps_pts) / len(gps_pts)
        center_lon = sum(p[1] for p in gps_pts) / len(gps_pts)
    else:
        center_lat, center_lon = MEKNES_LAT, MEKNES_LON

    m = create_base_map(center_lat, center_lon, zoom=15)
    m = _add_trajectory(m, detections)
    m = _add_video_markers_group(m, detections)
    m = _add_heatmap_group(m, detections, center_lat, center_lon)
    m = _add_minimap(m)
    m = _add_legend_video(m, len(detections))
    folium.LayerControl(collapsed=False).add_to(m)
    return m


# ── Trajectoire stylisée (mode VIDEO) ─────────────────────
def _add_trajectory(m, detections):
    """Ligne de trajet avec ombre, marqueurs départ/arrivée."""
    fg = folium.FeatureGroup(name="📍 Trajectoire GPS", show=True)
    coords = [(d["lat"], d["lon"]) for d in detections
              if d.get("lat") and d.get("lon")]

    if len(coords) >= 2:
        # Ombre portée
        folium.PolyLine(
            coords, color="#000000", weight=6, opacity=0.25,
        ).add_to(fg)
        # Ligne principale
        folium.PolyLine(
            coords, color="#FF6B35", weight=3.5, opacity=0.9,
            tooltip="Trajectoire parcourue",
        ).add_to(fg)

    if coords:
        folium.CircleMarker(
            coords[0], radius=8, color="#2ECC71", fill=True,
            fill_color="#2ECC71", fill_opacity=1.0,
            tooltip="🟢 Début du trajet",
        ).add_to(fg)
    if len(coords) > 1:
        folium.CircleMarker(
            coords[-1], radius=8, color="#E74C3C", fill=True,
            fill_color="#E74C3C", fill_opacity=1.0,
            tooltip="🔴 Fin du trajet",
        ).add_to(fg)

    fg.add_to(m)
    return m


# ── Marqueurs vidéo avec popup image ─────────────────────
def _add_video_markers_group(m, detections):
    """Marqueurs cliquables — popup = frame annotée + infos."""
    fg = folium.FeatureGroup(name="📸 Détections (avec images)", show=True)
    cluster = MarkerCluster(
        options={"maxClusterRadius": 40, "disableClusteringAtZoom": 17}
    ).add_to(fg)

    for det in detections:
        lat = det.get("lat")
        lon = det.get("lon")
        if lat is None or lon is None:
            continue
        cls_id = det["class_id"]
        icon_name, color = CLASS_ICONS.get(cls_id, ("info-sign", "gray"))
        label = CLASS_LABELS_FR.get(cls_id, "Inconnu")
        conf  = det.get("confidence", 0)

        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(_make_popup_video(det, lat, lon), max_width=330),
            tooltip=f"{label} — {conf:.0%}",
            icon=folium.Icon(color=color, icon=icon_name, prefix="glyphicon"),
        ).add_to(cluster)

    fg.add_to(m)
    return m


# ── Légende professionnelle (mode VIDEO) ─────────────────
def _add_legend_video(m, total_detections):
    """Légende thème sombre avec résumé trajectoire."""
    legend_html = f"""
    <div style="
        position:fixed; bottom:30px; left:10px; z-index:1000;
        background:linear-gradient(160deg,#1A1D23 0%,#12151A 100%);
        padding:14px 18px; border-radius:12px;
        border:1px solid #2D3139; font-family:Inter,sans-serif;
        font-size:12px; color:#FAFAFA; min-width:200px;
        box-shadow:0 6px 24px rgba(0,0,0,0.55);">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
        <b style="font-size:14px;color:#FF6B35;">🛣️ RoadScan-MA</b>
      </div>
      <div style="color:#8B92A5;font-size:10px;text-transform:uppercase;
                  letter-spacing:.1em;margin-bottom:6px;">Dégradations</div>
      <span style="color:#3498DB;">●</span> Fissure linéaire<br>
      <span style="color:#8B4513;">●</span> Fissure alligator<br>
      <span style="color:#2ECC71;">●</span> Nid-de-poule mineur<br>
      <span style="color:#F39C12;">●</span> Nid-de-poule moyen<br>
      <span style="color:#E74C3C;">●</span> Nid-de-poule majeur<br>
      <hr style="border:none;border-top:1px solid #2D3139;margin:8px 0;">
      <div style="color:#8B92A5;font-size:10px;text-transform:uppercase;
                  letter-spacing:.1em;margin-bottom:6px;">Trajectoire</div>
      <span style="color:#FF6B35;font-size:14px;">━━</span> Parcours GPS<br>
      <span style="color:#2ECC71;">●</span> Départ &nbsp;
      <span style="color:#E74C3C;">●</span> Arrivée<br>
      <hr style="border:none;border-top:1px solid #2D3139;margin:8px 0;">
      <div style="color:#8B92A5;font-size:10px;">
        📍 <b style="color:#FAFAFA;">{total_detections}</b> détections<br>
        🔥 Couche chaleur disponible
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


# ── Popup HTML — mode IMAGE ────────────────────────────────
def _make_popup(det, lat, lon):
    color     = CLASS_COLORS_HEX.get(det["class_id"], "#FFFFFF")
    label     = CLASS_LABELS_FR.get(det["class_id"], "Inconnu")
    conf      = det.get("confidence", 0)
    frame     = det.get("frame", "—")
    ts        = det.get("timestamp_s", "—")
    conf_pct  = f"{conf:.0%}"
    bar_width = int(conf * 100)
    frame_row = (f"<tr><td>Frame</td><td style='color:#FAFAFA;text-align:right;'>"
                 f"{frame}</td></tr>") if frame != "—" else ""
    ts_row    = (f"<tr><td>Temps</td><td style='color:#FAFAFA;text-align:right;'>"
                 f"{ts}s</td></tr>") if ts != "—" else ""
    return f"""
    <div style="font-family:Inter,sans-serif;min-width:210px;
                background:#1E2130;border-radius:10px;
                padding:12px;color:#FAFAFA;border:1px solid #2D3139;">
      <div style="display:flex;align-items:center;margin-bottom:8px;">
        <span style="color:{color};font-size:18px;margin-right:6px;">●</span>
        <b style="font-size:13px;color:{color};">{label}</b>
      </div>
      <div style="background:#2D3139;border-radius:4px;height:4px;margin-bottom:10px;">
        <div style="background:{color};width:{bar_width}%;height:4px;border-radius:4px;"></div>
      </div>
      <table style="font-size:11px;color:#8B92A5;width:100%;">
        <tr><td>Confiance</td>
            <td style="color:#FAFAFA;text-align:right;"><b>{conf_pct}</b></td></tr>
        <tr><td>Latitude</td>
            <td style="color:#FAFAFA;text-align:right;">{lat:.5f}</td></tr>
        <tr><td>Longitude</td>
            <td style="color:#FAFAFA;text-align:right;">{lon:.5f}</td></tr>
        {frame_row}{ts_row}
      </table>
    </div>
    """


# ── Popup HTML — mode VIDEO (avec frame annotée base64) ───
def _make_popup_video(det, lat, lon):
    """Popup enrichi avec la frame annotée encodée en base64."""
    color     = CLASS_COLORS_HEX.get(det["class_id"], "#FFFFFF")
    label     = CLASS_LABELS_FR.get(det["class_id"], "Inconnu")
    conf      = det.get("confidence", 0)
    frame     = det.get("frame", "—")
    ts        = det.get("timestamp_s", "—")
    conf_pct  = f"{conf:.0%}"
    bar_width = int(conf * 100)

    # Image base64 de la frame annotée
    img_html   = ""
    frame_path = det.get("annotated_frame_path", "")
    if frame_path and os.path.exists(frame_path):
        with open(frame_path, "rb") as fh:
            img_b64 = base64.b64encode(fh.read()).decode()
        img_html = (
            f'<img src="data:image/jpeg;base64,{img_b64}" '
            f'style="width:100%;border-radius:6px;margin-bottom:10px;'
            f'display:block;">'
        )

    frame_row = (f"<tr><td>Frame</td><td style='color:#FAFAFA;text-align:right;'>"
                 f"#{frame}</td></tr>") if frame != "—" else ""
    ts_row    = (f"<tr><td>Temps</td><td style='color:#FAFAFA;text-align:right;'>"
                 f"{ts}s</td></tr>") if ts != "—" else ""

    return f"""
    <div style="font-family:Inter,sans-serif;width:310px;
                background:#1E2130;border-radius:10px;
                padding:12px;color:#FAFAFA;border:1px solid #2D3139;">
      {img_html}
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
        <span style="color:{color};font-size:16px;">●</span>
        <b style="font-size:12px;color:{color};flex:1;">{label}</b>
        <span style="background:{color}25;color:{color};font-size:10px;
                     font-weight:700;padding:2px 8px;border-radius:10px;">
          {conf_pct}
        </span>
      </div>
      <div style="background:#2D3139;border-radius:3px;height:3px;margin-bottom:8px;">
        <div style="background:{color};width:{bar_width}%;height:3px;border-radius:3px;">
        </div>
      </div>
      <table style="font-size:10px;color:#8B92A5;width:100%;border-collapse:collapse;">
        <tr><td style="padding:2px 0;">Coordonnées</td>
            <td style="color:#FAFAFA;text-align:right;padding:2px 0;">
              {lat:.5f}, {lon:.5f}</td></tr>
        {frame_row}{ts_row}
      </table>
    </div>
    """


# ── Test rapide ────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Test map_generator.py (version améliorée) ===")

    fake = [
        {"class_id": 4, "class_name": "major_pothole",  "confidence": 0.87,
         "bbox": [100,100,200,200], "color_hex": "#E74C3C",
         "lat": 33.8935, "lon": -5.5473},
        {"class_id": 3, "class_name": "medium_pothole", "confidence": 0.72,
         "bbox": [200,200,300,300], "color_hex": "#F39C12",
         "lat": 33.8940, "lon": -5.5480},
        {"class_id": 0, "class_name": "linear_crack",   "confidence": 0.65,
         "bbox": [50, 50, 150, 80],  "color_hex": "#3498DB",
         "lat": 33.8928, "lon": -5.5465},
        {"class_id": 2, "class_name": "minor_pothole",  "confidence": 0.58,
         "bbox": [300,300,400,400], "color_hex": "#2ECC71",
         "lat": 33.8950, "lon": -5.5490},
    ]

    m = generate_video_map(fake)
    m.save("test_map_v2.html")

    print("✅ Carte générée : test_map_v2.html")
    print("   Nouveautés   : Heatmap + LayerControl + MiniMap")
    print("   Marqueurs    : 4 détections")
    print("   Ouvre le fichier dans ton navigateur !")
    print("=== Fichier OK ===")