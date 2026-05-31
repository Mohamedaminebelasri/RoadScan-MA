"""
RoadScan-MA — map_generator.py
Génération de cartes Folium interactives avec :
  - Marqueurs colorés par classe
  - Heatmap des zones critiques
  - LayerControl (basculer entre vues)
  - MiniMap de navigation
  - Légende personnalisée
"""

import folium
from folium.plugins import MarkerCluster, HeatMap, MiniMap
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
    """Carte complète pour le mode VIDÉO."""
    m = create_base_map()
    m = _add_markers_group(m, detections, MEKNES_LAT, MEKNES_LON)
    m = _add_heatmap_group(m, detections, MEKNES_LAT, MEKNES_LON)
    m = _add_minimap(m)
    m = _add_legend(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m


# ── Popup HTML ─────────────────────────────────────────────
def _make_popup(det, lat, lon):
    color = CLASS_COLORS_HEX.get(det["class_id"], "#FFFFFF")
    label = CLASS_LABELS_FR.get(det["class_id"], "Inconnu")
    conf  = det.get("confidence", 0)
    frame = det.get("frame", "—")
    ts    = det.get("timestamp_s", "—")

    return f"""
    <div style="font-family:sans-serif;font-size:13px;min-width:180px;">
      <b style="color:{color};font-size:14px;">● {label}</b><br><br>
      <b>Confiance :</b> {conf:.0%}<br>
      <b>Latitude  :</b> {lat:.6f}<br>
      <b>Longitude :</b> {lon:.6f}<br>
      {"<b>Frame :</b> " + str(frame) + "<br>" if frame != "—" else ""}
      {"<b>Temps :</b> " + str(ts) + "s<br>" if ts != "—" else ""}
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