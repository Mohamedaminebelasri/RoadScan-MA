"""
RoadScan-MA — utils.py
Fonctions utilitaires partagées entre app.py et map_generator.py
"""

import random
import numpy as np
from inference import CLASS_COLORS_HEX, CLASS_LABELS_FR, SEVERITY_WEIGHTS


def compute_severity_index(detections: list) -> float:
    """Indice de sévérité global 0-100 incluant fissures et potholes."""
    _weights = {0: 0.5, 1: 1.0, 2: 1.0, 3: 2.0, 4: 3.0}
    if not detections:
        return 0.0
    score_raw = sum(_weights.get(d["class_id"], 0) for d in detections)
    return round((score_raw / (len(detections) * 3)) * 100, 1)


# ── Niveau de sévérité textuel ─────────────────────────────
def severity_level(index: float) -> dict:
    """
    Convertit l'indice numérique en niveau lisible + couleur.

    Retourne un dict {"label": str, "color": str, "emoji": str}
    """
    if index == 0:
        return {"label": "Aucune dégradation", "color": "#2ECC71", "emoji": "✅"}
    elif index < 30:
        return {"label": "Faible",             "color": "#2ECC71", "emoji": "🟢"}
    elif index < 60:
        return {"label": "Modérée",            "color": "#F39C12", "emoji": "🟡"}
    elif index < 80:
        return {"label": "Élevée",             "color": "#E67E22", "emoji": "🟠"}
    else:
        return {"label": "Critique",           "color": "#E74C3C", "emoji": "🔴"}



# ── Extraction GPS réel depuis EXIF ───────────────────────
def extract_gps_exif(image_file) -> tuple:
    """
    Tente d'extraire les coordonnées GPS depuis les métadonnées EXIF.
    Retourne (lat, lon) si trouvé, sinon (None, None).
    """
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS, GPSTAGS
        img = Image.open(image_file)
        exif_data = img._getexif()
        if not exif_data:
            return None, None
        gps_info = {}
        for tag, value in exif_data.items():
            if TAGS.get(tag) == "GPSInfo":
                for gps_tag, gps_val in value.items():
                    gps_info[GPSTAGS.get(gps_tag, gps_tag)] = gps_val
        if not gps_info or "GPSLatitude" not in gps_info:
            return None, None
        def to_deg(v):
            return float(v[0]) + float(v[1])/60 + float(v[2])/3600
        lat = to_deg(gps_info["GPSLatitude"])
        lon = to_deg(gps_info["GPSLongitude"])
        if gps_info.get("GPSLatitudeRef") == "S":  lat = -lat
        if gps_info.get("GPSLongitudeRef") == "W": lon = -lon
        return round(lat, 6), round(lon, 6)
    except Exception:
        return None, None

# ── Coordonnées GPS simulées pour les images ──────────────
def generate_image_coords(
    base_lat: float = 33.8935,
    base_lon: float = -5.5473,
    spread: float = 0.01,
) -> tuple:
    """
    Génère des coordonnées GPS fictives autour de Meknès.
    Utilisé en mode IMAGE (pas de vrai GPS disponible).
    """
    lat = base_lat + random.uniform(-spread, spread)
    lon = base_lon + random.uniform(-spread, spread)
    return round(lat, 6), round(lon, 6)


# ── Coordonnées GPS simulées pour les vidéos ──────────────
def generate_video_coords(
    frame_index: int,
    base_lat: float = 33.8935,
    base_lon: float = -5.5473,
    step: float = 0.0001,
) -> tuple:
    """
    Simule un déplacement GPS linéaire pour les vidéos.
    +0.0001° de latitude par frame = ~11m entre chaque point.
    """
    lat = base_lat + (frame_index * step)
    lon = base_lon + random.uniform(-0.0002, 0.0002)
    return round(lat, 6), round(lon, 6)



def validate_image(uploaded):
    """Valide une image uploadée et retourne (ok, message)."""
    MAX_SIZE_MB = 10
    FORMATS_OK  = ["jpg", "jpeg", "png", "bmp", "webp"]
    ext  = uploaded.name.split(".")[-1].lower()
    size = uploaded.size / (1024 * 1024)
    if ext not in FORMATS_OK:
        return False, (f"❌ Format **{ext.upper()}** non supporté. "
                       f"Formats acceptés : {', '.join(f.upper() for f in FORMATS_OK)}")
    if size > MAX_SIZE_MB:
        return False, f"❌ Image trop lourde ({size:.1f} MB). Maximum : {MAX_SIZE_MB} MB"
    return True, "✅ Image valide"

def validate_video(uploaded):
    """Valide une vidéo uploadée et retourne (ok, message)."""
    MAX_SIZE_MB  = 500
    FORMATS_OK   = ["mp4", "avi", "mov"]
    CODECS_WARN  = ["whatsapp", "hevc", "h265"]
    ext  = uploaded.name.split(".")[-1].lower()
    size = uploaded.size / (1024 * 1024)
    if ext not in FORMATS_OK:
        return False, (f"❌ Format **{ext.upper()}** non supporté. "
                       f"Formats acceptés : MP4, AVI, MOV (H.264)")
    if size > MAX_SIZE_MB:
        return False, (f"❌ Vidéo trop lourde ({size:.0f} MB). "
                       f"Maximum : {MAX_SIZE_MB} MB. Utilisez un clip court.")
    if size > 100:
        return True, (f"⚠️ Vidéo volumineuse ({size:.0f} MB) — "
                      f"traitement estimé : {size/50:.0f} min sur CPU.")
    return True, f"✅ Vidéo valide ({size:.1f} MB)"

def validate_gpx(uploaded):
    """Valide un fichier GPX et retourne (ok, message, points_count)."""
    if uploaded is None:
        return True, "ℹ️ Pas de fichier GPX — coordonnées simulées.", 0
    ext = uploaded.name.split(".")[-1].lower()
    if ext != "gpx":
        return False, f"❌ Format **{ext.upper()}** non valide. Attendu : .GPX", 0
    if uploaded.size == 0:
        return False, "❌ Fichier GPX vide.", 0
    try:
        import xml.etree.ElementTree as ET
        uploaded.seek(0)
        tree = ET.parse(uploaded)
        points = tree.findall(".//{*}trkpt")
        if len(points) == 0:
            return False, "❌ Fichier GPX sans points GPS (trkpt). Vérifiez l'export.", 0
        uploaded.seek(0)
        return True, f"✅ GPX valide — {len(points)} points GPS trouvés", len(points)
    except Exception as e:
        return False, f"❌ Fichier GPX corrompu : {e}", 0


def sync_gpx_to_frames(gpx_file, all_detections, fps=30.0):
    """
    Synchronise les points GPX aux frames vidéo par timestamps.
    Fallback proportionnel si pas de timestamps dans le GPX.
    """
    import xml.etree.ElementTree as ET
    from datetime import datetime

    gpx_file.seek(0)
    tree = ET.parse(gpx_file)
    points = tree.findall(".//{*}trkpt")

    # Essai synchro par timestamps
    timed_points = []
    for p in points:
        time_el = p.find("{*}time")
        lat = float(p.get("lat"))
        lon = float(p.get("lon"))
        if time_el is not None:
            try:
                t = datetime.fromisoformat(
                    time_el.text.replace("Z", "+00:00")
                )
                timed_points.append((t, lat, lon))
            except:
                timed_points.append((None, lat, lon))
        else:
            timed_points.append((None, lat, lon))

    has_timestamps = all(t[0] is not None for t in timed_points)
    total_pts   = len(timed_points)
    total_frames = max((d.get("frame", 0) for d in all_detections), default=1)

    if has_timestamps and len(timed_points) >= 2:
        # Synchro par interpolation temporelle
        t_start = timed_points[0][0]
        t_end   = timed_points[-1][0]
        total_gpx_secs = (t_end - t_start).total_seconds()

        for det in all_detections:
            frame_secs = det.get("frame", 0) / fps
            ratio = min(frame_secs / max(total_gpx_secs, 1), 1.0)
            idx = int(ratio * (total_pts - 1))
            det["lat"], det["lon"] = timed_points[idx][1], timed_points[idx][2]
        return "timestamp"
    else:
        # Fallback proportionnel
        for det in all_detections:
            ratio = det.get("frame", 0) / max(total_frames, 1)
            idx = int(ratio * (total_pts - 1))
            det["lat"], det["lon"] = timed_points[idx][1], timed_points[idx][2]
        return "proportionnel"

# ── Comptage par classe ────────────────────────────────────
def count_by_class(detections: list) -> dict:
    """
    Retourne un dict {class_name: count} pour toutes les détections.
    """
    counts = {
        "linear_crack":    0,
        "alligator_crack": 0,
        "minor_pothole":   0,
        "medium_pothole":  0,
        "major_pothole":   0,
    }
    for det in detections:
        name = det.get("class_name", "")
        if name in counts:
            counts[name] += 1
    return counts


# ── Résumé formaté pour l'affichage ───────────────────────
def format_summary(detections: list) -> dict:
    """
    Prépare toutes les statistiques pour l'affichage Streamlit.
    """
    counts   = count_by_class(detections)
    severity = compute_severity_index(detections)
    level    = severity_level(severity)

    cracks   = counts["linear_crack"] + counts["alligator_crack"]
    potholes = counts["minor_pothole"] + counts["medium_pothole"] + counts["major_pothole"]

    return {
        "total":           len(detections),
        "cracks":          cracks,
        "potholes":        potholes,
        "counts":          counts,
        "severity_index":  severity,
        "severity_level":  level,
        "colors":          CLASS_COLORS_HEX,
        "labels_fr":       CLASS_LABELS_FR,
    }

# ── Export CSV ─────────────────────────────────────────────
def export_csv(detections: list, filepath: str = "detections.csv") -> str:
    """
    Exporte les détections en fichier CSV téléchargeable.
    Colonnes : classe, label_fr, confiance, lat, lon, frame, timestamp
    """
    import csv
    rows = []
    for det in detections:
        rows.append({
            "classe":     det.get("class_name", ""),
            "label":      det.get("label_fr", ""),
            "confiance":  f"{det.get('confidence', 0):.2%}",
            "latitude":   det.get("lat", ""),
            "longitude":  det.get("lon", ""),
            "frame":      det.get("frame", ""),
            "timestamp_s": det.get("timestamp_s", ""),
        })

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    return filepath
# ── Test rapide ────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Test utils.py ===")

    # Simuler quelques détections
    fake_detections = [
        {"class_id": 2, "class_name": "minor_pothole"},
        {"class_id": 3, "class_name": "medium_pothole"},
        {"class_id": 4, "class_name": "major_pothole"},
        {"class_id": 0, "class_name": "linear_crack"},
    ]

    summary = format_summary(fake_detections)
    level   = summary["severity_level"]

    print(f"Total détections : {summary['total']}")
    print(f"Potholes         : {summary['potholes']}")
    print(f"Fissures         : {summary['cracks']}")
    print(f"Sévérité         : {summary['severity_index']}/100 "
          f"{level['emoji']} {level['label']}")
    print(f"GPS simulé       : {generate_image_coords()}")
    print("=== Fichier OK ===")
