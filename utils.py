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
