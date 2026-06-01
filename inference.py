"""
RoadScan-MA — inference.py
Wrapper YOLO pour la détection des 5 types de dégradations routières.

5 classes du modèle final :
    0 → linear_crack     (fissures linéaires)   #3498DB bleu
    1 → alligator_crack  (fissures crocodile)   #8B4513 marron
    2 → minor_pothole    (petit trou)            #2ECC71 vert
    3 → medium_pothole   (trou moyen)            #F39C12 orange
    4 → major_pothole    (grand trou)            #E74C3C rouge
"""

from __future__ import annotations  # annotations lazy → YOLO n'a pas besoin d'être importé au niveau module

import os
import cv2
import numpy as np
from pathlib import Path
# ultralytics est importé en lazy dans load_model() :
# évite ~3-5 s de délai au démarrage de Streamlit quand le modèle n'est pas encore demandé.


# ── Configuration des 5 classes ────────────────────────────
CLASS_NAMES = {
    0: "linear_crack",
    1: "alligator_crack",
    2: "minor_pothole",
    3: "medium_pothole",
    4: "major_pothole",
}

CLASS_LABELS_FR = {
    0: "Fissure linéaire",
    1: "Fissure alligator",
    2: "Nid-de-poule mineur",
    3: "Nid-de-poule moyen",
    4: "Nid-de-poule majeur",
}

# Couleurs BGR pour OpenCV (attention : OpenCV = BGR, pas RGB)
CLASS_COLORS_BGR = {
    0: (219, 152,  52),   # Bleu   #3498DB
    1: ( 19,  69, 139),   # Marron #8B4513
    2: (113, 204,  46),   # Vert   #2ECC71
    3: ( 18, 156, 243),   # Orange #F39C12
    4: ( 60,  76, 231),   # Rouge  #E74C3C
}

# Couleurs HEX pour Streamlit / Folium
CLASS_COLORS_HEX = {
    0: "#3498DB",
    1: "#8B4513",
    2: "#2ECC71",
    3: "#F39C12",
    4: "#E74C3C",
}

# Poids de sévérité pour le calcul de l'indice global
SEVERITY_WEIGHTS = {
    0: 0,   # fissures : pas comptées dans l'indice pothole
    1: 0,
    2: 1,   # minor × 1
    3: 2,   # medium × 2
    4: 3,   # major  × 3
}


# ── Chargement du modèle ───────────────────────────────────
def load_model(model_path: str = "models/yolo_final_best.pt") -> YOLO:
    """Charge le modèle YOLO depuis le fichier .pt."""
    from ultralytics import YOLO  # import lazy : n'exécuté qu'au premier appel
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {model_path}\n"
            f"→ Copier best.pt dans le dossier models/"
        )
    return YOLO(str(path))
def apply_clahe(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab_eq = cv2.merge([clahe.apply(l), a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
# ── Filtre Solaire (Calibré pour Meknès) ───────────────────
def appliquer_filtre_solaire(image_bgr, contraste=0.80, luminosite=-20, gamma=0.80):
    """Réduit l'éblouissement pour aider YOLO à voir la route."""
    # Contraste et Luminosité
    img_adj = cv2.convertScaleAbs(image_bgr, alpha=contraste, beta=luminosite)
    
    # Correction Gamma
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img_adj, table)
# ── Inférence sur une image ────────────────────────────────
def predict_image(model: YOLO, image: np.ndarray, confidence: float = 0.25):
    """
    Lance la détection YOLO sur une image NumPy (BGR).
    Applique CLAHE pour gérer la surexposition (soleil Meknes).
    """
    image_processed = apply_clahe(image)

    results = model.predict(
        source=image_processed,
        conf=confidence,
        verbose=False,
    )[0]

    detections = []
    annotated  = image_processed.copy()

    if results.boxes is not None:
        for box in results.boxes:
            cls_id     = int(box.cls[0].item())
            conf_score = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            detection = {
                "class_id":   cls_id,
                "class_name": CLASS_NAMES.get(cls_id, f"class_{cls_id}"),
                "label_fr":   CLASS_LABELS_FR.get(cls_id, "Inconnu"),
                "confidence": round(conf_score, 3),
                "bbox":       [x1, y1, x2, y2],
                "color_hex":  CLASS_COLORS_HEX.get(cls_id, "#FFFFFF"),
            }
            detections.append(detection)
            annotated = _draw_bbox(annotated, detection)

    return detections, annotated


# ── Inférence sur une vidéo ────────────────────────────────
def _save_annotated_frame(annotated: np.ndarray, frame_idx: int,
                           save_dir: str, detections: list) -> None:
    """Redimensionne et sauvegarde la frame annotée ; stocke le chemin dans chaque détection."""
    os.makedirs(save_dir, exist_ok=True)
    h, w = annotated.shape[:2]
    if w > 400:
        new_w, new_h = 400, int(h * 400 / w)
        annotated = cv2.resize(annotated, (new_w, new_h))
    path = os.path.join(save_dir, f"frame_{frame_idx:06d}.jpg")
    cv2.imwrite(path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 72])
    for det in detections:
        det["annotated_frame_path"] = path


def predict_video(
    model: YOLO,
    video_path: str,
    confidence: float = 0.25,
    frame_interval: int = 10,
    progress_callback=None,
    save_frames_dir: str = None,
):
    """
    Traite une vidéo frame par frame.

    Args:
        frame_interval : traiter 1 frame toutes les N frames (défaut=10)
        progress_callback : fonction(current, total) pour la barre Streamlit

    Retourne :
        all_detections (list) : toutes les détections avec numéro de frame
        summary (dict)        : statistiques globales
    """
    # Tentative OpenCV (H.264)
    cap = cv2.VideoCapture(video_path)
    use_imageio = not cap.isOpened()
    if not use_imageio:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

    # Fallback imageio (H.265 WhatsApp)
    if use_imageio:
        try:
            import imageio.v3 as iio
            reader = iio.imopen(video_path, "r", plugin="pyav")
            meta = iio.improps(video_path, plugin="pyav")
            total_frames = meta.n_images if hasattr(meta, "n_images") else 999
            fps = 30.0
        except Exception as e:
            raise ValueError(f"Impossible d'ouvrir la vidéo : {video_path}\n"
                             f"Installez : pip install imageio[ffmpeg]\nErreur: {e}")

    all_detections = []
    frame_idx = 0

    if use_imageio:
        import imageio.v3 as iio
        for frame_rgb in iio.imiter(video_path, plugin="pyav"):
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            if frame_idx % frame_interval == 0:
                detections, annotated = predict_image(model, frame, confidence)
                if save_frames_dir and detections:
                    _save_annotated_frame(annotated, frame_idx, save_frames_dir, detections)
                for det in detections:
                    det["frame"] = frame_idx
                    det["timestamp_s"] = round(frame_idx / fps, 2)
                    all_detections.append(det)
                if progress_callback:
                    progress_callback(frame_idx, total_frames)
            frame_idx += 1
    else:
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                detections, annotated = predict_image(model, frame, confidence)
                if save_frames_dir and detections:
                    _save_annotated_frame(annotated, frame_idx, save_frames_dir, detections)
                for det in detections:
                    det["frame"] = frame_idx
                    det["timestamp_s"] = round(frame_idx / fps, 2)
                    all_detections.append(det)
                if progress_callback:
                    progress_callback(frame_idx, total_frames)
            frame_idx += 1
        cap.release()

    summary = _compute_summary(all_detections)
    return all_detections, summary


# ── Résumé statistique ────────────────────────────────────
def _compute_summary(detections: list) -> dict:
    """
    Calcule les statistiques à partir d'une liste de détections.
    """
    counts = {name: 0 for name in CLASS_NAMES.values()}
    for det in detections:
        name = det.get("class_name", "")
        if name in counts:
            counts[name] += 1

    total = len(detections)

    # Indice de sévérité (0-100) basé sur les potholes
    pothole_detections = [
        d for d in detections if d["class_id"] in (2, 3, 4)
    ]
    if pothole_detections:
        score_raw = sum(SEVERITY_WEIGHTS[d["class_id"]] for d in pothole_detections)
        severity_index = round((score_raw / (len(pothole_detections) * 3)) * 100, 1)
    else:
        severity_index = 0.0

    cracks   = counts["linear_crack"] + counts["alligator_crack"]
    potholes = counts["minor_pothole"] + counts["medium_pothole"] + counts["major_pothole"]

    return {
        "total":          total,
        "counts":         counts,
        "cracks":         cracks,
        "potholes":       potholes,
        "severity_index": severity_index,
    }


# ── Dessin des bboxes ─────────────────────────────────────
def _draw_bbox(image: np.ndarray, detection: dict) -> np.ndarray:
    """
    Dessine une bounding box colorée sur l'image.
    """
    x1, y1, x2, y2 = detection["bbox"]
    color           = CLASS_COLORS_BGR.get(detection["class_id"], (255, 255, 255))
    label           = f"{detection['label_fr']} {detection['confidence']:.0%}"

    # Rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)

    # Fond du label
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(image, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)

    # Texte
    cv2.putText(
        image, label,
        (x1 + 2, y1 - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (255, 255, 255), 1, cv2.LINE_AA,
    )
    return image


# ── Utilitaire : BGR NumPy → RGB PIL ──────────────────────
def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convertit une image OpenCV BGR en RGB pour l'affichage Streamlit."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ── Test rapide (lance : python inference.py) ─────────────
if __name__ == "__main__":
    import sys
    print("=== Test inference.py ===")
    try:
        model = load_model("models/yolo_final_best.pt")
        print(f"✅ Modèle chargé")
        print(f"   Classes : {model.names}")
    except FileNotFoundError as e:
        print(f"⚠️  {e}")
        print("   (Normal si best.pt pas encore disponible)")
    print("=== Fichier OK ===")
