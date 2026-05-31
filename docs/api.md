# Référence API

## inference.py

### `RoadDetector`

Classe principale pour l'inférence YOLOv8 avec prétraitement CLAHE.

```python
from inference import RoadDetector

detector = RoadDetector(model_path="models/best.pt", conf=0.25)
results = detector.predict(image)
```

#### `predict(image)`

| Paramètre | Type | Description |
|---|---|---|
| `image` | `np.ndarray` | Image BGR (OpenCV) |

**Retourne :** Liste de détections `[{class, confidence, bbox}]`

---

## map_generator.py

### `generate_map(detections, gps_coords)`

Génère une carte Folium interactive.

```python
from map_generator import generate_map

folium_map = generate_map(detections=results, gps_coords=coords)
folium_map.save("carte.html")
```

| Paramètre | Type | Description |
|---|---|---|
| `detections` | `list` | Détections YOLO |
| `gps_coords` | `list[tuple]` | Liste (lat, lon) |

**Retourne :** Objet `folium.Map`

---

## pdf_report.py

### `generate_pdf(detections, map_path, recommendations)`

Génère un rapport PDF complet.

```python
from pdf_report import generate_pdf

pdf_bytes = generate_pdf(
    detections=results,
    map_path="carte.html",
    recommendations=rapport_ia
)
```

**Retourne :** `bytes` — contenu PDF téléchargeable

---

## rag_advisor.py

### `load_knowledge_base(path)`

```python
from rag_advisor import load_knowledge_base
kb = load_knowledge_base("knowledge_base/road_maintenance_guide.txt")
```

### `calculate_severity(detections)`

```python
from rag_advisor import calculate_severity

severity = calculate_severity({
    "major_pothole": 2,
    "alligator_crack": 1
})
# {"score": 13, "level": "CRITIQUE", "color": "#8b0000", "urgency": "..."}
```

### `generate_recommendations(detections, kb_text, api_key)`

```python
from rag_advisor import generate_recommendations

rapport = generate_recommendations(
    detections={"major_pothole": 1, "linear_crack": 3},
    kb_text=kb,
    api_key="gsk_..."
)
print(rapport)  # Markdown formaté
```

| Paramètre | Type | Description |
|---|---|---|
| `detections` | `dict` | `{classe: count}` |
| `kb_text` | `str` | Texte référentiel technique |
| `api_key` | `str` | Clé Groq (ou via `.env`) |

**Retourne :** `str` — Rapport en Markdown

---

## Classes de dégradation

| ID | Classe | Poids | Description |
|---|---|---|---|
| 0 | `linear_crack` | 1 | Fissure longitudinale ou transversale |
| 1 | `alligator_crack` | 3 | Réseau de fissures interconnectées |
| 2 | `minor_pothole` | 2 | Cavité < 5 cm de profondeur |
| 3 | `medium_pothole` | 4 | Cavité 5-10 cm |
| 4 | `major_pothole` | 5 | Cavité > 10 cm — danger immédiat |
