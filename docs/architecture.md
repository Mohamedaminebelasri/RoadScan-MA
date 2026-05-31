# Architecture technique

## Vue d'ensemble

```
┌─────────────────────────────────────────────────────────┐
│                    ROADSCAN-MA                          │
│                                                         │
│  Input          Detection         Output                │
│  ──────         ─────────         ──────                │
│  Image/Video    YOLOv8s           Carte Folium          │
│  (pare-brise) → CLAHE Prepro  →   Rapport PDF           │
│                 5 classes         Recommandations IA    │
│                 GPS metadata                            │
└─────────────────────────────────────────────────────────┘
```

## Stack technologique

| Composant | Technologie | Rôle |
|---|---|---|
| Détection | YOLOv8s (Ultralytics) | Inférence objet temps réel |
| Preprocessing | OpenCV CLAHE | Amélioration contraste |
| Interface | Streamlit | Dashboard web interactif |
| Cartographie | Folium + Leaflet.js | Carte GPS interactive |
| Rapport | ReportLab | Génération PDF |
| IA Générative | Groq API (Llama-3.1) | Recommandations textuelles |
| Embeddings | Base documentaire TXT | Contexte pseudo-RAG |
| Tracking | Variables de session | Déduplication frames vidéo |

---

## Modèle de détection

### Architecture YOLOv8s

- **Backbone :** CSPDarknet (small)
- **Neck :** PANet (Path Aggregation Network)
- **Head :** Détection multi-échelle (3 niveaux)
- **Input :** 640×640 px
- **Paramètres :** ~11M
- **Inférence CPU :** ~200ms/image

### Entraînement

| Paramètre | Valeur |
|---|---|
| Dataset de base | RDD2022 (47 420 images, 6 pays) |
| Dataset local | Routes Meknès (images custom) |
| Époques | 100 (RDD2022) + 25 (adaptation locale) |
| Hardware | Google Colab T4 |
| mAP50 final | **0.452** |

### Transfer Learning

```
RDD2022 (Japon, Inde, USA, Chine, Norvège, Inde)
        ↓  Fine-tuning 100 epochs
    Modèle général
        ↓  Fine-tuning 25 epochs
    Dataset routes Meknès
        ↓
    Modèle adapté Maroc
```

!!! info "Choix des données indiennes"
    Les données indiennes de RDD2022 sont les plus proches des conditions marocaines (revêtement, conditions climatiques). Elles ont été surpondérées lors de l'entraînement.

---

## Pipeline de traitement

### Mode Image

```python
Image → CLAHE Preprocessing → YOLOv8s Inference
     → Bounding Boxes + Confidence → Severity Score
     → GPS Extraction (metadata) → Folium Map
     → RAG Advisor (Groq API) → PDF Report
```

### Mode Vidéo

```python
Video → Frame Extraction (1 frame/2m) → CLAHE
     → YOLOv8s Inference par frame
     → Aggregation temporelle
     → GPS Timeline → Folium Heatmap
     → Rapport PDF agrégé
```

---

## Structure des fichiers

```
RoadScan-MA/
├── app.py                    # Dashboard Streamlit principal
├── inference.py              # Wrapper YOLOv8 + CLAHE
├── map_generator.py          # Carte Folium GPS
├── pdf_report.py             # Génération PDF ReportLab
├── rag_advisor.py            # Module RAG + Groq API
├── knowledge_base/
│   └── road_maintenance_guide.txt  # Référentiel technique DRCR
├── models/
│   └── best.pt               # Poids YOLOv8s entraîné
├── .env                      # Clé API (non versionné)
├── .env.example              # Modèle de configuration
└── requirements.txt
```
