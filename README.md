# 🛣️ RoadScan-MA

> Détection automatique des dégradations routières avec cartographie GPS pour les municipalités marocaines

## Lancement rapide

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Placer le modèle YOLO
cp best.pt models/

# 3. Lancer l'application
streamlit run app.py
```

L'app sera disponible sur **http://localhost:8501**

## Classes détectées

| ID | Classe | Couleur |
|----|--------|---------|
| 0 | linear_crack | 🔵 Bleu |
| 1 | alligator_crack | 🟫 Marron |
| 2 | minor_pothole | 🟢 Vert |
| 3 | medium_pothole | 🟠 Orange |
| 4 | major_pothole | 🔴 Rouge |

## Structure

```
RoadScan-MA/
├── app.py              ← Dashboard Streamlit
├── inference.py        ← Wrapper YOLOv8
├── map_generator.py    ← Cartes Folium
├── utils.py            ← Fonctions partagées
├── models/best.pt      ← Modèle YOLO (à placer manuellement)
├── assets/             ← Logo et ressources visuelles
├── sample_data/        ← Images/vidéos de test pour démo
└── .streamlit/         ← Configuration thème sombre
```
