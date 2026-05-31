# 🛣️ RoadScan-MA

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)
![Groq](https://img.shields.io/badge/LLM-Llama--3.1-purple?logo=meta)
![License](https://img.shields.io/badge/License-MIT-green)
[![Documentation](https://readthedocs.org/projects/roadscan-ma/badge/?version=latest)](https://roadscan-ma.readthedocs.io)
[![Kaggle Dataset](https://img.shields.io/badge/Dataset-Kaggle-20beff?logo=kaggle)](https://www.kaggle.com/datasets/mohamedaminebelasri7/roadscan-dataset)

**Détection automatique des dégradations routières avec cartographie GPS et recommandations IA**

*Projet académique IATD — Mohamed Amine Belasri & Yahya Amajane — ENSAM Meknès 2026*

</div>

---

## 📌 Présentation

RoadScan-MA est un système complet de vision par ordinateur qui détecte automatiquement les dégradations routières depuis une vidéo ou une image prise avec un smartphone fixé au pare-brise. Le système génère ensuite :

- Une **carte interactive géolocalisée** des dégradations
- Un **rapport PDF professionnel** téléchargeable
- Des **recommandations IA** générées par un LLM (Llama-3.1) pour les services techniques municipaux

> **Contexte :** Le réseau routier marocain dépasse **57 000 km** et subit une dégradation accélérée par les conditions climatiques extrêmes. L'inspection est manuelle, coûteuse et lente. RoadScan-MA automatise ce processus avec un simple smartphone.

---

## ✨ Fonctionnalités

| Module | Description |
|---|---|
| 🔍 **Détection YOLO** | YOLOv8s fine-tuné — 5 classes, mAP50 = **0.452** |
| 🗺️ **Carte Folium GPS** | Marqueurs colorés par type et sévérité sur carte OpenStreetMap |
| 🤖 **RAG + Recommandations IA** | Llama-3.1 via Groq API — rapport structuré pour la municipalité |
| 📄 **Export PDF** | Rapport complet avec ReportLab — prêt à soumettre |
| 📊 **Dashboard Streamlit** | Interface web — modes Image / Vidéo / Démo |
| 📈 **Indice de sévérité** | Score pondéré par classe — 5 niveaux d'urgence |

---

## 🎯 Classes détectées

```
linear_crack      →  Fissure linéaire              (poids sévérité : 1)
alligator_crack   →  Fissure en peau de crocodile  (poids sévérité : 3)
minor_pothole     →  Nid-de-poule mineur  < 5 cm   (poids sévérité : 2)
medium_pothole    →  Nid-de-poule moyen   5-10 cm  (poids sévérité : 4)
major_pothole     →  Grand nid-de-poule   > 10 cm  (poids sévérité : 5)
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       ROADSCAN-MA                           │
│                                                             │
│  📱 Input            🧠 Traitement          📤 Output       │
│  ──────────          ────────────           ──────────      │
│  Image / Vidéo  →   CLAHE Preprocess   →   Carte Folium    │
│  (pare-brise)   →   YOLOv8s Inference  →   Rapport PDF     │
│                 →   GPS Extraction     →   Reco. IA (LLM)  │
│                 →   Severity Scoring   →   Export CSV       │
└─────────────────────────────────────────────────────────────┘
```

### Stack technique

| Composant | Technologie |
|---|---|
| Détection | YOLOv8s (Ultralytics) |
| Preprocessing | OpenCV CLAHE |
| Interface | Streamlit |
| Cartographie | Folium + Leaflet.js |
| Rapport PDF | ReportLab |
| LLM / RAG | Groq API — llama-3.1-8b-instant |
| Base documentaire | Référentiel DRCR Maroc (TXT) |
| Environnement | CPU only — Windows / Linux |

---

## 📦 Dataset

- **RDD2022** : 47 420 images, 6 pays (base d'entraînement principale)
- **Dataset local Meknès** : Images custom collectées sur les routes nationales et urbaines de Meknès (pare-brise)
- **Dataset Kaggle public** : [roadscan-dataset](https://www.kaggle.com/datasets/mohamedaminebelasri7/roadscan-dataset)

> Les données indiennes de RDD2022 ont été surpondérées car les conditions de revêtement sont les plus proches du contexte marocain.

---

## ⚙️ Installation

### Prérequis

- Python 3.10+
- Pas de GPU requis (CPU only)
- Clé API Groq gratuite : [console.groq.com](https://console.groq.com)

### Étapes

```bash
# 1. Cloner le dépôt
git clone https://github.com/Mohamedaminebelasri/RoadScan-MA.git
cd RoadScan-MA

# 2. Créer l'environnement virtuel
python -m venv roadscan-env
source roadscan-env/bin/activate      # Linux/macOS
# roadscan-env\Scripts\activate       # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer la clé API
echo "GROQ_API_KEY=gsk_votre_cle_ici" > .env

# 5. Lancer l'application
streamlit run app.py
```

---

## 🚀 Utilisation

### Mode IMAGE
Upload d'une photo de route → détection instantanée → carte + rapport PDF + recommandations IA

### Mode VIDÉO
Upload d'une vidéo pare-brise → traitement frame par frame → carte du parcours complet + rapport agrégé

### Mode DÉMO
Test immédiat avec des images pré-chargées (routes de Meknès) — sans upload

### Recommandations IA (RAG)

Après chaque détection, cliquer **"🧠 Générer les recommandations IA"** :

```
Le système :
1. Calcule l'indice de sévérité (score pondéré)
2. Charge le référentiel technique DRCR Maroc
3. Envoie tout au LLM Llama-3.1 via Groq API
4. Génère un rapport structuré en ~2 secondes
```

### Indice de sévérité

| Score | Niveau | Urgence |
|---|---|---|
| 0 | 🟢 Aucune | Pas d'intervention |
| 1-4 | 🟡 Faible | Routine (30-60 jours) |
| 5-10 | 🟠 Modéré | Court terme |
| 11-18 | 🔴 Élevé | Prioritaire (15 jours) |
| >18 | 🚨 CRITIQUE | Urgence (< 72h) |

---

## 📁 Structure du projet

```
RoadScan-MA/
├── app.py                              # Dashboard Streamlit principal
├── inference.py                        # Wrapper YOLOv8 + CLAHE
├── map_generator.py                    # Carte Folium GPS interactive
├── pdf_report.py                       # Génération rapport PDF
├── rag_advisor.py                      # Module RAG + Groq API (Llama-3.1)
├── knowledge_base/
│   └── road_maintenance_guide.txt      # Référentiel technique DRCR Maroc
├── models/
│   └── best.pt                         # Poids YOLOv8s entraînés
├── docs/                               # Documentation MkDocs
├── .env.example                        # Modèle de configuration
├── .readthedocs.yaml                   # Config ReadTheDocs
├── mkdocs.yml                          # Config documentation
└── requirements.txt                    # Dépendances Python
```

---

## 📊 Performances du modèle

| Métrique | Valeur |
|---|---|
| mAP50 | **0.452** |
| Architecture | YOLOv8s |
| Dataset base | RDD2022 (47 420 images) |
| Fine-tuning | Routes Meknès (custom) |
| Inférence CPU | ~200ms/image |
| Hardware entraînement | Google Colab T4 |

---

## 📚 Documentation

Documentation complète disponible sur ReadTheDocs :

👉 **[roadscan-ma.readthedocs.io](https://roadscan-ma.readthedocs.io)**

---

## 🔗 Liens utiles

- **GitHub :** [Mohamedaminebelasri/RoadScan-MA](https://github.com/Mohamedaminebelasri/RoadScan-MA)
- **Dataset Kaggle :** [roadscan-dataset](https://www.kaggle.com/datasets/mohamedaminebelasri7/roadscan-dataset)
- **Documentation :** [roadscan-ma.readthedocs.io](https://roadscan-ma.readthedocs.io)
- **Clé Groq gratuite :** [console.groq.com](https://console.groq.com)

---

## 👥 Auteurs

**Mohamed Amine Belasri** & **Yahya Amajane**
Étudiants ingénieurs — Filière IATD (Intelligence Artificielle et Technologies des Données)
ENSAM Meknès — Université Moulay Ismail — 2026

---

## 📜 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.
