# RoadScan-MA

**Détection automatique des dégradations routières avec cartographie GPS et recommandations IA**

---

!!! info "Projet académique"
    Projet de fin d'études — Filière IATD (Intelligence Artificielle et Technologies des Données)
    **ENSAM Meknès — 2026**

## Présentation

RoadScan-MA est un système de vision par ordinateur qui détecte automatiquement les dégradations routières depuis une vidéo ou une image prise avec un smartphone fixé au pare-brise. Le système génère ensuite une carte interactive géolocalisée et des recommandations professionnelles à destination des services techniques municipaux.

## Fonctionnalités principales

| Fonctionnalité | Description |
|---|---|
| 🔍 **Détection YOLO** | 5 classes de dégradations en temps réel (mAP50 = 0.452) |
| 🗺️ **Cartographie GPS** | Carte Folium interactive avec marqueurs colorés par sévérité |
| 🤖 **Recommandations IA** | Rapport professionnel généré par Llama-3.1 via Groq API |
| 📄 **Export PDF** | Rapport complet téléchargeable pour la municipalité |
| 📊 **Dashboard Streamlit** | Interface web intuitive, modes Image / Vidéo / Démo |

## Les 5 classes détectées

```
linear_crack      → Fissure linéaire
alligator_crack   → Fissure en peau de crocodile
minor_pothole     → Nid-de-poule mineur  (< 5 cm)
medium_pothole    → Nid-de-poule moyen   (5-10 cm)
major_pothole     → Grand nid-de-poule   (> 10 cm)
```

## Démarrage rapide

```bash
git clone https://github.com/Mohamedaminebelasri/RoadScan-MA.git
cd RoadScan-MA
pip install -r requirements.txt
echo "GROQ_API_KEY=gsk_votre_cle" > .env
streamlit run app.py
```

## Liens

- **GitHub :** [github.com/Mohamedaminebelasri/RoadScan-MA](https://github.com/Mohamedaminebelasri/RoadScan-MA)
- **Dataset Kaggle :** [roadscan-dataset](https://www.kaggle.com/datasets/mohamedaminebelasri7/roadscan-dataset)
- **Clé Groq gratuite :** [console.groq.com](https://console.groq.com)

## Contexte

Le réseau routier marocain dépasse **57 000 km** et subit une dégradation accélérée par les conditions climatiques extrêmes. L'inspection manuelle est coûteuse et lente. RoadScan-MA propose une solution automatisée, légère (CPU only) et adaptée au contexte marocain.
