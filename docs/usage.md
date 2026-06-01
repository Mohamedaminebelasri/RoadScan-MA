# Guide d'utilisation

## Lancer le dashboard

```bash
streamlit run app.py
```

Ouvre [http://localhost:8501](http://localhost:8501) dans le navigateur.

---

## Mode IMAGE

Idéal pour analyser une photo isolée de route.

1. Dans la **sidebar**, sélectionner **"📷 Image"**
2. Ajuster le **seuil de confiance** (défaut : 0.25)
3. Uploader une image JPG/PNG via le bouton
4. Le système affiche :
    - Image originale vs image annotée avec bounding boxes
    - Tableau des détections par classe
    - Score de sévérité global (🟢 Faible → 🚨 Critique)
    - Carte Folium avec marqueurs GPS
5. Cliquer **"🧠 Générer les recommandations IA"** pour le rapport Llama-3.1
6. Télécharger le **rapport PDF** complet

---

## Mode VIDÉO

Idéal pour analyser un parcours filmé depuis le pare-brise.

1. Sélectionner **"🎥 Vidéo"**
2. Uploader un fichier MP4/AVI
3. Le système traite chaque frame avec une barre de progression
4. Résultats agrégés sur l'ensemble du parcours :
    - Total détections par classe
    - Carte avec tous les points GPS détectés
    - Rapport PDF du parcours complet

!!! tip "Conseil terrain"
    Pour de meilleurs résultats, filmer à 30-40 km/h sur route sèche. Fixer le smartphone au centre du pare-brise avec un support adapté.

---

## Mode DÉMO

Test rapide sans upload.

1. Sélectionner **"🎬 Démo"**
2. Le système utilise des images pré-chargées (routes de Meknès)
3. Résultats immédiats sans configuration

---

## Recommandations IA (RAG)

Après toute détection, la section **"🤖 Recommandations IA"** est disponible.

Le système :

1. Calcule l'**indice de sévérité** pondéré par classe
2. Charge le **référentiel technique** (normes DRCR Maroc)
3. Envoie les données au **LLM Llama-3.1** via Groq API
4. Génère un rapport structuré incluant :

```
## Diagnostic
## Niveau d'urgence
## Interventions recommandées
## Calendrier d'intervention
## Mesures préventives
```

!!! note "Mode hors-ligne"
    Sans clé API Groq, le système bascule automatiquement sur un rapport heuristique pré-programmé.

---

## Indice de sévérité

| Score | Niveau | Urgence |
|---|---|---|
| 0 | Aucune 🟢 | Pas d'intervention |
| 1-4 | Faible 🟡 | Entretien de routine (30-60 jours) |
| 5-10 | Modéré 🟠 | Planification à court terme |
| 11-18 | Élevé 🔴 | Intervention prioritaire (15 jours) |
| > 18 | CRITIQUE 🚨 | Urgence sécurité (< 72h) |

Poids par classe : `linear_crack=1`, `minor_pothole=2`, `alligator_crack=3`, `medium_pothole=4`, `major_pothole=5`

## Upload fichier GPX (Mode Vidéo)

Un fichier GPX optionnel peut être fourni avec la vidéo pour une géolocalisation réelle :

1. Upload la vidéo MP4
2. Upload le fichier `.gpx` (optionnel)
3. Si GPX fourni → carte avec vraies coordonnées GPS
4. Si pas de GPX → message informatif, détections sans carte

## Validation des fichiers

Le système valide automatiquement chaque fichier uploadé :

| Type | Formats acceptés | Taille max |
|---|---|---|
| Image | JPG, PNG, BMP, WEBP | 10 MB |
| Vidéo | MP4, AVI, MOV (H.264) | 500 MB |
| GPX | .gpx | Illimité |

En cas de fichier invalide, un message d'erreur clair indique la cause.
