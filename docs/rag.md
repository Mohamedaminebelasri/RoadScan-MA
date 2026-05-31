# Module RAG / Recommandations IA

## Principe

Le module RAG (Retrieval-Augmented Generation) de RoadScan-MA génère des recommandations professionnelles pour les services techniques municipaux en combinant :

1. Les **détections YOLO** (types et quantités de dégradations)
2. Un **référentiel technique** sur la maintenance routière marocaine
3. Le **LLM Llama-3.1** via l'API Groq (gratuit, ultra-rapide)

---

## Architecture pseudo-RAG

```
knowledge_base/road_maintenance_guide.txt
              ↓  chargé au démarrage (@st.cache_resource)
         Texte de référence (2600 caractères)
              ↓
         Injection dans le contexte LLM
              ↓
         Groq API (llama-3.1-8b-instant)
              ↓
         Rapport structuré Markdown
```

!!! note "Pseudo-RAG vs RAG vectoriel"
    Pour ce projet, la base documentaire est suffisamment compacte (< 3000 caractères) pour être injectée entièrement dans le contexte LLM sans recherche vectorielle. Cette approche est plus simple, plus rapide et aussi efficace pour un domaine aussi ciblé que la maintenance routière.

---

## Calcul de l'indice de sévérité

Chaque classe de dégradation a un **poids de sévérité** :

```python
SEVERITY_WEIGHTS = {
    "linear_crack":    1,   # Fissure linéaire
    "minor_pothole":   2,   # Nid-de-poule mineur
    "alligator_crack": 3,   # Fissure en crocodile
    "medium_pothole":  4,   # Nid-de-poule moyen
    "major_pothole":   5    # Grand nid-de-poule (critique)
}
```

**Score = Σ (poids × nombre de détections)**

| Score | Niveau | Couleur | Action |
|---|---|---|---|
| 0 | Aucune | 🟢 Vert | Maintenance préventive |
| 1-4 | Faible | 🟡 Jaune | Routine (30-60 jours) |
| 5-10 | Modéré | 🟠 Orange | Court terme |
| 11-18 | Élevé | 🔴 Rouge | Prioritaire (15 jours) |
| > 18 | CRITIQUE | 🚨 Bordeaux | Urgence (< 72h) |

---

## Rapport généré

Le LLM produit un rapport structuré avec ces sections :

```markdown
## Diagnostic
[Analyse des dégradations détectées]

## Niveau d'urgence
[Justification + délai d'intervention]

## Interventions recommandées
[Méthodes techniques par type de dégradation]

## Calendrier d'intervention
[Délais priorisés]

## Mesures préventives
[Actions long terme]
```

---

## Configuration

### Clé API Groq (gratuite)

```bash
# .env
GROQ_API_KEY=gsk_votre_cle_ici
```

Obtenir une clé gratuite : [console.groq.com](https://console.groq.com)

### Modèle utilisé

```python
model = "llama-3.1-8b-instant"
# Ultra-rapide (~300 tokens/sec), 128K contexte, gratuit
```

---

## Mode hors-ligne (fallback)

Sans clé API ou en cas d'erreur réseau, le système bascule automatiquement sur un rapport heuristique pré-programmé basé sur des règles `if/elif` par classe de dégradation. La démonstration reste fonctionnelle même sans connexion internet.

---

## Fichier source : `rag_advisor.py`

Fonctions principales :

| Fonction | Description |
|---|---|
| `load_knowledge_base(path)` | Charge le référentiel technique (cached) |
| `calculate_severity(detections)` | Calcule le score et niveau d'urgence |
| `generate_recommendations(detections, kb_text, api_key)` | Appelle Groq et retourne le rapport |
| `_fallback(detections, severity_info)` | Rapport heuristique sans API |
