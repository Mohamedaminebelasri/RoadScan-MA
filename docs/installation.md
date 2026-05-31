# Installation

## Prérequis

- Python **3.10+**
- Windows 10/11 (testé), Linux, macOS
- **Pas de GPU requis** — fonctionne en CPU only
- Compte Groq gratuit : [console.groq.com](https://console.groq.com)

## Étapes d'installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/Mohamedaminebelasri/RoadScan-MA.git
cd RoadScan-MA
```

### 2. Créer l'environnement virtuel

=== "Windows"

    ```powershell
    python -m venv roadscan-env
    roadscan-env\Scripts\activate
    ```

=== "Linux / macOS"

    ```bash
    python -m venv roadscan-env
    source roadscan-env/bin/activate
    ```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

Dépendances principales :

| Package | Rôle |
|---|---|
| `ultralytics` | Détection YOLOv8 |
| `streamlit` | Dashboard web |
| `folium` | Cartographie interactive |
| `opencv-python` | Traitement vidéo + CLAHE |
| `reportlab` | Génération PDF |
| `groq` | API LLM Llama-3.1 |
| `python-dotenv` | Gestion variables d'environnement |

### 4. Configurer la clé API Groq

Créer un fichier `.env` à la racine du projet :

```bash
GROQ_API_KEY=gsk_votre_cle_ici
```

!!! warning "Sécurité"
    Ne commitez jamais votre `.env` sur GitHub. Le fichier `.gitignore` l'exclut automatiquement.
    Utilisez `.env.example` comme modèle.

### 5. Lancer l'application

```bash
streamlit run app.py
```

L'application s'ouvre automatiquement sur [http://localhost:8501](http://localhost:8501)

## Vérification de l'installation

```bash
python verification_complete.py
```

Résultat attendu :

```
✅ .env trouvé
✅ Clé trouvée : gsk_...
✅ Trouvé — 2648 caractères
✅ Import OK
✅ Sévérité calculée : Élevé (score 11)
✅ API répond : GROQ_OK
🎉 TOUT EST OK
```
