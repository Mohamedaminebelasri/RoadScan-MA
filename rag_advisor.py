import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# ── 1. CONSTANTES ────────────────────────────────────────────────────────────

CLASS_LABELS_FR = {
    "linear_crack":   "Fissure lineaire",
    "alligator_crack":"Fissure en peau de crocodile",
    "minor_pothole":  "Nid-de-poule mineur",
    "medium_pothole": "Nid-de-poule moyen",
    "major_pothole":  "Nid-de-poule majeur"
}

SEVERITY_WEIGHTS = {
    "linear_crack":   1,
    "minor_pothole":  2,
    "alligator_crack":3,
    "medium_pothole": 4,
    "major_pothole":  5
}

# ── 2. CHARGEMENT BASE DE CONNAISSANCES ──────────────────────────────────────

@st.cache_resource
def load_knowledge_base(path="knowledge_base/road_maintenance_guide.txt"):
    """Charge le referentiel technique (Pseudo-RAG)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        st.warning(f"Base de connaissances non chargee : {e}")
        return "Contexte technique non disponible."

# ── 3. CALCUL SEVERITE ───────────────────────────────────────────────────────

def calculate_severity(detections: dict) -> dict:
    """Calcule le score global de severite a partir des detections."""
    score = sum(SEVERITY_WEIGHTS.get(cls, 0) * count
                for cls, count in detections.items())

    if score == 0:
        return {"score": 0,  "level": "Aucune",   "color": "#28a745",
                "urgency": "Pas d'intervention requise"}
    elif score <= 4:
        return {"score": score, "level": "Faible",   "color": "#ffc107",
                "urgency": "Entretien de routine (30-60 jours)"}
    elif score <= 10:
        return {"score": score, "level": "Modere",   "color": "#fd7e14",
                "urgency": "Planification a court terme"}
    elif score <= 18:
        return {"score": score, "level": "Eleve",    "color": "#dc3545",
                "urgency": "Intervention prioritaire (15 jours)"}
    else:
        return {"score": score, "level": "CRITIQUE", "color": "#8b0000",
                "urgency": "Urgence securite immediate (< 72h)"}

# ── 4. GENERATION DES RECOMMANDATIONS ────────────────────────────────────────

def generate_recommendations(detections: dict, kb_text: str, api_key: str = "") -> str:
    """
    Genere un rapport professionnel via Groq API (Llama-3.1).
    Bascule automatiquement sur le fallback si l'API est indisponible.
    """
    severity_info = calculate_severity(detections)

    # Cle API : sidebar → .env → fallback
    api_key = api_key or os.getenv("GROQ_API_KEY", "")

    if not api_key or len(api_key.strip()) < 10:
        st.warning("Mode hors-ligne : cle API absente.")
        return _fallback(detections, severity_info)

    if not kb_text or not kb_text.strip():
        kb_text = "Contexte non disponible."

    det_summary = "\n".join([
        f"- {CLASS_LABELS_FR.get(k, k)} : {v} detection(s)"
        for k, v in detections.items() if v > 0
    ])

    severity = calculate_severity(detections)

    system_prompt = (
        "Tu es un ingenieur expert en maintenance routiere au Maroc (DRCR). "
        "Genere des rapports professionnels en francais pour les services municipaux. "
        "L'indice de severite est toujours exprime sur une echelle de 0 a 100. "
        "REGLE ABSOLUE : n'inclus JAMAIS d'estimation budgetaire ni de montants en MAD."
    )

    score_100 = min(severity['score'] * 4, 100)

    user_prompt = (
        f"DEGRADATIONS DETECTEES :\n{det_summary}\n\n"
        f"SEVERITE : {severity['level']} (indice {score_100}/100)\n\n"
        f"REFERENTIEL TECHNIQUE :\n{kb_text[:1800]}\n\n"
        "Genere un rapport structure avec ces sections uniquement :\n"
        "## Diagnostic\n"
        "## Niveau d'urgence\n"
        "## Interventions recommandees\n"
        "## Calendrier d'intervention\n"
        "## Mesures preventives"
    )

    try:
        client = Groq(api_key=api_key.strip())
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=900,
            timeout=25.0
        )
        return response.choices[0].message.content

    except Exception as e:
        import traceback
        st.error(
            f"Erreur API Groq ({type(e).__name__}) : {e}\n\n"
            "Solutions : verifier la cle API sur console.groq.com "
            "ou utiliser le hotspot telephone si reseau bloque."
        )
        print(traceback.format_exc())
        return _fallback(detections, severity_info)

# ── 5. FALLBACK HEURISTIQUE (SANS API) ───────────────────────────────────────

def _fallback(detections: dict, severity_info: dict) -> str:
    """Rapport pre-programme si l'API Groq est indisponible."""

    if severity_info["score"] == 0:
        return "### Bilan\nAucune anomalie constatee. Maintenir le cycle de surveillance habituel."

    score_100 = min(severity_info['score'] * 4, 100)
    report = [
        "*(Genere hors-ligne)*\n",
        f"## Diagnostic (Indice de severite : {score_100}/100)",
        f"## Niveau d'urgence : {severity_info['urgency']}\n",
        "## Interventions recommandees"
    ]

    if detections.get("major_pothole", 0) > 0:
        report.append(
            f"- **Nid-de-poule majeur** ({detections['major_pothole']}) : "
            "Balisage d'urgence requis. Purge au marteau-piqueur et compactage d'enrobe a chaud."
        )
    if detections.get("medium_pothole", 0) > 0:
        report.append(
            f"- **Nid-de-poule moyen** ({detections['medium_pothole']}) : "
            "Nettoyage et rebouchage avec couche d'accrochage et enrobe a froid."
        )
    if detections.get("minor_pothole", 0) > 0:
        report.append(
            f"- **Nid-de-poule mineur** ({detections['minor_pothole']}) : "
            "Point-a-temps simple a l'enrobe a froid."
        )
    if detections.get("alligator_crack", 0) > 0:
        report.append(
            f"- **Fissures crocodile** ({detections['alligator_crack']}) : "
            "Fatigue structurelle. Decaissement et enduit superficiel requis."
        )
    if detections.get("linear_crack", 0) > 0:
        report.append(
            f"- **Fissures lineaires** ({detections['linear_crack']}) : "
            "Colmatage par pontage a chaud avec mastic bitumineux."
        )

    report.append("\n## Mesures preventives")
    report.append(
        "Assurer le curage des fosses pour eviter la stagnation d'eau, "
        "principal facteur de degradation sur le reseau de Meknes."
    )

    return "\n\n".join(report)