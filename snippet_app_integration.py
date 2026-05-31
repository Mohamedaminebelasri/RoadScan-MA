# =====================================================================
# AJOUT 1 : Tout en haut du fichier (dans les imports)
# =====================================================================
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from rag_advisor import load_knowledge_base, generate_recommendations, calculate_severity

# =====================================================================
# AJOUT 2 : Juste après le @st.cache_resource qui charge YOLO
# =====================================================================
# (Cherchez la fonction def load_model(...) dans votre app.py)
# Insérez ceci en dessous :

kb_text = load_knowledge_base(os.path.join(os.path.dirname(__file__), "knowledge_base", "road_maintenance_guide.txt"))

if "recommendations" not in st.session_state:
    st.session_state["recommendations"] = None
if "last_analyzed_data" not in st.session_state:
    st.session_state["last_analyzed_data"] = None

# =====================================================================
# AJOUT 3 : Dans la fonction qui rend la Sidebar (ex: def render_sidebar():)
# =====================================================================
# À la fin de la sidebar, avant le return, ajoutez :
st.divider()
st.markdown("### ⚙️ Configuration IA (RAG)")
groq_api_key = st.text_input("Clé API Groq (Optionnelle)", type="password", value=os.environ.get("GROQ_API_KEY", ""))
st.caption("Obtenez une clé gratuite sur [console.groq.com](https://console.groq.com/)")
# Assurez-vous que votre fonction retourne la clé si besoin, ou stockez la dans st.session_state
st.session_state["groq_api_key"] = groq_api_key

# =====================================================================
# AJOUT 4 : Dans l'affichage des résultats (Mode IMAGE et VIDÉO)
# =====================================================================
# Cherchez l'endroit où vous affichez les boutons de téléchargement CSV/PDF.
# Juste avant (ou juste après), insérez ce bloc :

# --- DEBUT SNIPPET RECOMMANDATIONS ---
st.divider()
st.markdown("### 🤖 Recommandations IA (Municipalité de Meknès)")

# Transformation des détections (si elles sont sous forme de liste de dictionnaires YOLO)
# Exemple: detections_list = [{"class_name": "linear_crack"}, {"class_name": "major_pothole"}]
# Adaptation selon votre format exact :
counts = {}
for det in detections: # 'detections' est la variable issue de votre prédiction YOLO
    cls_name = det.get("class_name")
    counts[cls_name] = counts.get(cls_name, 0) + 1

# Calcul Sévérité
severity_info = calculate_severity(counts)
st.markdown(f"**Indice de sévérité détecté :** <span style='color:{severity_info['color']}; font-size: 18px; font-weight: bold;'>{severity_info['score']} ({severity_info['level']})</span>", unsafe_allow_html=True)
st.caption(severity_info['urgency'])

# Bouton de génération
if st.button("🧠 Générer les recommandations IA"):
    # On vérifie si on a analysé une nouvelle image/vidéo pour ne pas régénérer inutilement
    with st.spinner("Génération du rapport d'expertise en cours via IA..."):
        api_key = st.session_state.get("groq_api_key", "")
        # Appel RAG
        reco = generate_recommendations(counts, kb_text, api_key)
        st.session_state["recommendations"] = reco
        st.session_state["last_analyzed_data"] = counts

# Affichage des recommandations stockées
if st.session_state["recommendations"]:
    st.info("Rapport d'intervention généré avec succès.")
    st.markdown(st.session_state["recommendations"])
    
    # Bouton d'export TXT
    st.download_button(
        label="📥 Télécharger le rapport (TXT)",
        data=st.session_state["recommendations"],
        file_name="rapport_ia_meknes.txt",
        mime="text/plain"
    )
# --- FIN SNIPPET RECOMMANDATIONS ---