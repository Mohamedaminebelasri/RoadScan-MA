"""
RoadScan-MA — pdf_report.py
Rapport PDF professionnel pour municipalités marocaines.
Contenu :
  - Page de couverture colorée
  - Résumé exécutif
  - Bar chart des détections (matplotlib)
  - Tableau détaillé par classe
  - Photo annotée (optionnelle)
  - Recommandations (IA ou Classique)
  - Section validation officielle
"""

import io
import re  # <--- AJOUT POUR LIRE LE MARKDOWN DE L'IA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame,
    Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image, PageBreak, KeepTogether
)
from reportlab.platypus.flowables import Flowable


# ── Palette couleurs ───────────────────────────────────────
C_PRIMARY  = colors.HexColor("#E74C3C")
C_DARK     = colors.HexColor("#1A1D23")
C_DARKGRAY = colors.HexColor("#2C3E50")
C_BLUE     = colors.HexColor("#3498DB")
C_BROWN    = colors.HexColor("#8B4513")
C_GREEN    = colors.HexColor("#2ECC71")
C_ORANGE   = colors.HexColor("#F39C12")
C_RED      = colors.HexColor("#E74C3C")
C_LIGHT    = colors.HexColor("#F8F9FA")
C_GRAY     = colors.HexColor("#95A5A6")
C_WHITE    = colors.white

CLASS_COLORS_HEX = {
    "linear_crack":    "#3498DB",
    "alligator_crack": "#8B4513",
    "minor_pothole":   "#2ECC71",
    "medium_pothole":  "#F39C12",
    "major_pothole":   "#E74C3C",
}
CLASS_LABELS_FR = {
    "linear_crack":    "Fissure linéaire",
    "alligator_crack": "Fissure alligator",
    "minor_pothole":   "Nid-de-poule mineur",
    "medium_pothole":  "Nid-de-poule moyen",
    "major_pothole":   "Nid-de-poule majeur",
}
W, H = A4


# ── Bande colorée ──────────────────────────────────────────
class ColorBand(Flowable):
    """Rectangle coloré pleine largeur."""
    def __init__(self, height=0.8*cm, color=C_PRIMARY):
        super().__init__()
        self.band_height = height
        self.band_color  = color
        self.width       = 17*cm

    def draw(self):
        self.canv.setFillColor(self.band_color)
        self.canv.rect(0, 0, self.width, self.band_height, fill=1, stroke=0)


# ── Header / Footer sur chaque page ───────────────────────
def _on_page(canvas, doc):
    """Header et footer sur toutes les pages sauf la couverture."""
    if doc.page == 1:
        return
    canvas.saveState()

    # Header — bande rouge + titre
    canvas.setFillColor(C_PRIMARY)
    canvas.rect(0, H - 1.2*cm, W, 1.2*cm, fill=1, stroke=0)
    canvas.setFillColor(C_WHITE)
    canvas.setFont("Helvetica-Bold", 11)
    canvas.drawString(2*cm, H - 0.85*cm, "  RoadScan-MA — Rapport d'Inspection Routière")
    canvas.setFont("Helvetica", 8)
    date_str = datetime.now().strftime("%d/%m/%Y")
    canvas.drawRightString(W - 2*cm, H - 0.85*cm, date_str)

    # Footer
    canvas.setFillColor(C_DARKGRAY)
    canvas.rect(0, 0, W, 0.8*cm, fill=1, stroke=0)
    canvas.setFillColor(C_WHITE)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(2*cm, 0.25*cm, "ENSAM Meknès — IATD | Belasri M.A. & Amajane Y. | RoadScan-MA v1.0")
    canvas.drawRightString(W - 2*cm, 0.25*cm, f"Page {doc.page}")

    canvas.restoreState()


# ── Page de couverture ────────────────────────────────────
def _make_cover(styles, summary, zone):
    """Génère les éléments de la page de couverture."""
    elements = []
    level  = summary.get("severity_level", {})
    index  = summary.get("severity_index", 0)
    emoji  = level.get("emoji", "")
    label  = level.get("label", "")
    color  = level.get("color", "#E74C3C")
    total  = summary.get("total", 0)
    date_s = datetime.now().strftime("%d %B %Y")

    elements.append(Spacer(1, 4.5*cm))
    elements.append(ColorBand(1.2*cm, C_PRIMARY))
    elements.append(Spacer(1, 0.8*cm))

    cover_title = ParagraphStyle("CoverTitle", parent=styles["Normal"],
        fontSize=32, textColor=C_DARK, alignment=TA_CENTER,
        fontName="Helvetica-Bold", spaceAfter=4)
    elements.append(Paragraph(" RoadScan-MA", cover_title))

    sub_style = ParagraphStyle("CoverSub", parent=styles["Normal"],
        fontSize=15, textColor=C_GRAY, alignment=TA_CENTER,
        fontName="Helvetica", spaceAfter=12)
    elements.append(Paragraph("Rapport d'Inspection Automatique des Routes", sub_style))

    elements.append(Spacer(1, 0.3*cm))
    elements.append(ColorBand(1.2*cm, C_DARK))
    elements.append(Spacer(1, 1.5*cm))

    metric_data = [[
        Paragraph(f"<b><font size='24' color='#E74C3C'>{total}</font></b><br/>"
                  f"<font size='10' color='#666'>Dégradations<br/>détectées</font>",
                  ParagraphStyle("M", alignment=TA_CENTER, parent=styles["Normal"])),
        Paragraph(f"<b><font size='24' color='{color}'>{index}/100</font></b><br/>"
                  f"<font size='10' color='#666'>Indice de<br/>sévérité</font>",
                  ParagraphStyle("M", alignment=TA_CENTER, parent=styles["Normal"])),
        Paragraph(f"<b><font size='24' color='#2ECC71'>{summary.get('potholes',0)}</font></b><br/>"
                  f"<font size='10' color='#666'>Nids-de-<br/>poule</font>",
                  ParagraphStyle("M", alignment=TA_CENTER, parent=styles["Normal"])),
    ]]
    metric_table = Table(metric_data, colWidths=[5.5*cm, 5.5*cm, 5.5*cm])
    metric_table.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,-1), C_LIGHT),
        ("BOX",         (0,0), (-1,-1), 1.5, C_PRIMARY),
        ("LINEAFTER",   (0,0), (1,0),   1, C_GRAY),
        ("PADDING",     (0,0), (-1,-1), 14),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
    ]))
    elements.append(metric_table)
    elements.append(Spacer(1, 1.5*cm))

    info_style = ParagraphStyle("Info", parent=styles["Normal"],
        fontSize=11, alignment=TA_CENTER, textColor=C_DARKGRAY, spaceAfter=6)
    elements.append(Paragraph(f"<b>Zone inspectée :</b> {zone}", info_style))
    elements.append(Paragraph(f"<b>Date du rapport :</b> {date_s}", info_style))
    elements.append(Paragraph(f"<b>Niveau de sévérité :</b> {emoji} {label}", info_style))
    elements.append(Spacer(1, 2*cm))

    elements.append(ColorBand(0.4*cm, C_PRIMARY))
    elements.append(Spacer(1, 0.3*cm))
    footer_style = ParagraphStyle("CoverFooter", parent=styles["Normal"],
        fontSize=9, alignment=TA_CENTER, textColor=C_GRAY)
    elements.append(Paragraph("ENSAM Meknès — IATD | Généré automatiquement par RoadScan-MA", footer_style))
    elements.append(PageBreak())
    return elements


# ── Bar chart matplotlib ───────────────────────────────────
def _make_bar_chart(summary) -> Image:
    counts  = summary.get("counts", {})
    labels  = [CLASS_LABELS_FR[k] for k in counts]
    values  = list(counts.values())
    colors_ = [CLASS_COLORS_HEX[k] for k in counts]

    fig, ax = plt.subplots(figsize=(8, 3.2))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    bars = ax.bar(labels, values, color=colors_, width=0.55, edgecolor="white", linewidth=1.2)

    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                    str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Nombre de détections", fontsize=9)
    ax.set_title("Répartition par type de dégradation", fontsize=11, fontweight="bold", pad=10)
    ax.set_ylim(0, max(values or [1]) * 1.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return Image(buf, width=16*cm, height=6.5*cm)


# ── Tableau détaillé ───────────────────────────────────────
def _make_class_table(summary):
    counts = summary.get("counts", {})
    total  = summary.get("total", 1) or 1

    data = [["Type de dégradation", "Nb", "%", "Priorité"]]
    priorities = {
        "linear_crack": "Surveillance",
        "alligator_crack": "Moyen terme",
        "minor_pothole": "Planifié",
        "medium_pothole": "Urgent",
        "major_pothole": "🚨 Immédiat",
    }
    for cls, label in CLASS_LABELS_FR.items():
        count   = counts.get(cls, 0)
        percent = f"{count/total*100:.1f}%"
        data.append([label, str(count), percent, priorities[cls]])

    t = Table(data, colWidths=[7.5*cm, 2*cm, 2.5*cm, 5*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",     (0,0), (-1,0),  C_DARK),
        ("TEXTCOLOR",      (0,0), (-1,0),  C_WHITE),
        ("FONTNAME",       (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",       (0,0), (-1,-1), 9),
        ("ALIGN",          (1,0), (-1,-1), "CENTER"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [C_WHITE, C_LIGHT]),
        ("GRID",           (0,0), (-1,-1), 0.4, C_GRAY),
        ("PADDING",        (0,0), (-1,-1), 7),
        ("TOPPADDING",     (0,0), (-1,0),  9),
    ]))
    return t


# ── Section photo ──────────────────────────────────────────
def _make_photo_section(styles, image_path):
    elements = []
    elements.append(_section_title(styles, "3. Exemple de Détection"))
    try:
        img = Image(image_path, width=14*cm, height=8*cm)
        img.hAlign = "CENTER"
        elements.append(img)
        cap = ParagraphStyle("Cap", parent=styles["Normal"],
            fontSize=8, textColor=C_GRAY, alignment=TA_CENTER, spaceAfter=6)
        elements.append(Paragraph("Image annotée par RoadScan-MA — bounding boxes colorées par type", cap))
    except Exception:
        pass
    return elements


# ── NOUVEAU : Parseur RAG (Convertit le Markdown IA en PDF) ──
def _parse_markdown_to_flowables(text, styles):
    """Traduit le texte brut de l'IA en paragraphes stylisés ReportLab."""
    elements = []
    
    # Styles personnalisés pour le RAG
    rag_normal = ParagraphStyle("RAG_Normal", parent=styles["Normal"], fontSize=9, spaceAfter=6, leading=13)
    rag_h2 = ParagraphStyle("RAG_H2", parent=styles["Normal"], fontSize=11, textColor=C_PRIMARY, fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=4)

    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # 1. Gestion des titres (##)
        if line.startswith('## ') or line.startswith('# '):
            clean_text = line.replace('## ', '').replace('# ', '').replace('**', '')
            elements.append(Paragraph(clean_text, rag_h2))
            
        # 2. Gestion des listes et du gras
        else:
            # Remplacer **texte** par <b>texte</b>
            line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
            
            # Puces
            if line.startswith('- ') or line.startswith('* '):
                line = "&nbsp;&nbsp;• " + line[2:]
                
            elements.append(Paragraph(line, rag_normal))
            
    return elements


# ── Recommandations (Classique de secours) ─────────────────
def _make_recommendations(styles, summary):
    index  = summary.get("severity_index", 0)
    counts = summary.get("counts", {})
    major  = counts.get("major_pothole", 0)
    medium = counts.get("medium_pothole", 0)

    if index == 0:
        actions = [("✅", "Aucune intervention requise.", C_GREEN)]
    elif index < 30:
        actions = [
            ("🔵", "Surveiller l'évolution des fissures.", C_BLUE),
            ("🟢", "Traitement préventif dans les 6 mois.", C_GREEN),
        ]
    elif index < 60:
        actions = [
            ("🟡", f"Colmater les {medium} nids-de-poule moyens sous 2-3 mois.", C_ORANGE),
            ("🔧", "Établir un plan de réparation par section.", C_BLUE),
            ("📋", "Inspecter à nouveau dans 30 jours.", C_GRAY),
        ]
    elif index < 80:
        actions = [
            ("🟠", f"Traiter {major} nids-de-poule majeurs en urgence (< 30 jours).", C_ORANGE),
            ("⚠️", "Signalisation temporaire sur les zones critiques.", C_RED),
            ("📞", "Contacter le service technique municipal.", C_DARK),
        ]
    else:
        actions = [
            ("🔴", f"CRITIQUE — {major} nids-de-poule majeurs détectés.", C_RED),
            ("🚧", "Fermeture partielle de la voie recommandée.", C_RED),
            ("📞", "Intervention d'urgence municipale requise.", C_DARK),
            ("📸", "Documenter et photographier chaque dégradation.", C_GRAY),
        ]

    rows = []
    for icon, text, color in actions:
        style = ParagraphStyle("Rec", parent=styles["Normal"], fontSize=9,
                               textColor=colors.HexColor(color.hexval() if hasattr(color, "hexval") else "#333333"))
        rows.append([Paragraph(f"{icon}  {text}", style)])

    t = Table(rows, colWidths=[17*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), C_LIGHT),
        ("PADDING",       (0,0), (-1,-1), 8),
        ("LINEBELOW",     (0,0), (-1,-2), 0.3, C_GRAY),
    ]))
    return t


# ── Section validation ─────────────────────────────────────
def _make_validation(styles):
    data = [
        ["Validé par :", "________________________________"],
        ["Fonction :",   "________________________________"],
        ["Date :",       "________________________________"],
        ["Signature :",  "________________________________"],
    ]
    t = Table(data, colWidths=[5*cm, 12*cm])
    t.setStyle(TableStyle([
        ("FONTSIZE",  (0,0), (-1,-1), 9),
        ("PADDING",   (0,0), (-1,-1), 8),
        ("FONTNAME",  (0,0), (0,-1),  "Helvetica-Bold"),
        ("LINEBELOW", (1,0), (1,-1),  0.5, C_GRAY),
        ("BACKGROUND",(0,0), (-1,-1), C_LIGHT),
    ]))
    return t


# ── Titre de section ───────────────────────────────────────
def _section_title(styles, title):
    return Paragraph(title, ParagraphStyle(
        "SecTitle", parent=styles["Heading2"],
        textColor=C_PRIMARY, fontSize=12,
        spaceBefore=10, spaceAfter=6,
        borderPad=4, fontName="Helvetica-Bold",
    ))


# ── Fonction principale ────────────────────────────────────
# 🔴 AJOUT DU PARAMÈTRE rapport_ia
def generate_report(summary: dict, output_path: str = "rapport_roadscan.pdf",
                    zone: str = "Meknès — Routes inspectées",
                    image_path: str = None,
                    rapport_ia: str = None) -> str:

    styles = getSampleStyleSheet()

    cover_frame   = Frame(0, 0, W, H, leftPadding=2*cm, rightPadding=2*cm,
                          topPadding=0, bottomPadding=0)
    content_frame = Frame(2*cm, 1.2*cm, W-4*cm, H-2.8*cm,
                          leftPadding=0, rightPadding=0,
                          topPadding=0.3*cm, bottomPadding=0.3*cm)

    doc = BaseDocTemplate(
        output_path, pagesize=A4,
        leftMargin=0, rightMargin=0, topMargin=0, bottomMargin=0,
    )
    doc.addPageTemplates([
        PageTemplate(id="cover",   frames=[cover_frame]),
        PageTemplate(id="content", frames=[content_frame], onPage=_on_page),
    ])

    story = []

    # ── Page 1 : Couverture ────────────────────────────────
    story += _make_cover(styles, summary, zone)

    from reportlab.platypus import NextPageTemplate
    story.append(NextPageTemplate("content"))

    # ── Page 2 : Résumé + graphique ───────────────────────
    story.append(_section_title(styles, "1. Résumé de l'Inspection"))

    resume_data = [
        ["Total détections",  str(summary.get("total", 0)),
         "Nids-de-poule",     str(summary.get("potholes", 0))],
        ["Fissures",          str(summary.get("cracks", 0)),
         "Indice sévérité",
         f"{summary.get('severity_index',0)}/100 "
         f"{summary.get('severity_level',{}).get('emoji','')} "
         f"{summary.get('severity_level',{}).get('label','')}"],
    ]
    resume_t = Table(resume_data, colWidths=[5*cm, 3*cm, 5*cm, 4*cm])
    resume_t.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (0,-1), C_DARK),
        ("BACKGROUND",  (2,0), (2,-1), C_DARK),
        ("TEXTCOLOR",   (0,0), (0,-1), C_WHITE),
        ("TEXTCOLOR",   (2,0), (2,-1), C_WHITE),
        ("FONTNAME",    (0,0), (-1,-1), "Helvetica"),
        ("FONTNAME",    (0,0), (0,-1),  "Helvetica-Bold"),
        ("FONTNAME",    (2,0), (2,-1),  "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 9),
        ("PADDING",     (0,0), (-1,-1), 8),
        ("GRID",        (0,0), (-1,-1), 0.4, C_GRAY),
        ("BACKGROUND",  (1,0), (1,-1), C_LIGHT),
        ("BACKGROUND",  (3,0), (3,-1), C_LIGHT),
    ]))
    story.append(resume_t)
    story.append(Spacer(1, 0.5*cm))

    story.append(_section_title(styles, "2. Répartition par Type de Dégradation"))
    story.append(_make_bar_chart(summary))
    story.append(Spacer(1, 0.3*cm))
    story.append(_make_class_table(summary))
    story.append(Spacer(1, 0.5*cm))

    # ── Photo annotée (optionnel) ──────────────────────────
    if image_path:
        story += _make_photo_section(styles, image_path)
        story.append(Spacer(1, 0.4*cm))

    # ── Recommandations ────────────────────────────────────
    num = "4" if image_path else "3"
    
    # 🔴 INTEGRATION DU RAPPORT IA ICI
    if rapport_ia:
        story.append(_section_title(styles, f"{num}. Recommandations d'Expertise (IA RAG)"))
        story += _parse_markdown_to_flowables(rapport_ia, styles)
    else:
        story.append(_section_title(styles, f"{num}. Recommandations"))
        story.append(_make_recommendations(styles, summary))
        
    story.append(Spacer(1, 0.6*cm))

    # ── Validation ─────────────────────────────────────────
    num2 = "5" if image_path else "4"
    story.append(_section_title(styles, f"{num2}. Validation Officielle"))
    story.append(_make_validation(styles))

    doc.build(story)
    return output_path


# ── Test rapide ────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Test pdf_report.py (version professionnelle) ===")
    from utils import format_summary

    fake = [
        {"class_id": 4, "class_name": "major_pothole",   "confidence": 0.87},
        {"class_id": 4, "class_name": "major_pothole",   "confidence": 0.81},
        {"class_id": 3, "class_name": "medium_pothole",  "confidence": 0.72},
        {"class_id": 3, "class_name": "medium_pothole",  "confidence": 0.65},
        {"class_id": 2, "class_name": "minor_pothole",   "confidence": 0.58},
        {"class_id": 0, "class_name": "linear_crack",    "confidence": 0.75},
        {"class_id": 1, "class_name": "alligator_crack", "confidence": 0.69},
    ]

    summary = format_summary(fake)
    
    # Simuler un rapport IA
    faux_rapport_ia = """
## 🔍 Diagnostic
Le tronçon présente une fatigue structurelle avec des nids-de-poule majeurs nécessitant une action rapide.
## ⚠️ Niveau d'urgence
**CRITIQUE**. Les fondations sont exposées.
## 🔧 Interventions
- **Nid-de-poule majeur** : Purge et compactage à chaud.
- **Fissure linéaire** : Pontage au mastic.
    """
    
    path    = generate_report(
        summary,
        output_path="test_rapport_pro.pdf",
        zone="Meknès — Avenue des FAR",
        rapport_ia=faux_rapport_ia
    )
    print(f"✅ Rapport généré : {path}")
    print(" === Fichier OK ===")