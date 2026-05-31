#!/usr/bin/env python3
# ============================================================
# RoadScan-MA — Telecharger et Consulter les 5 Datasets
# Fichier   : src/consulter_datasets.py
# Projet    : C:\Users\PC\RoadScan-MA\
# Usage     : Coller dans Google Colab et executer
# ============================================================
#
# STRUCTURE DU PROJET :
# RoadScan-MA/
# ├── annotations/
# ├── app/
# ├── data/              ← datasets telecharges ici
# │    ├── D1_RDD2022/
# │    ├── D2_lorenzoarcioni/
# │    ├── D3_potholes/
# │    ├── D4_severity/
# │    └── D5_roboflow/
# ├── docs/
# ├── models/
# ├── notebooks/
# ├── outputs/
# ├── src/               ← ce fichier est ici
# ├── app.py
# └── requirements.txt
# ============================================================

import os, sys, subprocess, random

def installer():
    for pkg in ["kaggle", "matplotlib", "Pillow"]:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

installer()

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR        = "/content/RoadScan-MA"
DATA_DIR        = f"{BASE_DIR}/data"
KAGGLE_USERNAME = "MohamedAmineBelasri765"
KAGGLE_TOKEN    = "KGAT_6ed695a127affd8e407c4f90393065b6"

# ============================================================
# INFORMATIONS DES 5 DATASETS
# ============================================================
DATASETS = {
    "D1": {
        "nom":             "RDD2022 — Road Damage Dataset 2022",
        "numero":          "Dataset 1",
        "source":          "Kaggle",
        "ref":             "aliabdelmenam/rdd-2022",
        "images":          "~6 000 (Inde uniquement sur 47 420 total)",
        "classes":         "D00 / D10 / D20 / D40",
        "usabilite":       "0.6875 / 1",
        "telechargements": "7 463",
        "taille":          "10.6 GB total — Inde ~1.5 GB",
        "fraicheur":       "Fevrier 2025",
        "format":          "XML Pascal VOC → convertir en YOLO .txt",
        "role":            "Base principale d'entrainement",
        "valeur_ajoutee":  "Dataset reference mondial — images indiennes proches des routes marocaines",
        "phase":           "Phase 1 — Entrainement de base",
        "lien":            "https://www.kaggle.com/datasets/aliabdelmenam/rdd-2022",
        "chemin":          f"{DATA_DIR}/D1_RDD2022",
        "ref_kaggle":      "aliabdelmenam/rdd-2022",
        "filtre":          "India",
    },
    "D2": {
        "nom":             "Road Damage Dataset — Potholes Cracks Manholes",
        "numero":          "Dataset 2",
        "source":          "Kaggle",
        "ref":             "lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes",
        "images":          "~2 000",
        "classes":         "Nids-de-poule / Fissures / Regards",
        "usabilite":       "0.875 / 1",
        "telechargements": "2 217",
        "taille":          "194 MB",
        "fraicheur":       "Fevrier 2026 — LE PLUS RECENT",
        "format":          "Images + annotations",
        "role":            "Routes reelles recentes — meme angle que notre smartphone",
        "valeur_ajoutee":  "Images prises depuis un vehicule en mouvement en 2026",
        "phase":           "Phase 1 — Entrainement de base",
        "lien":            "https://www.kaggle.com/datasets/lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes",
        "chemin":          f"{DATA_DIR}/D2_lorenzoarcioni",
        "ref_kaggle":      "lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes",
        "filtre":          None,
    },
    "D3": {
        "nom":             "Annotated Potholes Image Dataset",
        "numero":          "Dataset 3",
        "source":          "Kaggle",
        "ref":             "chitholian/annotated-potholes-dataset",
        "images":          "~665",
        "classes":         "D40 — Nids-de-poule uniquement",
        "usabilite":       "0.875 / 1",
        "telechargements": "6 230",
        "taille":          "~200 MB",
        "fraicheur":       "2021",
        "format":          "XML Pascal VOC avec bounding boxes",
        "role":            "Renforcer la detection des nids-de-poule D40",
        "valeur_ajoutee":  "Bounding boxes annotees manuellement — tres precises",
        "phase":           "Phase 1 — Entrainement de base",
        "lien":            "https://www.kaggle.com/datasets/chitholian/annotated-potholes-dataset",
        "chemin":          f"{DATA_DIR}/D3_potholes",
        "ref_kaggle":      "chitholian/annotated-potholes-dataset",
        "filtre":          None,
    },
    "D4": {
        "nom":             "Annotated Potholes with Severity Levels",
        "numero":          "Dataset 4",
        "source":          "Kaggle",
        "ref":             "Rechercher : pothole severity annotated",
        "images":          "~500",
        "classes":         "0=Leger / 1=Modere / 2=Critique",
        "usabilite":       "1.0 / 1 — SCORE PARFAIT",
        "telechargements": "1 171",
        "taille":          "Variable",
        "fraicheur":       "2023",
        "format":          "YOLO natif .txt — zero conversion",
        "role":            "Apprendre les niveaux de severite directement",
        "valeur_ajoutee":  "SEUL dataset avec severite pre-annotee — differenciateur RoadScan-MA",
        "phase":           "Phase 2 — Fine-tuning",
        "lien":            "https://www.kaggle.com/search?q=annotated+potholes+severity+levels",
        "chemin":          f"{DATA_DIR}/D4_severity",
        "ref_kaggle":      None,
        "filtre":          None,
    },
    "D5": {
        "nom":             "Pothole Detection YOLOv8 — Roboflow Universe",
        "numero":          "Dataset 5",
        "source":          "Roboflow Universe",
        "ref":             "universe.roboflow.com",
        "images":          "Variable",
        "classes":         "Nids-de-poule + Fissures",
        "usabilite":       "Format YOLOv8 natif — Score 13/15",
        "telechargements": "Voir Roboflow",
        "taille":          "Variable",
        "fraicheur":       "2024-2025",
        "format":          "YOLOv8 natif — ZERO preprocessing",
        "role":            "Tester le pipeline rapidement sans conversion",
        "valeur_ajoutee":  "Split train/val/test deja fait + augmentation deja appliquee",
        "phase":           "Phase 3 — Validation",
        "lien":            "https://universe.roboflow.com/search?q=pothole+detection+yolov8&t=object-detection",
        "chemin":          f"{DATA_DIR}/D5_roboflow",
        "ref_kaggle":      None,
        "filtre":          None,
    },
}

# ============================================================
# UTILITAIRES
# ============================================================
def configurer_kaggle():
    os.makedirs("/root/.kaggle", exist_ok=True)
    with open("/root/.kaggle/kaggle.json", "w") as f:
        f.write(f'{{"username":"{KAGGLE_USERNAME}","key":"{KAGGLE_TOKEN}"}}')
    os.chmod("/root/.kaggle/kaggle.json", 0o600)
    print("Kaggle configure")

def creer_dossiers():
    for key in DATASETS:
        os.makedirs(DATASETS[key]["chemin"], exist_ok=True)
    print(f"Dossiers data crees dans : {DATA_DIR}")

def trouver_images(dossier, filtre=None):
    images = []
    for root, dirs, files in os.walk(dossier):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                if filtre is None or filtre in f:
                    images.append(os.path.join(root, f))
    return images

def sep(titre):
    print("\n" + "="*60)
    print(f"  {titre}")
    print("="*60)

# ============================================================
# AFFICHAGE DES FICHES
# ============================================================
def afficher_fiche(key):
    d = DATASETS[key]
    sep(f"{d['numero']} — {d['nom']}")
    champs = [
        ("Source",           "source"),
        ("Reference",        "ref"),
        ("Nombre images",    "images"),
        ("Classes",          "classes"),
        ("Usabilite",        "usabilite"),
        ("Telechargements",  "telechargements"),
        ("Taille",           "taille"),
        ("Fraicheur",        "fraicheur"),
        ("Format",           "format"),
        ("Role",             "role"),
        ("Valeur ajoutee",   "valeur_ajoutee"),
        ("Phase",            "phase"),
        ("Lien",             "lien"),
        ("Chemin Colab",     "chemin"),
    ]
    for label, cle in champs:
        print(f"  {label:<18} : {d[cle]}")

# ============================================================
# AFFICHAGE DES IMAGES
# ============================================================
def afficher_images(key, n=10):
    d      = DATASETS[key]
    chemin = d["chemin"]
    filtre = d["filtre"]

    if not os.path.exists(chemin) or not os.listdir(chemin):
        print(f"\nDataset non telecharge dans : {chemin}")
        if d["ref_kaggle"]:
            print(f"Lance : telecharger_{key.lower()}()")
        else:
            print(f"Voir : {d['lien']}")
        return

    images = trouver_images(chemin, filtre)
    if not images:
        print(f"Aucune image dans {chemin}")
        return

    print(f"\nTotal images : {len(images)}")
    echantillon = random.sample(images, min(n, len(images)))

    cols = 5
    rows = (len(echantillon) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 4))
    axes_flat = axes.flat if rows > 1 else (axes if cols > 1 else [axes])

    fig.suptitle(
        f"{d['nom']} — {d['images']} images\n{d['role']}",
        fontsize=11, fontweight='bold'
    )
    for i, ax in enumerate(axes_flat):
        if i < len(echantillon):
            try:
                img  = mpimg.imread(echantillon[i])
                nom  = os.path.basename(echantillon[i])
                size = os.path.getsize(echantillon[i]) / 1024
                ax.imshow(img)
                ax.set_title(f"{nom[:16]}\n{size:.1f} kB", fontsize=7)
            except:
                ax.text(0.5, 0.5, 'Erreur', ha='center', va='center')
        else:
            ax.set_visible(False)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# ============================================================
# TELECHARGEMENTS
# ============================================================
def telecharger_d1():
    print("\nDataset 1 — RDD2022 (~15-20 min)")
    chemin = DATASETS["D1"]["chemin"]
    os.makedirs(chemin, exist_ok=True)
    os.system(f"kaggle datasets download -d aliabdelmenam/rdd-2022 -p {chemin} --unzip -q")
    imgs = trouver_images(chemin, "India")
    print(f"Dataset 1 OK — {len(imgs)} images Inde")

def telecharger_d2():
    print("\nDataset 2 — lorenzoarcioni (~2-3 min)")
    chemin = DATASETS["D2"]["chemin"]
    os.makedirs(chemin, exist_ok=True)
    os.system(f"kaggle datasets download -d lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes -p {chemin} --unzip -q")
    imgs = trouver_images(chemin)
    print(f"Dataset 2 OK — {len(imgs)} images")

def telecharger_d3():
    print("\nDataset 3 — Annotated Potholes (~1-2 min)")
    chemin = DATASETS["D3"]["chemin"]
    os.makedirs(chemin, exist_ok=True)
    os.system(f"kaggle datasets download -d chitholian/annotated-potholes-dataset -p {chemin} --unzip -q")
    imgs = trouver_images(chemin)
    print(f"Dataset 3 OK — {len(imgs)} images")

def telecharger_rapide():
    """Telecharger D2 + D3 seulement (petits et rapides)"""
    configurer_kaggle()
    creer_dossiers()
    telecharger_d2()
    telecharger_d3()
    sep("D2 + D3 telecharges dans data/")

def telecharger_tout():
    """Telecharger D1 + D2 + D3"""
    configurer_kaggle()
    creer_dossiers()
    telecharger_d1()
    telecharger_d2()
    telecharger_d3()
    sep("D1 + D2 + D3 telecharges")
    print(f"  Chemin : {DATA_DIR}/")
    print(f"\n  D4 : {DATASETS['D4']['lien']}")
    print(f"  D5 : {DATASETS['D5']['lien']}")

# ============================================================
# CONSULTATION PAR DATASET
# ============================================================
def consulter_d1():
    afficher_fiche("D1")
    afficher_images("D1", n=10)

def consulter_d2():
    afficher_fiche("D2")
    afficher_images("D2", n=10)

def consulter_d3():
    afficher_fiche("D3")
    afficher_images("D3", n=10)

def consulter_d4():
    afficher_fiche("D4")
    afficher_images("D4", n=10)

def consulter_d5():
    afficher_fiche("D5")
    print(f"\nOuvrir dans le navigateur : {DATASETS['D5']['lien']}")

# ============================================================
# RESUME ET STATISTIQUES
# ============================================================
def resume_tous():
    sep("ROADSCAN-MA — RESUME DES 6 DATASETS")
    lignes = [
        ("#",  "Dataset",              "Images",   "Usabilite", "Phase",   "Role"),
        ("1",  "RDD2022 Inde",         "~6 000",   "0.6875",    "Phase 1", "Base principale"),
        ("2",  "lorenzoarcioni",        "~2 000",   "0.875",     "Phase 1", "Routes reelles 2026"),
        ("3",  "Annotated Potholes",    "~665",     "0.875",     "Phase 1", "Specialise D40"),
        ("4",  "Severity Levels",       "~500",     "1.0",       "Phase 2", "Niveaux severite"),
        ("5",  "Roboflow yolov8",       "Variable", "YOLO natif","Phase 3", "Zero conversion"),
        ("MA", "Dataset Marocain",      "200-500",  "A collecter","Phase 2","Valeur ajoutee"),
    ]
    widths = [4, 24, 10, 12, 10, 22]
    for row in lignes:
        print("  " + "".join(str(v).ljust(widths[i]) for i, v in enumerate(row)))

    print("\n  Solution poids dataset marocain :")
    print("  300 images x augmentation x10 x repetition x3 = 9 000 = 20% du poids")
    print(f"\n  Chemin data sur Colab : {DATA_DIR}/")

def statistiques():
    sep("STATISTIQUES DES DATASETS TELECHARGES")
    total = 0
    for key, d in DATASETS.items():
        chemin = d["chemin"]
        if os.path.exists(chemin) and os.listdir(chemin):
            imgs   = trouver_images(chemin, d.get("filtre"))
            total += len(imgs)
            print(f"  {d['numero']:<12} → {len(imgs):>6} images  {chemin}")
        else:
            print(f"  {d['numero']:<12} → Non telecharge")
    print(f"\n  TOTAL images disponibles : {total}")

# ============================================================
# MENU
# ============================================================
def menu():
    sep("ROADSCAN-MA — MENU CONSULTATION DATASETS")
    print("  1    → Fiche + images  Dataset 1 (RDD2022 Inde)")
    print("  2    → Fiche + images  Dataset 2 (lorenzoarcioni)")
    print("  3    → Fiche + images  Dataset 3 (Annotated Potholes)")
    print("  4    → Fiche + images  Dataset 4 (Severity Levels)")
    print("  5    → Fiche + images  Dataset 5 (Roboflow)")
    print("  all  → Resume de tous les datasets")
    print("  stat → Statistiques des datasets telecharges")
    print("  dl2  → Telecharger D2 + D3 seulement (~3 min)")
    print("  dl   → Telecharger D1 + D2 + D3 (~20 min)")
    print("="*60)

    choix = input("  Ton choix : ").strip().lower()

    if   choix == "1":    consulter_d1()
    elif choix == "2":    consulter_d2()
    elif choix == "3":    consulter_d3()
    elif choix == "4":    consulter_d4()
    elif choix == "5":    consulter_d5()
    elif choix == "all":  resume_tous()
    elif choix == "stat": statistiques()
    elif choix == "dl2":  telecharger_rapide()
    elif choix == "dl":   telecharger_tout()
    else:
        print("Choix non reconnu — tape 1, 2, 3, 4, 5, all, stat, dl2 ou dl")

# ============================================================
# LANCEMENT
# ============================================================
if __name__ == "__main__":
    resume_tous()
    menu()
