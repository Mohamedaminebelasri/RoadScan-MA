import os

structure = {
    "donnees/videos_brutes": [],
    "donnees/images_extraites": [],
    "donnees/annotations": [],
    "modele_entraine": [],
    "code_source": [
        "preparation_donnees.py",
        "detection_degradations.py",
        "localisation_gps.py",
        "generation_carte.py",
        "generation_rapport_pdf.py",
        "export_google_maps.py"
    ],
    "resultats/cartes_interactives": [],
    "resultats/rapports_pdf": [],
    "resultats/fichiers_google_maps": [],
}

fichiers_racine = [
    "interface_utilisateur.py",
    "liste_bibliotheques.txt"
]

for dossier, fichiers in structure.items():
    os.makedirs(dossier, exist_ok=True)
    print(f"Dossier cree : {dossier}/")
    for fichier in fichiers:
        chemin = os.path.join(dossier, fichier)
        with open(chemin, "w") as f:
            f.write(f"# {fichier} - RoadScan-MA\n")
        print(f"   Fichier cree : {chemin}")

for fichier in fichiers_racine:
    with open(fichier, "w") as f:
        f.write(f"# {fichier} - RoadScan-MA\n")
    print(f"Fichier cree : {fichier}")

print("\nStructure RoadScan-MA creee avec succes !")
print(f"Emplacement : C:\\Users\\PC\\Desktop\\projetCV")