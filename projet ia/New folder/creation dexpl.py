from pathlib import Path

import cv2
import numpy as np

SORTIE = Path(__file__).resolve().parent / "scans"
SORTIE.mkdir(exist_ok=True)


def page(lignes, nom, angle=2.0):
    img = np.full((1100, 800, 3), 255, np.uint8)
    y = 80
    for texte, taille in lignes:
        cv2.putText(img, texte, (50, y), cv2.FONT_HERSHEY_SIMPLEX,
                    taille, (20, 20, 20), 2 if taille > 0.8 else 1, cv2.LINE_AA)
        y += int(55 * max(taille, 0.7) + 15)
    # petite rotation + bruit pour simuler un vrai scan
    M = cv2.getRotationMatrix2D((400, 550), angle, 1.0)
    img = cv2.warpAffine(img, M, (800, 1100), borderValue=(255, 255, 255))
    bruit = np.random.normal(0, 8, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + bruit, 0, 255).astype(np.uint8)
    cv2.imwrite(str(SORTIE / nom), img)


page([
    ("FACTURE", 1.4),
    ("Facture N: FA-2026-0142", 0.8),
    ("Date: 15/11/2026", 0.8),
    ("Client: Societe Tunis Services", 0.8),
    ("Email: contact@tunisservices.tn", 0.8),
    ("Tel: 71 123 456", 0.8),
    ("Designation        Quantite     Prix unitaire", 0.7),
    ("Ordinateur portable    2        1250,000 DT", 0.7),
    ("Souris sans fil        5          35,500 DT", 0.7),
    ("Total HT: 2677,500 DT", 0.8),
    ("TVA 19%: 508,725 DT", 0.8),
    ("Total TTC: 3186,225 DT", 0.9),
], "facture_exemple.png")

page([
    ("ATTESTATION DE TRAVAIL", 1.1),
    ("Je soussigne, Directeur des Ressources", 0.8),
    ("Humaines, atteste que Monsieur Ahmed Ben Ali,", 0.8),
    ("CIN: 09876543, est employe dans notre", 0.8),
    ("societe depuis le 01/09/2020.", 0.8),
    ("Cette attestation est delivree pour servir", 0.8),
    ("et valoir ce que de droit.", 0.8),
    ("Fait a Tunis, le 20/11/2026", 0.8),
], "attestation_exemple.png", angle=-1.5)

print("Exemples crees dans :", SORTIE)

