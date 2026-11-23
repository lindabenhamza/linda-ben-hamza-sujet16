
"""
Sujet 16 - Reconnaissance automatique des documents administratifs

Reconnaitre le texte contenu dans des documents scannes et convertir les
informations en donnees exploitables (JSON / CSV).

Pipeline :
    1. Chargement (images ou PDF)
    2. Pretraitement (niveaux de gris, debruitage, redressement, binarisation)
    3. OCR (Tesseract ou EasyOCR)
    4. Classification du type de document (facture, CIN, attestation...)
    5. Extraction des champs (dates, montants, email, telephone, CIN, IBAN...)
    6. Export JSON + CSV


"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

EXTENSIONS_IMAGES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
EXTENSIONS_PDF = {".pdf"}



# 1. Chargement
def charger_images(chemin: Path):
    """Retourne une liste d'images BGR (une par page pour les PDF)."""
    ext = chemin.suffix.lower()
    if ext in EXTENSIONS_PDF:
        try:
            import fitz  # PyMuPDF
        except ImportError:
            sys.exit("Pour lire les PDF : pip install pymupdf")
        pages = []
        with fitz.open(chemin) as doc:
            for page in doc:
                pix = page.get_pixmap(dpi=300)
                img = np.frombuffer(pix.samples, dtype=np.uint8)
                img = img.reshape(pix.height, pix.width, pix.n)
                if pix.n == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                elif pix.n == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                pages.append(img)
        return pages

    # imdecode gere mieux les chemins avec accents sous Windows
    data = np.fromfile(str(chemin), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Impossible de lire l'image : {chemin}")
    return [img]



# 2. Pretraitement
def redresser(gris: np.ndarray) -> np.ndarray:
    """Corrige une legere inclinaison (-10 a +10 degres) du scan."""
    h, w = gris.shape
    echelle = 600 / w if w > 600 else 1.0
    petit = cv2.resize(gris, None, fx=echelle, fy=echelle)
    _, binaire = cv2.threshold(petit, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    meilleur_angle, meilleur_score = 0.0, -1.0
    ph, pw = binaire.shape
    centre = (pw // 2, ph // 2)
    for angle in np.arange(-10, 10.5, 0.5):
        M = cv2.getRotationMatrix2D(centre, angle, 1.0)
        rot = cv2.warpAffine(binaire, M, (pw, ph), flags=cv2.INTER_NEAREST)
        score = np.var(np.sum(rot, axis=1))
        if score > meilleur_score:
            meilleur_score, meilleur_angle = score, angle

    if abs(meilleur_angle) < 0.3:
        return gris
    M = cv2.getRotationMatrix2D((w // 2, h // 2), meilleur_angle, 1.0)
    return cv2.warpAffine(gris, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def pretraiter(img_bgr: np.ndarray) -> np.ndarray:
    """Niveaux de gris -> agrandissement -> debruitage -> redressement -> binarisation."""
    gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Agrandir les petites images (l'OCR fonctionne mieux autour de 300 dpi)
    if max(gris.shape) < 1500:
        gris = cv2.resize(gris, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    gris = cv2.fastNlMeansDenoising(gris, None, h=10)
    gris = redresser(gris)

    binaire = cv2.adaptiveThreshold(
        gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )
    return binaire



# 3. OCR
class MoteurOCR:
    """Enveloppe commune pour Tesseract et EasyOCR."""

    def __init__(self, moteur: str, langues: str, tesseract_cmd: str = None):
        self.moteur = moteur
        if moteur == "tesseract":
            try:
                import pytesseract
            except ImportError:
                sys.exit("pip install pytesseract")
            if tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            self.pytesseract = pytesseract
            self.langues = langues  # ex: "fra+eng"
        elif moteur == "easyocr":
            try:
                import easyocr
            except ImportError:
                sys.exit("pip install easyocr")
            codes = [c for c in re.split(r"[+,]", langues) if c]
            mapping = {"fra": "fr", "eng": "en", "ara": "ar"}
            codes = [mapping.get(c, c) for c in codes]
            self.reader = easyocr.Reader(codes, gpu=False)
        else:
            raise ValueError("Moteur inconnu : " + moteur)

    def lire(self, image: np.ndarray):
        """Retourne (texte, confiance_moyenne_en_%)."""
        if self.moteur == "tesseract":
            config = "--oem 3 --psm 6"
            texte = self.pytesseract.image_to_string(
                image, lang=self.langues, config=config)
            data = self.pytesseract.image_to_data(
                image, lang=self.langues, config=config,
                output_type=self.pytesseract.Output.DICT)
            confs = [float(c) for c, t in zip(data["conf"], data["text"])
                     if str(c) != "-1" and t.strip()]
            return texte, (sum(confs) / len(confs) if confs else 0.0)

        # EasyOCR : regrouper les boites par ligne
        resultats = self.reader.readtext(image)
        if not resultats:
            return "", 0.0
        elements = []
        for boite, txt, conf in resultats:
            ys = [p[1] for p in boite]
            xs = [p[0] for p in boite]
            elements.append({"x": min(xs), "y": (min(ys) + max(ys)) / 2,
                             "h": max(ys) - min(ys), "txt": txt, "conf": conf})
        elements.sort(key=lambda e: e["y"])
        lignes, courante = [], [elements[0]]
        for e in elements[1:]:
            if abs(e["y"] - courante[-1]["y"]) < 0.6 * max(e["h"], 1):
                courante.append(e)
            else:
                lignes.append(courante)
                courante = [e]
        lignes.append(courante)
        texte = "\n".join(
            " ".join(e["txt"] for e in sorted(l, key=lambda e: e["x"]))
            for l in lignes)
        conf_moy = 100 * sum(e["conf"] for e in elements) / len(elements)
        return texte, conf_moy



# 4. Classification du type de document (par mots-cles)
TYPES_DOCUMENTS = {
    "facture": ["facture", "total ttc", "total ht", "tva", "montant", "client",
                "quantite", "prix unitaire", "invoice"],
    "carte_identite": ["carte d'identite", "carte nationale", "cin",
                       "republique tunisienne", "nom et prenom", "ne le", "a "
                       "date de naissance"],
    "attestation": ["attestation", "atteste", "certifie", "je soussigne",
                    "delivree", "pour servir et valoir"],
    "releve_bancaire": ["releve", "iban", "solde", "debit", "credit", "rib",
                        "banque", "agence"],
    "contrat": ["contrat", "entre les soussignes", "article", "clause",
                "parties", "resiliation"],
    "bulletin_paie": ["bulletin de paie", "salaire", "net a payer", "cnss",
                      "brut", "retenue", "cotisation"],
    "recu": ["recu", "recu de paiement", "paye", "quittance", "versement"],
    "lettre_administrative": ["objet", "monsieur", "madame", "veuillez agreer",
                              "salutations", "cordialement"],
}


def normaliser(txt: str) -> str:
    """Minuscules + suppression des accents (pour comparer les mots-cles)."""
    import unicodedata
    txt = unicodedata.normalize("NFD", txt.lower())
    return "".join(c for c in txt if unicodedata.category(c) != "Mn")


def classifier(texte: str):
    """Retourne (type, score) selon le nombre de mots-cles trouves."""
    t = normaliser(texte)
    scores = {}
    for type_doc, mots in TYPES_DOCUMENTS.items():
        scores[type_doc] = sum(1 for m in mots if normaliser(m) in t)
    meilleur = max(scores, key=scores.get)
    if scores[meilleur] == 0:
        return "inconnu", 0
    return meilleur, scores[meilleur]



# 5. Extraction des informations
REGEX = {
    "dates": r"\b(?:0?[1-9]|[12]\d|3[01])[/.\-](?:0?[1-9]|1[0-2])[/.\-](?:19|20)?\d{2}\b",
    "emails": r"[\w.+\-]+@[\w\-]+\.[\w.\-]+",
    "telephones": r"(?:\+216|00216)?[\s.]?\b\d{2}[\s.]?\d{3}[\s.]?\d{3}\b",
    "montants": r"(?<![\d/.\-])\d+(?: \d{3})*[.,]\d{2,3}(?![\d/]|[.,]\d)\s?(?:DT|TND|EUR|€|\$)?",
    "iban": r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){3,7}(?:\s?[A-Z0-9]{1,4})?\b",
    "cin": r"(?:CIN|C\.I\.N)\D{0,5}(\d{8})",
    "numero_document": r"\b(?:n[°o]?|num[eé]ro)\s*[:°.]\s*(?=[A-Z0-9/\-]*\d)([A-Z0-9][A-Z0-9/\-]{2,})",
    "matricule_fiscal": r"\b\d{7}\s?[A-Z]\s?/?\s?[A-Z]\s?/?\s?[A-Z]?\s?/?\s?\d{3}\b",
}


def extraire_champs(texte: str) -> dict:
    champs = {}
    for nom, motif in REGEX.items():
        flags = re.IGNORECASE if nom in ("cin", "numero_document") else 0
        trouves = re.findall(motif, texte, flags)
        # dedoublonner en gardant l'ordre
        propres = list(dict.fromkeys(t.strip() for t in trouves if t.strip()))
        if propres:
            champs[nom] = propres

    # Un numero de CIN (8 chiffres) ne doit pas etre pris pour un telephone
    if "cin" in champs and "telephones" in champs:
        chiffres = lambda x: re.sub(r"\D", "", x)
        tel = [t for t in champs["telephones"]
               if chiffres(t)[-8:] not in {chiffres(c) for c in champs["cin"]}]
        if tel:
            champs["telephones"] = tel
        else:
            del champs["telephones"]

    # Total : on prefere "TTC" / "net a payer", sinon le dernier "total"
    candidats = []
    for ligne in texte.splitlines():
        l = normaliser(ligne)
        if "total" in l or "net a payer" in l:
            m = re.search(REGEX["montants"], ligne)
            if m:
                priorite = 2 if ("ttc" in l or "net a payer" in l) else 1
                candidats.append((priorite, m.group(0).strip()))
    if candidats:
        champs["total"] = max(candidats, key=lambda c: c[0])[1]
    return champs



# 6. Traitement d'un fichier + export
def traiter_fichier(chemin: Path, moteur: MoteurOCR, dossier_debug: Path = None):
    resultats = []
    for num_page, img in enumerate(charger_images(chemin), start=1):
        prep = pretraiter(img)
        if dossier_debug:
            dossier_debug.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dossier_debug / f"{chemin.stem}_p{num_page}.png"), prep)

        texte, confiance = moteur.lire(prep)
        type_doc, score = classifier(texte)
        resultats.append({
            "fichier": chemin.name,
            "page": num_page,
            "type_document": type_doc,
            "score_classification": score,
            "confiance_ocr": round(confiance, 1),
            "champs": extraire_champs(texte),
            "texte": texte.strip(),
        })
    return resultats


def lister_fichiers(entree: Path):
    if entree.is_file():
        return [entree]
    ok = EXTENSIONS_IMAGES | EXTENSIONS_PDF
    return sorted(p for p in entree.rglob("*") if p.suffix.lower() in ok)


def exporter(resultats, dossier: Path):
    dossier.mkdir(parents=True, exist_ok=True)

    with open(dossier / "resultats.json", "w", encoding="utf-8") as f:
        json.dump(resultats, f, ensure_ascii=False, indent=2)

    with open(dossier / "resultats.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["fichier", "page", "type_document", "confiance_ocr",
                    "dates", "montants", "total", "emails", "telephones",
                    "cin", "iban", "numero_document"])
        for r in resultats:
            c = r["champs"]
            joindre = lambda k: " | ".join(c.get(k, [])) if isinstance(c.get(k), list) else c.get(k, "")
            w.writerow([r["fichier"], r["page"], r["type_document"],
                        r["confiance_ocr"], joindre("dates"), joindre("montants"),
                        joindre("total"), joindre("emails"), joindre("telephones"),
                        joindre("cin"), joindre("iban"), joindre("numero_document")])

    # Un fichier texte brut par document
    for r in resultats:
        nom = f"{Path(r['fichier']).stem}_p{r['page']}.txt"
        (dossier / nom).write_text(r["texte"], encoding="utf-8")


# Programme principal
def main():
    p = argparse.ArgumentParser(
        description="OCR et extraction d'informations de documents administratifs")
    p.add_argument("entree", type=Path, nargs="?", default=None,
                   help="image, PDF ou dossier de scans (optionnel : sinon "
                        "dossier 'scans' ou fenetre de choix de fichier)")
    p.add_argument("--moteur", choices=["tesseract", "easyocr"], default="easyocr")
    p.add_argument("--langues", default="fra+eng",
                   help="langues OCR, ex: fra+eng (defaut), fra+ara")
    p.add_argument("--sortie", type=Path, default=Path("resultats_ocr"))
    p.add_argument("--tesseract-cmd", default=None,
                   help="chemin de tesseract.exe si hors PATH")
    p.add_argument("--debug", action="store_true",
                   help="sauvegarde les images pretraitees")
    args = p.parse_args()

    # Aucun argument (ex: bouton "Run" de Thonny) :
    # 1) dossier "scans" a cote du script, 2) sinon fenetre de choix de fichier
    if args.entree is None:
        dossier_scans = Path(__file__).resolve().parent / "scans"
        if dossier_scans.is_dir() and lister_fichiers(dossier_scans):
            args.entree = dossier_scans
        else:
            try:
                import tkinter as tk
                from tkinter import filedialog
                racine = tk.Tk()
                racine.withdraw()
                racine.attributes("-topmost", True)
                choix = filedialog.askopenfilename(
                    title="Choisir un document scanne",
                    filetypes=[("Documents", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.pdf")])
                racine.destroy()
                if choix:
                    args.entree = Path(choix)
            except Exception:
                pass
        if args.entree is None:
            sys.exit("Aucun document choisi. Mettez vos scans dans le dossier "
                     "'scans' ou lancez : python ocr_documents_administratifs.py image.jpg")

    if not args.entree.exists():
        sys.exit(f"Introuvable : {args.entree}")

    fichiers = lister_fichiers(args.entree)
    if not fichiers:
        sys.exit("Aucun document (image/PDF) trouve.")

    print(f"Moteur : {args.moteur} | {len(fichiers)} fichier(s)")
    moteur = MoteurOCR(args.moteur, args.langues, args.tesseract_cmd)
    dossier_debug = args.sortie / "debug" if args.debug else None

    tous = []
    for f in fichiers:
        print(f"  -> {f.name} ...", end=" ", flush=True)
        try:
            res = traiter_fichier(f, moteur, dossier_debug)
            tous.extend(res)
            print(f"{res[0]['type_document']} (confiance {res[0]['confiance_ocr']}%)")
        except Exception as e:
            print(f"ERREUR : {e}")

    exporter(tous, args.sortie)
    print(f"\nTermine. Resultats dans : {args.sortie.resolve()}")


if __name__ == "__main__":
    main()

