"""Extraction PyMuPDF, adaptée des cellules 9–14 du support officiel."""
import hashlib
import re
from pathlib import Path

import pymupdf

from .config import Settings, ecrire_json, lire_json


def extraire(pdf: Path, settings: Settings) -> dict:
    pdf = Path(pdf)
    if not pdf.is_file():
        raise FileNotFoundError("PDF introuvable. Utiliser la commande de préparation.")
    digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
    dossier = settings.data / "extracted" / digest
    manifest = dossier / "manifest.json"
    ancien = lire_json(manifest)
    if ancien and all(settings.chemin(x["source"]).is_file() for x in ancien["textes"] + ancien["images"]):
        return ancien
    dossier.mkdir(parents=True, exist_ok=True)
    textes, images, complets, deja_vus = [], [], [], {}
    with pymupdf.open(pdf) as document:
        if document.needs_pass:
            raise ValueError("Les PDF protégés par mot de passe ne sont pas pris en charge.")
        for numero, page in enumerate(document, 1):
            texte = page.get_text("text", sort=True)
            fichier = dossier / f"page_{numero:03d}.txt"
            fichier.write_text(texte, encoding="utf-8")
            textes.append({"page": numero, "source": fichier.relative_to(settings.data).as_posix()})
            complets.append(f"===== PAGE PDF {numero} =====\n{texte}")
            for xref in sorted({x[0] for x in page.get_images(full=True)}):
                image = document.extract_image(xref)
                if not image:
                    continue
                cle = hashlib.sha256(image["image"]).hexdigest()
                if cle not in deja_vus:
                    cible = dossier / f"image_{cle[:16]}.{image['ext']}"
                    cible.write_bytes(image["image"])
                    deja_vus[cle] = cible.relative_to(settings.data).as_posix()
                images.append({"page": numero, "source": deja_vus[cle], "sha256": cle,
                               "figure": 0, "methode": "image_integree"})
            # get_images() ignore les figures vectorielles. Leurs dessins sont rendus
            # dans la région située juste au-dessus de leur légende « Figure n: ».
            dessins = [d["rect"] for d in page.get_drawings() if not d["rect"].is_empty]
            rasters = [(xref, r) for xref in {x[0] for x in page.get_images(full=True)}
                       for r in page.get_image_rects(xref)]
            blocs = page.get_text("blocks")
            captions = sorted([(b, re.search(r"Figure\s+(\d+)\s*:", b[4])) for b in blocs
                               if re.search(r"Figure\s+(\d+)\s*:", b[4])], key=lambda x: x[0][1])
            limite = 0
            for bloc, match in captions:
                candidats = [r for r in dessins if r.y0 >= limite and r.y1 <= bloc[1] + 3]
                rasters_figure = [(xref, r) for xref, r in rasters
                                  if r.y0 >= limite and r.y1 <= bloc[1] + 3]
                # Une figure raster unique est déjà extraite : ne pas la dupliquer.
                if len(rasters_figure) == 1 and not candidats:
                    xref = rasters_figure[0][0]
                    digest_image = hashlib.sha256(document.extract_image(xref)["image"]).hexdigest()
                    for item in images:
                        if item["page"] == numero and item["sha256"] == digest_image:
                            item["figure"] = int(match.group(1))
                    limite = bloc[3]
                    continue
                candidats += [r for _, r in rasters_figure]
                if not candidats:
                    limite = bloc[3]
                    continue
                region = pymupdf.Rect(candidats[0])
                for rect in candidats[1:]:
                    region |= rect
                region = pymupdf.Rect(region.x0 - 35, region.y0 - 25,
                                     region.x1 + 10, min(region.y1 + 10, bloc[1])) & page.rect
                if region.width < 50 or region.height < 50:
                    continue
                nombre = int(match.group(1))
                # Une composition de plusieurs sous-images est indexée comme une
                # figure complète. Les fichiers bruts restent extraits sur disque.
                hashes_composants = {hashlib.sha256(document.extract_image(xref)["image"]).hexdigest()
                                     for xref, _ in rasters_figure}
                images[:] = [im for im in images if not (im["page"] == numero and im["sha256"] in hashes_composants)]
                cible = dossier / f"page_{numero:03d}_figure_{nombre}.png"
                page.get_pixmap(matrix=pymupdf.Matrix(2, 2), clip=region, alpha=False).save(cible)
                images.append({"page": numero, "source": cible.relative_to(settings.data).as_posix(),
                               "sha256": hashlib.sha256(cible.read_bytes()).hexdigest(),
                               "figure": nombre, "methode": "rendu_region", "bbox": list(region)})
                limite = bloc[3]
        resultat = {"document_id": digest, "pdf": pdf.name, "pages": len(document),
                    "textes": textes, "images": images, "images_integrees_uniques": len(deja_vus),
                    "pages_sans_texte": [x["page"] for x in textes
                        if not settings.chemin(x["source"]).read_text(encoding="utf-8").strip()]}
    (dossier / "texte_complet.txt").write_text("\n\n".join(complets), encoding="utf-8")
    ecrire_json(manifest, resultat)
    return resultat
