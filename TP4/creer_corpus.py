"""Corpus français originaux, narratif et semi-synthétique, créés pour ce TP."""
import hashlib
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RECIT = """le carnet des heures tranquilles

au début du mois d'août, une petite bibliothèque ouvrait ses fenêtres sur une place calme. la lumière dessinait un carré sur le bureau. une bibliothécaire préparait un carnet destiné aux visiteurs. elle souhaitait raconter la vie du quartier sans transformer chaque souvenir en règle générale. son premier texte parlait d'une île imaginaire, d'un cœur curieux et d'un voyageur naïf. il contenait des détails précis, mais aucune personne réelle ne devait être reconnue dans cette fiction.

chaque matin, un technicien traversait la place avec une boîte à outils. il vérifiait les lampes, relevait une mesure et notait l'heure. un chiffre isolé lui semblait peu utile. pour comprendre une variation, il cherchait ce qui avait changé avant elle. la température, l'ouverture des portes et la présence des visiteurs formaient un contexte. il ne confondait pas une observation avec une explication. parfois, il revenait simplement le lendemain pour comparer deux journées.

dans l'atelier voisin, une artisane réparait des objets donnés par les habitants. elle commençait par observer les pièces, puis dessinait un plan très simple. un appareil silencieux n'était pas nécessairement un appareil sans défaut. elle testait chaque fonction séparément. lorsqu'une réparation échouait, elle conservait ses notes au lieu de raconter une réussite imaginaire. le carnet servait à retrouver les essais utiles et à éviter de refaire exactement le même travail.

le jardin de la bibliothèque suivait un autre rythme. les plantes recevaient de l'eau le soir, lorsque la chaleur diminuait. une rangée de semis poussait à l'ombre, une autre près du mur ensoleillé. les jardiniers comparaient leur développement sans oublier la différence de lumière. une plante plus haute ne prouvait pas qu'une méthode était toujours meilleure. les racines, le sol et les semaines précédentes comptaient aussi. cette prudence rendait les discussions plus intéressantes.

une table de lecture réunissait des livres de cuisine, des récits de voyage et des carnets de musique. les visiteurs choisissaient un sujet puis échangeaient leurs impressions. certains mots revenaient souvent, tandis que d'autres n'apparaissaient qu'une seule fois. un lecteur remarqua qu'un mot inconnu pouvait parfois être compris grâce à la phrase entière. il regardait à gauche et à droite, cherchait les relations entre les idées et acceptait de rester incertain quand le contexte manquait.

à l'étage, une classe préparait un petit journal. les élèves écrivaient plusieurs versions de chaque paragraphe. ils devaient annoncer clairement ce qui avait été observé et ce qui restait une hypothèse. une courbe élégante ne remplaçait pas la description de l'expérience. un nombre accompagné de son unité devenait plus facile à interpréter. la classe apprit aussi à garder une partie des exemples de côté pour vérifier ses choix après le travail.

le compteur du bâtiment enregistrait une puissance qui changeait au fil des heures. le matin, la cuisine et les éclairages fonctionnaient ensemble. à midi, certaines salles restaient vides. le soir, un atelier pouvait prolonger l'activité. personne ne supposait que tous les jours se ressemblaient exactement. quand une mesure manquait, le technicien la signalait. il préférait un intervalle visible dans son graphique à une ligne continue qui aurait donné une fausse impression de certitude.

un après-midi de pluie, les visiteurs imaginèrent une machine capable de poursuivre une histoire. ils lui proposèrent une courte amorce puis examinèrent la suite. quelques phrases semblaient fluides, d'autres mélangeaient les lieux et les objets. une continuation plausible pouvait être inventée de toutes pièces. le groupe comprit que reconnaître des régularités dans un texte n'était pas la même chose que connaître le monde. le plaisir de lire pouvait coexister avec une vérification attentive.

la salle de musique possédait une grande horloge. un instrument jouait une note, puis un autre lui répondait. pour suivre la mélodie, il fallait conserver une partie du passé. pour lire une partition déjà écrite, on pouvait examiner plusieurs mesures ensemble. ces deux façons de traiter une séquence n'avaient pas les mêmes contraintes. elles rendaient certaines relations faciles à repérer et en cachaient parfois d'autres. aucune comparaison sérieuse ne pouvait oublier la tâche demandée.

avant de fermer la bibliothèque, l'équipe relisait le carnet. elle corrigeait les unités, vérifiait les dates et séparait les brouillons des textes achevés. les documents extérieurs restaient attribués à leurs auteurs. les dessins nouveaux étaient conservés avec leurs sources de travail. on pouvait reprendre un chapitre sans effacer les observations précédentes. le carnet ne prétendait pas tout expliquer ; il aidait surtout à poser des questions plus précises et à partager honnêtement ce qui avait été appris.
"""

THEMES = [
    (['le technicien','la technicienne','le chercheur','la chercheuse'], ['vérifie la mesure du capteur','compare les deux courbes','note la température exacte','prépare une expérience simple','contrôle les unités du tableau','examine les résultats du laboratoire']),
    (['le lecteur','la lectrice','le libraire','la bibliothécaire'], ['ouvre un livre illustré','range les ouvrages du rayon','cherche une histoire courte','relit un passage difficile','consulte le catalogue local','partage une découverte littéraire']),
    (['le jardinier','la jardinière','le voisin','la voisine'], ['arrose les jeunes plantes','observe les feuilles du jardin','récolte les légumes mûrs','prépare la terre humide','protège les fleurs fragiles','plante un nouvel arbre']),
    (['le cuisinier','la cuisinière','le responsable','la responsable'], ['prépare une soupe chaude','range les assiettes propres','vérifie la cuisson du pain','partage une recette simple','nettoie la grande table','choisit des produits de saison']),
    (['le professeur','la professeure','le stagiaire','la stagiaire'], ['explique une méthode nouvelle','dessine un schéma clair','corrige les erreurs du tableau','présente les limites du modèle','vérifie les réponses du groupe','pose une question précise']),
    (['le musicien','la musicienne','le pianiste','la pianiste'], ['répète une mélodie douce','accorde les instruments anciens','écoute le rythme du morceau','prépare le prochain concert','étudie une partition difficile','joue une nouvelle chanson']),
    (['le voyageur','la voyageuse','le conducteur','la conductrice'], ['consulte les horaires du train','prépare le trajet du lendemain','observe les lumières du quai','attend le départ du bus','vérifie la direction du chemin','cherche une place tranquille']),
    (['le compteur','la sonde','le capteur','le dispositif'], ['enregistre une puissance moyenne','transmet une nouvelle mesure','indique une variation rapide','conserve les valeurs précédentes','signale une interruption temporaire','affiche une valeur différente']),
]
CONTEXTES = ['pendant la matinée','avant la pause','avec beaucoup de soin','dans le bâtiment du quartier','à la fin de la journée','selon le programme prévu','après une longue discussion','sans modifier les observations précédentes']


def creer():
    paths = [ROOT/'corpus.txt', ROOT/'corpus_bert.txt', ROOT/'corpus_origine.json']
    if all(p.exists() for p in paths):
        return json.loads(paths[2].read_text(encoding='utf-8'))
    rng = random.Random(42)
    phrases = [s.strip().replace('\n',' ') + '.' for s in re.split(r'\.\s+', RECIT.strip()) if len(s.split()) >= 6]
    for sujets, actions in THEMES:
        possibles = [f'{s} {a} {c}.' for s in sujets for a in actions for c in CONTEXTES]
        phrases.extend(rng.sample(possibles, 48))
    phrases = list(dict.fromkeys(phrases))
    rng.shuffle(phrases)
    texte = RECIT.strip() + '\n\nobservations du carnet\n\n' + '\n'.join(phrases[:160]) + '\n'
    paths[0].write_text(texte,encoding='utf-8',newline='\n')
    paths[1].write_text('\n'.join(phrases)+'\n',encoding='utf-8',newline='\n')
    origine = {'origine':'Textes originaux rédigés et assemblés pour ce TP pour Hachoum Mouad avec assistance de Codex. Aucun extrait externe.',
        'graine_assemblage':42,
        'gpt_caracteres':len(texte), 'bert_phrases':len(phrases),
        'limites':'Domaines restreints ; récit et phrases semi-synthétiques avec patrons récurrents. Les modèles ne prouvent aucune compréhension générale.',
        'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths[:2]}}
    paths[2].write_text(json.dumps(origine,indent=2,ensure_ascii=False),encoding='utf-8')
    return origine


if __name__=='__main__':
    print(json.dumps(creer(),indent=2,ensure_ascii=False))
