
import psycopg2         
import nltk              
import numpy            
from nltk.stem.snowball import FrenchStemmer  
from nltk import word_tokenize                

nltk.download('punkt',      quiet=True)
nltk.download('stopwords',  quiet=True)
nltk.download('punkt_tab',  quiet=True)

from nltk.corpus import stopwords  


DB_CONFIG = {
    "dbname":   "recommandation", 
    "user":     "postgres",         
    "password": "1234",             
    "host":     "localhost",
    "port":     "5432",
}


def creer_base_et_inserer_produits():

    produits = [
        (1,  "Smartphone Samsung",   "Téléphone intelligent avec écran AMOLED haute résolution et batterie longue durée"),
        (2,  "Laptop HP",            "Ordinateur portable léger avec processeur rapide et mémoire vive importante"),
        (3,  "Tablette iPad",        "Tablette tactile avec écran Retina et processeur puissant pour applications mobiles"),
        (4,  "Casque Sony",          "Casque audio sans fil avec réduction de bruit active et son haute fidélité"),
        (5,  "Montre connectée",     "Montre intelligente avec suivi de santé, GPS et notifications téléphone"),
        (6,  "Imprimante Canon",     "Imprimante jet d'encre couleur rapide pour documents et photos haute qualité"),
        (7,  "Clavier Logitech",     "Clavier sans fil ergonomique avec touches silencieuses et longue autonomie"),
        (8,  "Souris optique",       "Souris sans fil précise avec molette silencieuse et design ergonomique"),
        (9,  "Disque dur externe",   "Stockage portable haute capacité pour sauvegarder fichiers photos et vidéos"),
        (10, "Clé USB 128Go",        "Clé USB rapide haute capacité pour transporter et partager vos fichiers"),
        (11, "Écran 27 pouces",      "Moniteur haute définition avec dalle IPS et résolution 4K pour créatifs"),
        (12, "Webcam HD",            "Caméra web haute définition pour visioconférence et streaming professionnel"),
        (13, "Enceinte Bluetooth",   "Enceinte portable sans fil avec son puissant et basses profondes"),
        (14, "Chargeur rapide",      "Chargeur universel rapide compatible avec téléphone tablette et ordinateur"),
        (15, "Batterie externe",     "Batterie portable haute capacité pour recharger téléphone et tablette en déplacement"),
        (16, "Hub USB-C",            "Concentrateur USB-C multiport pour connecter écran disque dur et périphériques"),
        (17, "Lampe de bureau LED",  "Lampe LED intelligente avec luminosité réglable et temperature de couleur variable"),
        (18, "Support laptop",       "Support ergonomique réglable pour ordinateur portable améliore posture bureau"),
        (19, "Tapis de souris XL",   "Grand tapis de bureau avec surface lisse optimisée pour souris optique précise"),
        (20, "Câble HDMI 4K",        "Câble HDMI haute vitesse pour connecter écran television ordinateur en 4K"),
    ]

    conn = psycopg2.connect(**DB_CONFIG)
    cur  = conn.cursor()

    cur.execute("DELETE FROM Produit;")

    for p in produits:
        cur.execute(
            "INSERT INTO Produit (id, nomPdt, description) VALUES (%s, %s, %s);",
            p
        )

    conn.commit()   
    cur.close()
    conn.close()
    print(f" Table créée et {len(produits)} produits insérés avec succès.\n")


def recuperer_produits():
    """
    Se connecte à la base et retourne tous les produits
    sous forme de liste de tuples (id, nomPdt, description).
    """
    conn = psycopg2.connect(**DB_CONFIG)
    cur  = conn.cursor()

    cur.execute("SELECT id, nomPdt, description FROM Produit ORDER BY id;")
    produits = cur.fetchall()   # Liste de tuples

    cur.close()
    conn.close()

    print(f"✅ {len(produits)} produits récupérés depuis la base de données.\n")
    return produits


def traiter_nlp(produits):
    """
    Pour chaque produit :
      1. Tokenize la description (découpe en mots)
      2. Applique le stemming (racinisation en français)
      3. Supprime les stop words et la ponctuation

    Retourne :
      - dict_produits  : {id_produit: [liste de mots traités]}
      - totalite_mots  : ensemble de TOUS les mots (vocabulaire global)
    """
    stop_list = set(stopwords.words('french'))
    stop_list.update([',', '.', ';', ':', '!', '?', "'", '"', '-', '(', ')'])

    stemmer       = FrenchStemmer()   
    dict_produits = {}               
    totalite_mots = set()            

    print(" Traitement NLP des descriptions :")
    print("-" * 50)

    for p in produits:
        id_pdt      = int(p[0])
        description = p[2]

        mots = word_tokenize(description, language='french')

        mots_stems = [stemmer.stem(m) for m in mots]

        mots_filtres = [
            m for m in mots_stems
            if m not in stop_list and m.isalpha()
        ]

        dict_produits[id_pdt] = mots_filtres

        for m in mots_filtres:
            totalite_mots.add(m)

        print(f"  Produit {id_pdt:02d} | {p[1]:<25} → {mots_filtres}")

    print()
    return dict_produits, totalite_mots


def construire_matrice_binaire(dict_produits, totalite_mots):
  
    nb_produits = len(dict_produits)
    liste_mots  = list(totalite_mots)   
    nb_mots     = len(liste_mots)

    matrice = numpy.zeros((nb_produits, nb_mots))

    for i, id_pdt in enumerate(dict_produits):
        for j, mot in enumerate(liste_mots):
            if mot in dict_produits[id_pdt]:
                matrice[i][j] = 1  

    print(f"Matrice binaire construite : {nb_produits} produits × {nb_mots} mots\n")
    return matrice, liste_mots


def similarite_cosinus(vecteur_a, vecteur_b):
   
    produit_scalaire = numpy.dot(vecteur_a, vecteur_b)
    norme_a = numpy.linalg.norm(vecteur_a)
    norme_b = numpy.linalg.norm(vecteur_b)

    if norme_a == 0 or norme_b == 0:
        return 0.0

    return produit_scalaire / (norme_a * norme_b)


def construire_matrice_similarite(matrice_binaire):
    """
    Calcule la similarité cosinus entre chaque paire de produits
    et retourne une matrice carrée (NbProduits x NbProduits).
    """
    nb_produits = matrice_binaire.shape[0]
    matrice_sim = numpy.zeros((nb_produits, nb_produits))

    for i in range(nb_produits):
        for j in range(nb_produits):
            matrice_sim[i][j] = similarite_cosinus(
                matrice_binaire[i], matrice_binaire[j]
            )

    print(" Matrice de similarité cosinus calculée.\n")
    return matrice_sim


def recommander(id_produit, matrice_similarite, dict_produits, produits, top_n=3):
    ids_produits = list(dict_produits.keys())

    if id_produit not in ids_produits:
        print(f" Produit {id_produit} introuvable.")
        return []

    index_ref = ids_produits.index(id_produit)

    scores = matrice_similarite[index_ref]

    indices_tries = numpy.argsort(scores)[::-1]
    recommandations = []
    for idx in indices_tries:
        if idx != index_ref:                    
            id_reco    = ids_produits[idx]
            nom_reco   = produits[idx][1]
            score_reco = scores[idx]
            recommandations.append((id_reco, nom_reco, score_reco))
        if len(recommandations) == top_n:
            break

    return recommandations


if __name__ == "__main__":

    print("=" * 60)
    print("   SYSTÈME DE RECOMMANDATION BASÉ SUR LE CONTENU")
    print("=" * 60)
    print()

    produits = recuperer_produits()

    dict_produits, totalite_mots = traiter_nlp(produits)

    print(f" Vocabulaire global : {len(totalite_mots)} mots uniques\n")

    matrice_binaire, liste_mots = construire_matrice_binaire(dict_produits, totalite_mots)

    matrice_similarite = construire_matrice_similarite(matrice_binaire)

    print("=" * 60)
    print("   RECOMMANDATIONS")
    print("=" * 60)

    produits_test = [1, 5, 9]   
    TOP_N = 3                    

    for id_test in produits_test:
        nom_ref = next((p[1] for p in produits if int(p[0]) == id_test), "?")

        print(f"\n Produit de référence : [{id_test}] {nom_ref}")
        print(f"   Top {TOP_N} produits similaires :")

        recommandations = recommander(
            id_produit         = id_test,
            matrice_similarite = matrice_similarite,
            dict_produits      = dict_produits,
            produits           = produits,
            top_n              = TOP_N
        )

        for rang, (id_r, nom_r, score_r) in enumerate(recommandations, start=1):
            barre = "█" * int(score_r * 20)  
            print(f"   {rang}. [{id_r:02d}] {nom_r:<25}  Score : {score_r:.4f}  {barre}")

    print("\n" + "=" * 60)
    print("Système de recommandation terminé avec succès !")
    print("=" * 60)