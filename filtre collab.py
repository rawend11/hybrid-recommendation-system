
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import psycopg2
from psycopg2.extras import RealDictCursor

DB_CONFIG = {
    "host":     "localhost",
    "database": "recommandation",
    "user":     "postgres",
    "password": "1234",
    "port":     5432
}

def get_connection():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print(" Connexion PostgreSQL réussie.")
        return conn
    except psycopg2.Error as e:
        print(f" Erreur de connexion : {e}")
        return None


def get_ratings_matrix():
    conn = get_connection()
    if conn is None:
        return None
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT u.nomuser, p.nompdt, n.note
            FROM notes n
            JOIN utilisateurs u ON n.iduser    = u.iduser
            JOIN produit      p ON n.idproduit = p.id
        """)
        rows = cursor.fetchall()
        if not rows:
            print(" Aucune note trouvée dans la base.")
            return None
        df     = pd.DataFrame(rows)
        matrix = df.pivot_table(index='nomuser', columns='nompdt', values='note')
        print(f"\n Matrice des notes ({matrix.shape[0]} users × {matrix.shape[1]} produits) :")
        print(matrix.to_string())
        return matrix
    except Exception as e:
        print(f" Erreur SQL : {e}")
        return None
    finally:
        conn.close()

def compute_similarity_matrix(ratings_matrix):
    filled = ratings_matrix.fillna(0).values
    sim    = cosine_similarity(filled)
    sim_df = pd.DataFrame(sim,
                          index=ratings_matrix.index,
                          columns=ratings_matrix.index)
    print("\n Matrice de similarité cosinus entre utilisateurs :")
    print(sim_df.round(4).to_string())
    return sim_df


def get_neighbors(sim_df, target_user, n=2):
    neighbors = (sim_df[target_user]
                 .drop(target_user)
                 .sort_values(ascending=False)
                 .head(n))
    print(f"\n Voisins de '{target_user}' :")
    for user, sim in neighbors.items():
        print(f"   → {user} : similarité = {sim:.4f}")
    return neighbors


def predict_rating(ratings_matrix, sim_df, target_user, target_item, n=2):
    neighbors    = get_neighbors(sim_df, target_user, n)
    numerateur   = 0.0
    denominateur = 0.0
    for voisin, sim in neighbors.items():
        note = ratings_matrix.loc[voisin, target_item]
        if not np.isnan(note):
            numerateur   += sim * note
            denominateur += abs(sim)
    if denominateur == 0:
        return np.nan
    pred = round(numerateur / denominateur, 2)
    print(f" Note prédite de '{target_user}' pour '{target_item}' = {pred}")
    return pred

def fill_missing_ratings(ratings_matrix, sim_df, n=2):
    filled = ratings_matrix.copy()
    print("\n Prédiction des notes manquantes...")
    for user in ratings_matrix.index:
        for item in ratings_matrix.columns:
            if np.isnan(ratings_matrix.loc[user, item]):
                pred = predict_rating(ratings_matrix, sim_df, user, item, n)
                if not np.isnan(pred):
                    filled.loc[user, item] = pred
    print("\n Matrice complète après prédiction :")
    print(filled.round(2).to_string())
    return filled


def recommend(filled_matrix, original_matrix, target_user, n=3):
    non_notes = original_matrix.loc[target_user][
        original_matrix.loc[target_user].isna()
    ].index.tolist()
    if not non_notes:
        print(f"\n {target_user} a déjà noté tous les produits !")
        return pd.Series(dtype=float)
    top = (filled_matrix.loc[target_user, non_notes]
           .sort_values(ascending=False)
           .head(n))
    print(f"\n Top {n} recommandations pour '{target_user}' :")
    print("-" * 50)
    for rank, (produit, score) in enumerate(top.items(), 1):
        etoiles = "⭐" * round(score) if not np.isnan(score) else ""
        print(f"  {rank}. {produit:<25} Note prédite : {score:.2f}  {etoiles}")
        print("   Recommandé car des utilisateurs similaires ont aimé ce produit")
    print("-" * 50)
    return top


def main():
    print("=" * 60)
    print("   FILTRAGE COLLABORATIF")
    print("=" * 60)

    N_VOISINS = 4

    ratings = get_ratings_matrix()
    if ratings is None:
        print(" Impossible de continuer sans données.")
        return

    sim_df = compute_similarity_matrix(ratings)
    filled = fill_missing_ratings(ratings, sim_df, N_VOISINS)

    print("\n" + "=" * 60)
    print("   RECOMMANDATIONS FINALES")
    print("=" * 60)
    for user in ratings.index:
        recommend(filled, ratings, user, n=3)

    return filled, sim_df

if __name__ == "__main__":
    main()