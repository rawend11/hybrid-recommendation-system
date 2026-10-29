
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta
import random


st.set_page_config(
    page_title="Moteur de Recommandation",
    page_icon="🛒",
    layout="centered"
)


DB_CONFIG = {
    "host":     "localhost",
    "database": "recommandation",
    "user":     "postgres",
    "password": "1234",
    "port":     5432
}

@st.cache_resource
def get_connection():
    try:
        return psycopg2.connect(**DB_CONFIG)
    except:
        return None


@st.cache_data(ttl=60)
def recuperer_matrice_notes():
    conn = get_connection()
    if conn is None:
        st.error("❌ Connexion à la base impossible.")
        return pd.DataFrame()
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
            st.error("❌ Aucune note trouvée dans la base.")
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        return df.pivot_table(index='nomuser', columns='nompdt', values='note')
    except Exception as e:
        st.error(f"❌ Erreur SQL : {e}")
        return pd.DataFrame()


@st.cache_data(ttl=60)
def recuperer_produits():
    conn = get_connection()
    if conn is None:
        return pd.DataFrame()
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT id, nompdt, description FROM produit ORDER BY id")
        rows = cursor.fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"❌ Erreur SQL : {e}")
        return pd.DataFrame()


@st.cache_data(ttl=60)
def recuperer_notes_avec_dates():
    conn = get_connection()
    if conn is None:
        return pd.DataFrame()
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT u.nomuser, p.nompdt, n.note, n.datenote
            FROM notes n
            JOIN utilisateurs u ON n.iduser    = u.iduser
            JOIN produit      p ON n.idproduit = p.id
            ORDER BY n.datenote
        """)
        rows = cursor.fetchall()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df['datenote'] = pd.to_datetime(df['datenote'])
        if df['datenote'].nunique() <= 1:
            today = datetime.now()
            random.seed(42)
            df['datenote'] = [today - timedelta(days=random.randint(10, 180))
                              for _ in range(len(df))]
        return df.sort_values('datenote')
    except Exception as e:
        st.error(f"❌ Erreur SQL : {e}")
        return pd.DataFrame()



def calculer_similarite_utilisateurs(matrice_notes):
    remplie = matrice_notes.fillna(0).values
    sim     = cosine_similarity(remplie)
    return pd.DataFrame(sim, index=matrice_notes.index, columns=matrice_notes.index)


def predire_note(matrice_notes, matrice_sim, utilisateur, produit, n=2):
    voisins     = matrice_sim[utilisateur].drop(utilisateur).sort_values(ascending=False).head(n)
    numerateur  = 0.0
    denominateur = 0.0
    for voisin, sim in voisins.items():
        note = matrice_notes.loc[voisin, produit]
        if not np.isnan(note):
            numerateur   += sim * note
            denominateur += abs(sim)
    return round(numerateur / denominateur, 2) if denominateur > 0 else np.nan


def remplir_matrice(matrice_notes, matrice_sim, n=2):
    remplie = matrice_notes.copy()
    for utilisateur in matrice_notes.index:
        for produit in matrice_notes.columns:
            if np.isnan(matrice_notes.loc[utilisateur, produit]):
                pred = predire_note(matrice_notes, matrice_sim, utilisateur, produit, n)
                if not np.isnan(pred):
                    remplie.loc[utilisateur, produit] = pred
    return remplie


def recommander_collaboratif(matrice_remplie, matrice_originale, utilisateur, n=3):
    non_notes = matrice_originale.loc[utilisateur][
        matrice_originale.loc[utilisateur].isna()
    ].index.tolist()
    if not non_notes:
        return pd.Series(dtype=float)
    scores = matrice_remplie.loc[utilisateur, non_notes].dropna()
    return scores.sort_values(ascending=False).head(n)



def nettoyer_texte(texte):
    if not isinstance(texte, str):
        return ''
    stop_list = {
        'le','la','les','de','du','un','une','des','et','en','pour','avec',
        'sur','par','dans','est','au','aux','ce','qui','que','se','son','sa',
        'ses','il','elle','the','a','an','for','with','and','or','in','on'
    }
    return ' '.join([m for m in texte.lower().split() if m.isalpha() and m not in stop_list])


def calculer_similarite_produits(produits_df):
    df = produits_df.copy()
    df['description_nettoyee'] = df['description'].apply(nettoyer_texte)
    vectoriseur = TfidfVectorizer()
    matrice     = vectoriseur.fit_transform(df['description_nettoyee'])
    sim         = cosine_similarity(matrice)
    return pd.DataFrame(sim, index=df['nompdt'].tolist(), columns=df['nompdt'].tolist())


def recommander_par_contenu(matrice_sim_produits, nom_produit, n=3):
    if nom_produit not in matrice_sim_produits.index:
        return pd.Series(dtype=float)
    return matrice_sim_produits[nom_produit].drop(nom_produit).sort_values(ascending=False).head(n)



def calculer_poids_temporel(date, aujourd_hui, decay=30):
    if pd.isna(date):
        return 0.0
    return np.exp(-(aujourd_hui - date).days / decay)


def calculer_notes_ponderees(notes_df, decay=30):
    aujourd_hui = datetime.now()
    df = notes_df.copy()
    df['poids']         = df['datenote'].apply(lambda d: calculer_poids_temporel(d, aujourd_hui, decay))
    df['note_ponderee'] = df['note'] * df['poids']
    resultat = df.groupby(['nomuser', 'nompdt']).apply(
        lambda g: g['note_ponderee'].sum() / g['poids'].sum() if g['poids'].sum() > 0 else np.nan,
        include_groups=False
    ).reset_index()
    resultat.columns = ['nomuser', 'nompdt', 'note_ta']
    return resultat.pivot_table(index='nomuser', columns='nompdt', values='note_ta')


def recommander_time_aware(notes_df, utilisateur, matrice_originale, n=3, decay=30):
    matrice_ta = calculer_notes_ponderees(notes_df, decay)
    if utilisateur not in matrice_ta.index or utilisateur not in matrice_originale.index:
        return pd.Series(dtype=float)
    sim_ta    = calculer_similarite_utilisateurs(matrice_ta)
    non_notes = matrice_originale.loc[utilisateur][matrice_originale.loc[utilisateur].isna()].index.tolist()
    scores    = {}
    for produit in non_notes:
        if produit in matrice_ta.columns:
            pred = predire_note(matrice_ta, sim_ta, utilisateur, produit, n=2)
            if not np.isnan(pred):
                scores[produit] = pred
    if not scores:
        return pd.Series(dtype=float)
    return pd.Series(scores).sort_values(ascending=False).head(n)


matrice_notes = recuperer_matrice_notes()
produits_df   = recuperer_produits()
notes_df      = recuperer_notes_avec_dates()

if matrice_notes.empty or produits_df.empty:
    st.error(" Impossible de charger les données depuis la base. Vérifiez votre connexion.")
    st.stop()

matrice_sim_utilisateurs = calculer_similarite_utilisateurs(matrice_notes)
matrice_remplie_base     = remplir_matrice(matrice_notes, matrice_sim_utilisateurs, 2)
matrice_sim_produits     = calculer_similarite_produits(produits_df)


st.title(" Moteur de Recommandation")
st.markdown("---")
st.success("🟢 Connecté à la base de données")

col1, col2, col3 = st.columns(3)
col1.metric("Utilisateurs",       len(matrice_notes))
col2.metric("Produits",           len(produits_df))
col3.metric("Notes enregistrées", len(notes_df))

st.markdown("---")


tab1, tab2, tab3, tab4 = st.tabs([
    " Filtrage Collaboratif",
    " Basé sur le Contenu",
    " Time-Aware",
    " Hybride"
])


with tab1:
    st.subheader(" Filtrage Collaboratif")
    st.info("Trouve les utilisateurs aux goûts similaires et prédit les notes manquantes.\n\n**Formule :** r* = Σ(sim × r) / Σ|sim|")

    utilisateur_cf = st.selectbox("Choisissez un utilisateur", matrice_notes.index.tolist(), key="u1")
    nb_voisins     = st.slider("Nombre de voisins (N)", 1, min(4, len(matrice_notes)-1), 2, key="n1")
    nb_recs        = st.slider("Nombre de recommandations (Top K)", 1, 5, 3, key="k1")

    if st.button(" Voir les recommandations", key="btn1"):
        sim_cf    = calculer_similarite_utilisateurs(matrice_notes)
        mat_cf    = remplir_matrice(matrice_notes, sim_cf, nb_voisins)
        recs_cf   = recommander_collaboratif(mat_cf, matrice_notes, utilisateur_cf, nb_recs)

        st.markdown(f"#### Voisins les plus proches de **{utilisateur_cf}** :")
        voisins = sim_cf[utilisateur_cf].drop(utilisateur_cf).sort_values(ascending=False).head(nb_voisins)
        for voisin, sim in voisins.items():
            st.write(f"→ **{voisin}** : similarité = `{sim:.4f}`")

        st.markdown(f"####  Top {nb_recs} recommandations pour **{utilisateur_cf}** :")
        if len(recs_cf) == 0:
            st.success(f"{utilisateur_cf} a déjà noté tous les produits !")
        else:
            for rang, (produit, score) in enumerate(recs_cf.items(), 1):
                st.markdown(f"**{rang}. {produit}**")
                st.write(f"   Note prédite : `{score}`")
                st.caption("   → Recommandé car des utilisateurs similaires ont aimé ce produit")


with tab2:
    st.subheader(" Filtrage Basé sur le Contenu")
    st.info("Analyse les descriptions des produits (TF-IDF) et recommande les produits au contenu le plus similaire.")

    produit_cb = st.selectbox("Choisissez un produit consulté", produits_df['nompdt'].tolist(), key="p2")
    nb_recs_cb = st.slider("Nombre de recommandations (Top K)", 1, 5, 3, key="k2")

    description = produits_df[produits_df['nompdt'] == produit_cb]['description'].values[0]
    st.write(f"**Description :** {description}")

    if st.button(" Voir les recommandations", key="btn2"):
        recs_cb = recommander_par_contenu(matrice_sim_produits, produit_cb, nb_recs_cb)
        st.markdown(f"####  Top {nb_recs_cb} produits similaires à **{produit_cb}** :")
        for rang, (produit, score) in enumerate(recs_cb.items(), 1):
            st.markdown(f"**{rang}. {produit}**")
            st.write(f"   Similarité cosinus : `{score:.4f}`")
            st.caption("   → Recommandé car sa description contient des mots similaires")


with tab3:
    st.subheader(" Recommandation Time-Aware")
    st.info("Les notes récentes ont plus de poids grâce à une décroissance exponentielle.\n\n**Formule :** w(t) = exp(−jours / decay) | note_pondérée = note × w(t)")

    utilisateur_ta = st.selectbox("Choisissez un utilisateur", matrice_notes.index.tolist(), key="u3")
    decay          = st.slider("Demi-vie temporelle (jours)", 7, 90, 30, key="d3")
    nb_recs_ta     = st.slider("Nombre de recommandations (Top K)", 1, 5, 3, key="k3")

    if st.button(" Voir les recommandations", key="btn3"):
        aujourd_hui = datetime.now()
        historique  = notes_df[notes_df['nomuser'] == utilisateur_ta].sort_values('datenote')

        st.markdown(f"####  Historique de **{utilisateur_ta}** :")
        for _, ligne in historique.iterrows():
            jours_ecoules = (aujourd_hui - ligne['datenote']).days
            poids = calculer_poids_temporel(ligne['datenote'], aujourd_hui, decay)
            st.write(f"→ **{ligne['nompdt']}** | Note: `{ligne['note']}` | Il y a {jours_ecoules} jours | Poids: `{poids:.3f}`")

        recs_ta = recommander_time_aware(notes_df, utilisateur_ta, matrice_notes, nb_recs_ta, decay)
        st.markdown(f"####  Top {nb_recs_ta} recommandations Time-Aware pour **{utilisateur_ta}** :")
        if len(recs_ta) == 0:
            st.info("Pas assez de données pour générer des recommandations Time-Aware.")
        else:
            for rang, (produit, score) in enumerate(recs_ta.items(), 1):
                st.markdown(f"**{rang}. {produit}**")
                st.write(f"   Score pondéré : `{score:.2f}`")
                st.caption("   → Recommandé car récemment populaire parmi les utilisateurs similaires")


with tab4:
    st.subheader("Recommandation Hybride")
    st.info("Combine les 3 méthodes avec une moyenne pondérée.\n\n**Score = w_CF × score_CF + w_CB × score_CB + w_TA × score_TA**")

    utilisateur_h = st.selectbox("Choisissez un utilisateur", matrice_notes.index.tolist(), key="u4")
    nb_recs_h     = st.slider("Nombre de recommandations (Top K)", 1, 5, 3, key="k4")

    col1, col2, col3 = st.columns(3)
    with col1: poids_cf = st.slider("Poids Collaboratif", 0.0, 1.0, 0.4, 0.1)
    with col2: poids_cb = st.slider("Poids Contenu",      0.0, 1.0, 0.3, 0.1)
    with col3: poids_ta = st.slider("Poids Time-Aware",   0.0, 1.0, 0.3, 0.1)

    total_poids = poids_cf + poids_cb + poids_ta
    if total_poids == 0:
        st.error("La somme des poids ne peut pas être 0 !")
    else:
        poids_cf_n = poids_cf / total_poids
        poids_cb_n = poids_cb / total_poids
        poids_ta_n = poids_ta / total_poids
        st.write(f"Poids normalisés → CF: `{poids_cf_n:.2f}` | CB: `{poids_cb_n:.2f}` | TA: `{poids_ta_n:.2f}`")

    if st.button(" Voir les recommandations", key="btn4") and total_poids > 0:

        sim_h       = calculer_similarite_utilisateurs(matrice_notes)
        mat_h       = remplir_matrice(matrice_notes, sim_h, 2)
        scores_cf   = {}
        for produit in matrice_notes.columns:
            if np.isnan(matrice_notes.loc[utilisateur_h, produit]):
                pred = mat_h.loc[utilisateur_h, produit]
                if not np.isnan(pred):
                    scores_cf[produit] = pred / 5.0

        produits_notes = matrice_notes.loc[utilisateur_h].dropna().index.tolist()
        scores_cb      = {}
        for produit in matrice_notes.columns:
            if np.isnan(matrice_notes.loc[utilisateur_h, produit]) and produit in matrice_sim_produits.columns:
                valides = [p for p in produits_notes if p in matrice_sim_produits.index]
                if valides:
                    scores_cb[produit] = float(np.mean([matrice_sim_produits.loc[p, produit] for p in valides]))

        recs_ta_h  = recommander_time_aware(notes_df, utilisateur_h, matrice_notes, len(matrice_notes.columns), 30)
        max_ta     = recs_ta_h.max() if len(recs_ta_h) > 0 else 1
        scores_ta  = {produit: score / max_ta for produit, score in recs_ta_h.items()} if max_ta > 0 else {}

        tous_produits = set(scores_cf) | set(scores_cb) | set(scores_ta)
        scores_hybrides = {
            produit: poids_cf_n * scores_cf.get(produit, 0)
                   + poids_cb_n * scores_cb.get(produit, 0)
                   + poids_ta_n * scores_ta.get(produit, 0)
            for produit in tous_produits
        }

        st.markdown(f"####  Top {nb_recs_h} recommandations hybrides pour **{utilisateur_h}** :")
        if not scores_hybrides:
            st.success(f"{utilisateur_h} a déjà noté tous les produits !")
        else:
            top_hybride = sorted(scores_hybrides.items(), key=lambda x: x[1], reverse=True)[:nb_recs_h]
            for rang, (produit, score) in enumerate(top_hybride, 1):
                st.markdown(f"**{rang}. {produit}**")
                st.write(f"   Score hybride : `{score:.3f}`")
                st.caption(
                    f"   CF: {scores_cf.get(produit,0):.2f} · "
                    f"CB: {scores_cb.get(produit,0):.2f} · "
                    f"TA: {scores_ta.get(produit,0):.2f}"
                )
                st.caption("   → Recommandé en combinant vos préférences, le contenu et les tendances récentes")            
st.markdown("---")