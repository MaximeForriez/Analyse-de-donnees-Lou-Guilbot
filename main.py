#coding:utf8

import pandas as pd
import math
import scipy
import scipy.stats

#C'est la partie la plus importante dans l'analyse de données. D'une part, elle n'est pas simple à comprendre tant mathématiquement que pratiquement. D'autre, elle constitue une application des probabilités. L'idée consiste à comparer une distribution de probabilité (théorique) avec des observations concrètes. De fait, il faut bien connaître les distributions vues dans la séance précédente afin de bien pratiquer cette comparaison. Les probabilités permettent de définir une probabilité critique à partir de laquelle les résultats ne sont pas conformes à la théorie probabiliste.
#Il n'est pas facile de proposer des analyses de données uniquement dans un cadre univarié. Vous utiliserez la statistique inférentielle principalement dans le cadre d'analyses multivariées. La statistique univariée est une statistique descriptive. Bien que les tests y soient possibles, comprendre leur intérêt et leur puissance d'analyse dans un tel cadre peut être déroutant.
#Peu importe dans quelle théorie vous êtes, l'idée de la statistique inférentielle est de vérifier si ce que vous avez trouvé par une méthode de calcul est intelligent ou stupide. Est-ce que l'on peut valider le résultat obtenu ou est-ce que l'incertitude qu'il présente ne permet pas de conclure ? Peu importe également l'outil, à chaque mesure statistique, on vous proposera un test pour vous aider à prendre une décision sur vos résultats. Il faut juste être capable de le lire.

#Par convention, on place les fonctions locales au début du code après les bibliothèques.
def ouvrirUnFichier(nom):
    with open(nom, "r") as fichier:
        contenu = pd.read_csv(fichier)
    return contenu

#Théorie de l'échantillonnage (intervalles de fluctuation)
#L'échantillonnage se base sur la répétitivité.
print("Résultat sur le calcul d'un intervalle de fluctuation")

donnees = pd.DataFrame(ouvrirUnFichier("./data/Echantillonnage-100-Echantillons.csv"))

# Calcul des moyennes par colonne et arrondi à 0 décimale avec la fonction native round()
moyennes = donnees.mean(axis=0)
moyennes_arrondies = moyennes.apply(lambda x: round(x, 0))

print("Moyennes par colonne (arrondies, 0 décimale) :")
for col, val in moyennes_arrondies.items():
    print(f"{col} : {val}")

# Calcul des fréquences de l'échantillon : somme des 3 moyennes puis normalisation
somme_moyennes = moyennes_arrondies.sum()
freq_echantillon = (moyennes_arrondies / somme_moyennes).round(2)

print("\nFréquences de l'échantillon (arrondies, 2 décimales) :")
for col, val in freq_echantillon.items():
    print(f"{col} : {val}")

# Calcul des fréquences de la population mère à partir des totaux par colonne (même principe)
totaux_population = donnees.sum(axis=0)
freq_population = (totaux_population / totaux_population.sum()).round(2)

print("\nFréquences de la population mère (arrondies, 2 décimales) :")
for col, val in freq_population.items():
    print(f"{col} : {val}")

# Intervalle de fluctuation à 95% (zC = 1.96)
z = 1.96
# effectif d'un échantillon estimé par la somme des moyennes (arrondies précédemment)
n = int(somme_moyennes)

# fréquences exactes sans arrondi pour le calcul statistique
freq_population_exact = totaux_population / totaux_population.sum()
freq_echantillon_exact = moyennes / moyennes.sum()

print("\nIntervalle de fluctuation à 95% pour chaque catégorie (arrondis 2 décimales) :")
for col in freq_population_exact.index:
    p = float(freq_population_exact[col])
    se = math.sqrt(p * (1 - p) / n)
    lo = p - z * se
    hi = p + z * se
    lo_r = round(lo, 2)
    hi_r = round(hi, 2)
    sample_p = float(freq_echantillon_exact[col])
    sample_p_r = round(sample_p, 2)
    inside = (sample_p >= lo) and (sample_p <= hi)
    print(f"{col} : [{lo_r} ; {hi_r}]  fréquence échantillon = {sample_p_r}  -> dans intervalle : {inside}")

#Théorie de l'estimation (intervalles de confiance)
#L'estimation se base sur l'effectif.
print("Résultat sur le calcul d'un intervalle de confiance")

#Théorie de la décision (tests d'hypothèse)
#La décision se base sur la notion de risques alpha et bêta.
#Comme à la séance précédente, l'ensemble des tests se trouve au lien : https://docs.scipy.org/doc/scipy/reference/stats.html
print("Théorie de la décision")

# Prendre le premier échantillon (première ligne) et le convertir en list()
premier_echantillon = list(donnees.iloc[0])
# Somme de la ligne (effectif total de l'échantillon isolé)
total_premier = sum(premier_echantillon)

print("\nPremier échantillon (ligne 0) :", premier_echantillon)
print("Somme du premier échantillon :", total_premier)

# Calcul des fréquences de ce premier échantillon et affichage (arrondies à 2 décimales)
freq_premier = [round(x / total_premier, 2) for x in premier_echantillon]

print("\nFréquences du premier échantillon (arrondies, 2 décimales) :")
for col, f in zip(donnees.columns, freq_premier):
    print(f"{col} : {f}")

# Intervalle de confiance (dépend uniquement de la taille de l'échantillon = total_premier)
z = 1.96
n = total_premier
freq_premier_exact = [x / n for x in premier_echantillon]

print("\nIntervalle de confiance 95% pour chaque opinion (arrondis, 2 décimales) :")
for col, p_hat in zip(donnees.columns, freq_premier_exact):
    se = math.sqrt(p_hat * (1 - p_hat) / n)
    lo = max(0.0, p_hat - z * se)
    hi = min(1.0, p_hat + z * se)
    print(f"{col} : [{round(lo,2)} ; {round(hi,2)}]  p_hat = {round(p_hat,2)}")

def tester_shapiro(nom_fichier):
    df = ouvrirUnFichier(nom_fichier)
    print(f"\nTest de Shapiro-Wilk pour {nom_fichier} :")
    # sélectionner les colonnes numériques
    numeric_cols = df.select_dtypes(include=['number']).columns if hasattr(df, "select_dtypes") else df.columns
    for col in numeric_cols:
        serie = df[col].dropna().values
        if serie.size < 3:
            print(f"  {col} : taille insuffisante pour Shapiro (n={serie.size})")
            continue
        stat, p = scipy.stats.shapiro(serie)
        verdict = "ne rejette pas H0 (distribution normale plausible)" if p > 0.05 else "rejette H0 (pas normale)"
        print(f"  {col} : W={stat:.4f}, p={p:.4f} -> {verdict}")

# Exécuter les tests sur les deux fichiers fournis
tester_shapiro("./data/Loi-normale-Test-1.csv")
tester_shapiro("./data/Loi-normale-Test-2.csv")
