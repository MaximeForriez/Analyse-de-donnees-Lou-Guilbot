#coding:utf8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import math

#Fonction pour ouvrir les fichiers
def ouvrirUnFichier(nom):
    with open(nom, "r", encoding="utf-8") as fichier:
        contenu = pd.read_csv(fichier)
    return contenu

#Fonction pour convertir les données en données logarithmiques
def conversionLog(liste):
    log = []
    for element in liste:
        log.append(math.log(element))
    return log

#Fonction pour trier par ordre décroissant les listes (îles et populations)
def ordreDecroissant(liste):
    liste.sort(reverse = True)
    return liste

#Fonction pour obtenir le classement des listes spécifiques aux populations
def ordrePopulation(pop, etat):
    ordrepop = []
    for element in range(0, len(pop)):
        if np.isnan(pop[element]) == False:
            ordrepop.append([float(pop[element]), etat[element]])
    ordrepop = ordreDecroissant(ordrepop)
    for element in range(0, len(ordrepop)):
        ordrepop[element] = [element + 1, ordrepop[element][1]]
    return ordrepop

#Fonction pour obtenir l'ordre défini entre deux classements (listes spécifiques aux populations)
def classementPays(ordre1, ordre2):
    classement = []
    if len(ordre1) <= len(ordre2):
        for element1 in range(0, len(ordre2) - 1):
            for element2 in range(0, len(ordre1) - 1):
                if ordre2[element1][1] == ordre1[element2][1]:
                    classement.append([ordre1[element2][0], ordre2[element1][0], ordre1[element2][1]])
    else:
        for element1 in range(0, len(ordre1) - 1):
            for element2 in range(0, len(ordre2) - 1):
                if ordre2[element2][1] == ordre1[element1][1]:
                    classement.append([ordre1[element1][0], ordre2[element2][0], ordre1[element][1]])
    return classement

#Partie sur les îles
iles = pd.DataFrame(ouvrirUnFichier("./data/island-index.csv"))

#Attention ! Il va falloir utiliser des fonctions natives de Python dans les fonctions locales que je vous propose pour faire l'exercice. Vous devez caster l'objet Pandas en list().

# --- Isolation de la colonne Surface (km2) et conversion en floats ---
col_candidates = ["Surface (km²)", "Surface (km2)", "Surface (km)"]
surface_series = None
for c in col_candidates:
    if c in iles.columns:
        surface_series = iles[c]
        break
if surface_series is None:
    raise KeyError("Colonne 'Surface (km2)' introuvable dans le fichier CSV.")

surface_list = []
for v in surface_series.tolist():
    try:
        if pd.isna(v):
            surface_list.append(float('nan'))
        else:
            s = str(v)
            s = s.replace('\u00b2', '')
            s = s.replace('km', '')
            s = s.replace(' ', '')
            surface_list.append(float(s))
    except Exception:
        try:
            surface_list.append(float(v))
        except Exception:
            surface_list.append(float('nan'))

# Ajouter les surfaces des continents (sans unité), en float
# Asie / Afrique / Europe : 85 545 323 km2
# Amérique : 37 856 841 km2
# Antarctique : 7 768 030 km2
# Australie : 7 605 049 km2
continent_surfaces = [85545323.0, 37856841.0, 7768030.0, 7605049.0]
surface_list.extend(continent_surfaces)

print('Extrait: {} valeurs (dont continents ajoutés)'.format(len(surface_list)))

# --- Trier, filtrer les valeurs non valides et préparer la loi rang-taille ---
# On enlève les NaN et les valeurs non strictement positives
valid_sizes = [v for v in surface_list if (not math.isnan(v)) and (v > 0)]

# Ordonner en décroissant
valid_sizes = ordreDecroissant(valid_sizes)

# Rangs (1..n)
ranks = list(range(1, len(valid_sizes) + 1))

# Conversion logarithmique des axes (utilise conversionLog)
log_ranks = conversionLog(ranks)
log_sizes = conversionLog(valid_sizes)

# Tracer la loi rang-taille en échelle logarithmique (axes déjà en log)
plt.figure(figsize=(8, 6))
plt.plot(log_ranks, log_sizes, marker='o', linestyle='none', markersize=3)
plt.xlabel('log(Rang)')
plt.ylabel('log(Surface (km2))')
plt.title('Loi rang-taille — Îles + continents')
output_path = './data/rank_size.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print('Image sauvegardée : {}'.format(output_path))








#Partie sur les populations des États du monde
# Réponse (test sur les rangs) :
# Il est possible de faire un test sur les rangs. Par exemple on peut
# tester la relation rang-taille en ajustant une régression linéaire
# sur les séries transformées en logarithme (log(rang) vs log(taille)).
# On peut utiliser la corrélation de Pearson ou un test de pente
# (vérifier si la pente est significativement différente de 0).
# (Cette note est fournie ici sous forme de commentaire demandé.)
#Source. Depuis 2007, tous les ans jusque 2025, M. Forriez a relevé l'intégralité du nombre d'habitants dans chaque États du monde proposé par un numéro hors-série du monde intitulé États du monde. Vous avez l'évolution de la population et de la densité par année.
# Remarque : le fichier se trouve dans le dossier parent `data/` du répertoire `src`.
monde = pd.DataFrame(ouvrirUnFichier(r"C:\Python\seance_6\data\Le-Monde-HS-Etats-du-monde-2007-2025.csv"))

# Test sur les rangs :
# Il est possible de tester les rangs — par exemple on peut vérifier que la
# liste des rangs est strictement croissante et qu'il n'y a pas de doublons.
# Un test simple consiste à vérifier `len(ranks) == len(set(ranks))` et
# `ranks == sorted(ranks)`. Cela garantit l'unicité et l'ordre des rangs.

# Attention ! Il va falloir utiliser des fonctions natives de Python dans les fonctions locales que je vous propose pour faire l'exercice. Vous devez caster l'objet Pandas en list().

# --- Isolation des colonnes demandées dans le fichier 'Le-Monde' ---
cols_demandees = ["État", "Pop 2007", "Pop 2025", "Densité 2007", "Densité 2025"]
missing = [c for c in cols_demandees if c not in monde.columns]
if missing:
    raise KeyError(f"Colonnes manquantes dans le fichier Le-Monde: {missing}")

monde_selection = monde[cols_demandees].copy()
# Forcer les types numériques en float pour les colonnes de population et densité
for col in ["Pop 2007", "Pop 2025", "Densité 2007", "Densité 2025"]:
    monde_selection[col] = pd.to_numeric(monde_selection[col], errors='coerce').astype(float)

print('Extrait Le-Monde :', monde_selection.shape[0], 'lignes, colonnes:', list(monde_selection.columns))

# --- Ordonner les listes demandées avec la fonction locale `ordrePopulation()` ---
etats = monde_selection['État'].tolist()
pop2007_list = monde_selection['Pop 2007'].tolist()
pop2025_list = monde_selection['Pop 2025'].tolist()
dens2007_list = monde_selection['Densité 2007'].tolist()
dens2025_list = monde_selection['Densité 2025'].tolist()

ordre_pop2007 = ordrePopulation(pop2007_list, etats)
ordre_pop2025 = ordrePopulation(pop2025_list, etats)
ordre_dens2007 = ordrePopulation(dens2007_list, etats)
ordre_dens2025 = ordrePopulation(dens2025_list, etats)

print('\nTop 10 Pop 2007 :', ordre_pop2007[:10])
print('Top 10 Pop 2025 :', ordre_pop2025[:10])
print('Top 10 Densité 2007 :', ordre_dens2007[:10])
print('Top 10 Densité 2025 :', ordre_dens2025[:10])

# --- Comparaison des classements via `classementPays()` ---
# Comparer population 2007 vs 2025
cmp_pop = classementPays(ordre_pop2007, ordre_pop2025)
cmp_pop.sort()
# cmp_pop entries: [rank_in_2007, rank_in_2025, 'État']
pop_rank_2007 = [c[0] for c in cmp_pop]
pop_rank_2025 = [c[1] for c in cmp_pop]

# Comparer densité 2007 vs 2025
cmp_dens = classementPays(ordre_dens2007, ordre_dens2025)
cmp_dens.sort()
# cmp_dens entries: [rank_in_2007, rank_in_2025, 'État']
dens_rank_2007 = [c[0] for c in cmp_dens]
dens_rank_2025 = [c[1] for c in cmp_dens]

print('\nComparaison population (extrait 10 premières lignes, ordonnées par rang 2007):')
for row in cmp_pop[:10]:
    print(row)

print('\nComparaison densité (extrait 10 premières lignes, ordonnées par rang 2007):')
for row in cmp_dens[:10]:
    print(row)

# --- Isoler les deux colonnes sous la forme de listes différentes en utilisant une boucle ---
# Pour la population (rangs 2007 et 2025)
pop_col_2007 = []
pop_col_2025 = []
for entry in cmp_pop:
    pop_col_2007.append(entry[0])
    pop_col_2025.append(entry[1])

print('\nExtrait listes population (longueurs):', len(pop_col_2007), len(pop_col_2025))
print('Premier éléments (pop 2007, pop 2025):', pop_col_2007[:5], pop_col_2025[:5])

# Pour la densité (rangs 2007 et 2025)
dens_col_2007 = []
dens_col_2025 = []
for entry in cmp_dens:
    dens_col_2007.append(entry[0])
    dens_col_2025.append(entry[1])

print('\nExtrait listes densité (longueurs):', len(dens_col_2007), len(dens_col_2025))
print('Premier éléments (dens 2007, dens 2025):', dens_col_2007[:5], dens_col_2025[:5])

# --- Calcul des corrélations de rangs (Spearman) et concordance (Kendall) ---
# On construit des dictionnaires state -> rang pour chaque métrique/année
pop2007_dict = {item[1]: item[0] for item in ordre_pop2007}
pop2025_dict = {item[1]: item[0] for item in ordre_pop2025}
dens2007_dict = {item[1]: item[0] for item in ordre_dens2007}
dens2025_dict = {item[1]: item[0] for item in ordre_dens2025}

# États communs entre les quatre classements
common_states = set(pop2007_dict) & set(pop2025_dict) & set(dens2007_dict) & set(dens2025_dict)
if not common_states:
    raise ValueError('Aucun État commun trouvé entre les classements.')

# Ordonnons les états par rang population 2007 pour stabilité
states_sorted = sorted(common_states, key=lambda s: pop2007_dict.get(s, float('inf')))

pop_ranks_2007 = [pop2007_dict[s] for s in states_sorted]
dens_ranks_2007 = [dens2007_dict[s] for s in states_sorted]

pop_ranks_2025 = [pop2025_dict[s] for s in states_sorted]
dens_ranks_2025 = [dens2025_dict[s] for s in states_sorted]

# Calculs via scipy.stats
spearman_2007 = scipy.stats.spearmanr(pop_ranks_2007, dens_ranks_2007)
kendall_2007 = scipy.stats.kendalltau(pop_ranks_2007, dens_ranks_2007)

spearman_2025 = scipy.stats.spearmanr(pop_ranks_2025, dens_ranks_2025)
kendall_2025 = scipy.stats.kendalltau(pop_ranks_2025, dens_ranks_2025)

print('\nCorrélations de rangs (Population vs Densité) — 2007:')
print('Spearman:', spearman_2007)
print('Kendall:', kendall_2007)

print('\nCorrélations de rangs (Population vs Densité) — 2025:')
print('Spearman:', spearman_2025)
print('Kendall:', kendall_2025)


# --- Isolation des colonnes demandées et conversion en listes / floats ---
cols_wanted = ["État", "Pop 2007", "Pop 2025", "Densité 2007", "Densité 2025"]
for col in cols_wanted:
    if col not in monde.columns:
        raise KeyError(f"Colonne attendue introuvable: {col}")

etat_list = monde["État"].astype(str).tolist()
pop2007_list = []
pop2025_list = []
dens2007_list = []
dens2025_list = []

for v in monde["Pop 2007" ].tolist():
    try:
        pop2007_list.append(float(v))
    except Exception:
        pop2007_list.append(float('nan'))
for v in monde["Pop 2025" ].tolist():
    try:
        pop2025_list.append(float(v))
    except Exception:
        pop2025_list.append(float('nan'))
for v in monde["Densité 2007" ].tolist():
    try:
        dens2007_list.append(float(v))
    except Exception:
        dens2007_list.append(float('nan'))
for v in monde["Densité 2025" ].tolist():
    try:
        dens2025_list.append(float(v))
    except Exception:
        dens2025_list.append(float('nan'))

print('Extrait populations: {} états'.format(len(etat_list)))
