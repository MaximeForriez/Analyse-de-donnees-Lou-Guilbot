#coding:utf8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import scipy.stats

def ouvrirUnFichier(nom):
    with open(nom, "r") as fichier:
        contenu = pd.read_csv(fichier)
    return contenu

data = pd.DataFrame(ouvrirUnFichier("./data/pib-vs-energie.csv"))
# Sélection des colonnes demandées
cols = ["PIB_2022", "Utilisation_d_energie_2022"]
print("Colonnes disponibles:", list(data.columns))

missing_cols = [c for c in cols if c not in data.columns]
if missing_cols:
    raise SystemExit(f"Colonnes manquantes dans le fichier CSV: {missing_cols}")

subset = data[cols].copy()

# Convertir en numérique pour traiter les valeurs censurées/non numériques
subset[cols] = subset[cols].apply(pd.to_numeric, errors='coerce')

# Exclure toutes les lignes partielles ou manquantes 
complete_pairs = subset.dropna(how='any')

print(f"Total lignes (original): {len(data)}")
print(f"Lignes avec couples complets (PIB + énergie 2022): {len(complete_pairs)}")

print("Aperçu des couples complets:")
print(complete_pairs.head())

# Sauvegarder le résultat nettoyé
output_path = "./data/complete_pib_energie_2022.csv"
complete_pairs.to_csv(output_path, index=False)
print(f"Résultat enregistré dans {output_path}")

# Calcul de la régression linéaire simple
# Variable explicative: Utilisation_d_energie_2022 (x)
# Variable à expliquer: PIB_2022 (y)
energies = complete_pairs["Utilisation_d_energie_2022"]
pib = complete_pairs["PIB_2022"]

if len(energies) < 2:
    print("Pas assez de points complets pour calculer une régression linéaire.")
else:
    lr = scipy.stats.linregress(energies, pib)
    print("Résultats de la régression linéaire (PIB expliqué par consommation d'énergie):")
    print(f"slope: {lr.slope}")
    print(f"intercept: {lr.intercept}")
    print(f"r-value: {lr.rvalue}")
    print(f"p-value: {lr.pvalue}")
    print(f"standard error: {lr.stderr}")
    # Calcul de la corrélation simple
    try:
        corr_pandas = pib.corr(energies)
        pearson_r, pearson_p = scipy.stats.pearsonr(energies, pib)
        print("\nCorrélation simple entre consommation d'énergie et PIB:")
        print(f"Correlation (pandas.Series.corr): {corr_pandas}")
        print(f"Correlation (scipy.stats.pearsonr): r={pearson_r}, p-value={pearson_p}")
    except Exception as e:
        print(f"Erreur lors du calcul de la corrélation: {e}")

    # Tracé de synthèse: nuage de points + droite de régression
    try:
        plt.figure(figsize=(8,6))
        plt.scatter(energies, pib, alpha=0.7, label='Données (couples complets)')
        # droite de régression
        x_vals = np.linspace(energies.min(), energies.max(), 100)
        y_vals = lr.intercept + lr.slope * x_vals
        plt.plot(x_vals, y_vals, color='red', label='Droite de régression')
        plt.xlabel('Utilisation d\'energie 2022')
        plt.ylabel('PIB 2022')
        plt.title('PIB 2022 expliqué par la consommation énergétique (2022)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.4)
        fig_path = './data/pib_vs_energie_regression_2022.png'
        plt.tight_layout()
        plt.savefig(fig_path)
        print(f"Graphique enregistré dans {fig_path}")
        # Afficher le graphique si l'environnement le permet
        try:
            plt.show()
        except Exception:
            pass
    except Exception as e:
        print(f"Erreur lors du tracé: {e}")


