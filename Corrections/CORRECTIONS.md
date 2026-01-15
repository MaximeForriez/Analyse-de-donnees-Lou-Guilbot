# Élements de corrections

## Séance 4.

### Questions

- Il manque quelques éléments.

### Code

- Excellent !

## Séance 5.

### Questions

- **Question 5.** Une statistique travaillant sur la population totale est une statistique exhaustive.

### Code

- La distribution test1 est normale. Il y a un problème avec le calcul de votre *p-value*.

## Séance 6

### Questions

- Excellent !

### Code

- **Question 7.** La réponse est négative, puisqu'il n'y a qu'un classement. Pour faire une comparaison, il en faut au moins deux.

- Le calcul des tests est faux. Vous avez écrit :

```
    spearman_2007 = scipy.stats.spearmanr(pop_ranks_2007, dens_ranks_2007)
    kendall_2007 = scipy.stats.kendalltau(pop_ranks_2007, dens_ranks_2007)

    spearman_2025 = scipy.stats.spearmanr(pop_ranks_2025, dens_ranks_2025)
    kendall_2025 = scipy.stats.kendalltau(pop_ranks_2025, dens_ranks_2025)
```

au lieu de :

```
    spearman_pop = scipy.stats.spearmanr(pop_ranks_2007, pop_ranks_2025)
    kendall_pop = scipy.stats.kendalltau(pop_ranks_2007, pop_ranks_2025)

    spearman_dens = scipy.stats.spearmanr(dens_ranks_2007, dens_ranks_2025)
    kendall_dens = scipy.stats.kendalltau(dens_ranks_2007, dens_ranks_2025)
```

- Il manque l'analyse des tests de Spearman et de Kendall dans votre rapport.

## Séance 7

### Questions

- Il manque quelques éléments.

### Code

- Excellent !

## Séance 8

### Questions

- **Question 4.** Le rapport de corrélation se mesure entre une variable qualitative et une variable quantitative.

### Code

- Excellent !

## Humanités numériques

- De bonnes remarques. Vous avez tout compris à la pédagogique de l'unité.

## Remarques générales

- Aucun dépôt régulier sur `GitHub`.

- Il ne faut jamais mettre les adresses absolues, comme `"C:\Python\seance_6\data\Le-Monde-HS-Etats-du-monde-2007-2025.csv"`. Il faut utiliser l'adresse relative `./data/Le-Monde-HS-Etats-du-monde-2007-2025.csv` à partir du dossier racine.

- Bon travail !
