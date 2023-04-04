from typing import List, Tuple
from collections import defaultdict


def naive_bayes(training_data: List[List[str]], ids: List[str], classes: List[str],
                test_data: List[str]) -> str:
    nbr_features = len(ids)
    nbr_classes = len(classes)

    classe_counts = defaultdict(int)
    feature_counts = defaultdict(lambda: defaultdict(int))

    # Compter les occurrences de chaque label et caractéristique
    for data in training_data:
        classe = data[nbr_features]  # recuperer la valeur du classe
        classe_counts[classe] += 1  # incrementer le nbr occur de la classe
        # pour chaque ligne
        for i in range(nbr_features):
            feature = data[i]  # recuperer valuer propriete de la ligne
            feature_classe_key = ids[i] + "_" + classe  # stocker nomPropriete_valeurClasse ex: Temps_Oui
            feature_counts[feature_classe_key][feature] += 1  # ajouter <"Soleil" , 2 + 1>

    # Calculer les probabilités pour chaque classe
    classes_probabilities = [0] * nbr_classes
    for i in range(nbr_classes):
        classe = classes[i]

        print(f"-------------------------Pour la Classe: ({classe})-------------------------")
        probability = classe_counts[classe] / len(training_data)  # probabilite de la classe
        print(f"P({classe})= {classe_counts[classe]}/{len(training_data)} = {probability}")
        for j in range(nbr_features):  # pour chaque valeur propreite
            feature = test_data[j]
            feature_label_key = ids[j] + "_" + classe
            feature_classe_counts = feature_counts[feature_label_key]
            feature_count = feature_classe_counts.get(feature, 0)  # recuperer ls
            feature_classe_prob = feature_count / classe_counts[classe]
            print(f"propriete: {ids[j]}, valeur: ({feature}), nombre pour la classe:({classe}): {feature_count}\n\t P({feature}|{classe}) = {feature_count}/{classe_counts[classe]} ={feature_classe_prob}")
            probability *= feature_classe_prob

        print(f"\t\t==========Probabilite de la classe ({classe}): {probability} ==========")
        print()
        classes_probabilities[i] = probability

    # Trouver la classe avec la probabilité la plus élevée
    max_index = 0
    for i in range(1, nbr_classes):
        if classes_probabilities[i] > classes_probabilities[max_index]:
            max_index = i

    return classes[max_index]

# Example training data
training_data = [
    ["0", "1", "1", "0", "0", "C1"],
    ["1", "1", "0", "0", "1", "C1"],
    ["1", "0", "1", "1", "0", "C1"],
    ["1", "0", "1", "0", "1", "C1"],
    ["1", "0", "0", "1", "0", "C1"],
    ["0", "1", "0", "1", "0", "C2"],
    ["1", "1", "1", "1", "1", "C2"],
    ["1", "1", "0", "1", "0", "C2"],
    ["1", "1", "1", "0", "1", "C2"],
    ["1", "0", "1", "0", "1", "C2"]
]

ids = ["x1", "x2", "x3", "x4", "x5"]
classes = ["C1", "C2"]

test_data = ["0", "0", "1", "1", "1"]

result = naive_bayes(training_data, ids, classes, test_data)
print(f"La prédiction pour les données de test {test_data} est: {result}")
