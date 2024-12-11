# Prédiction du churn des clients 

## Description 

Dans ce projet de churn, nous disposons d'une base de données d'une entreprise de telecom (Orange) divisée en deux parties. 
La première composée de 80% des individus est celle qui sera consacrée à l'entraînement de notre modèle et la seconde pour le testing. 

## Fonctionnalité principale 

Ce modèle permet de prédire le risque de churn des clients grâce à une probalité pour aider l'entreprise à prendre des dispositions nécéssaires. 

## Installation 

1- Cloner le dépôt 
git clone https://github.com/VivoAZ/Churn-project 

cd Churn-project 

2- Créer et activer un environnement virtuel (venv) 

python -m venv env
source env/bin/activate  # Pour Linux/macOS
env\Scripts\activate     # Pour Windows

3- Installer les dépendances 

pip install -r requirements.txt 

## Execution 

Pour lancer le projet dans votre environnement : 
python main.py

N'oubliez pas de vérifier le chemin d'accès du fichier main.py selon où vous l'avez sauvegardé sur votre machine. 

## Structure du projet 

main.py : Script principal pour l’entraînement et la prédiction du modèle.
churn-bigml-20.csv : Le jeu de données de test.
churn-bigml-80.csv : Le jeu de données de train.
gradient_boosting_model.pkl : Modèle sauvegardé au format .pkl.
Projet churn.ipynb : Notebook Jupyter pour l’analyse exploratoire et les tests.
requirements.txt : Liste des dépendances nécessaires. 

## Données 

Les informations proviennent de la plateforme publique Kaggle. 

## Collaboration 

Si vous souhaitez contribuer :

1- Forkez le projet.
2- Créez une branche (git checkout -b ma-branche).
3- Soumettez une Pull Request. 















