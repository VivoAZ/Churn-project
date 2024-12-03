import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report


def load_data(file_path, target_column):
    """
    Charge un fichier CSV et sépare les caractéristiques des étiquettes.

    Parameters:
    - file_path (str): Chemin vers le fichier CSV.
    - target_column (str): Nom de la colonne cible.

    Returns:
    - X (DataFrame): Données sans la colonne cible.
    - y (Series): Colonne cible.
    """
    df = pd.read_csv(file_path)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def preprocess_data(numeric_features, categorical_features):
    """
    Configure un préprocesseur combinant standardisation des variables numériques 
    et encodage des variables catégorielles.

    Parameters:
    - numeric_features (list): Liste des colonnes numériques.
    - categorical_features (list): Liste des colonnes catégoriques.

    Returns:
    - preprocessor (ColumnTransformer): Préprocesseur configuré.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(), categorical_features)
        ]
    )
    return preprocessor


def train_and_evaluate_models(models, X_train, y_train, X_test, y_test):
    """
    Entraîne et évalue plusieurs modèles de classification.

    Parameters:
    - models (dict): Dictionnaire de modèles à entraîner.
    - X_train (array): Données d'entraînement.
    - y_train (Series): Étiquettes d'entraînement.
    - X_test (array): Données de test.
    - y_test (Series): Étiquettes de test.

    Returns:
    - results (dict): Résultats des métriques pour chaque modèle.
    """
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        auc_score = roc_auc_score(y_test, y_pred)
        results[name] = {
            "classification_report": classification_report(y_test, y_pred),
            "AUC": auc_score
        }
        print(f"Model: {name}")
        print(results[name]["classification_report"])
        print(f"AUC: {auc_score}\n")
    return results


# Définition des colonnes
categorical_features = ['State', 'International plan', 'Voice mail plan', 'Area code']
numeric_features = ['Account length', 'Number vmail messages', 'Total day minutes', 'Total day calls',
                    'Total day charge', 'Total eve minutes', 'Total eve calls', 'Total eve charge',
                    'Total night minutes', 'Total night calls', 'Total night charge', 'Total intl minutes',
                    'Total intl calls', 'Total intl charge', 'Customer service calls']

# Chargement des données
train_file = 'C:/Users/HP PROBOOK/Documents/churn-bigml-80.csv'
test_file = 'C:/Users/HP PROBOOK/Documents/churn-bigml-20.csv'
X_train, y_train = load_data(train_file, target_column="Churn")
X_test, y_test = load_data(test_file, target_column="Churn")

# Prétraitement
preprocessor = preprocess_data(numeric_features, categorical_features)
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Définition des modèles
models = {
    'Regression Logistique': LogisticRegression(random_state=42),
    'Support Vecteur Machine': SVC(probability=True, random_state=42),
    'K-Neighbors': KNeighborsClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Forêt Aléatoire': RandomForestClassifier(random_state=42)
}

# Entraînement et évaluation
results = train_and_evaluate_models(models, X_train, y_train, X_test, y_test)
