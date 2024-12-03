# Import des données 

import pandas as pd 
file_path_train = 'C:/Users/HP PROBOOK/Documents/churn-bigml-80.csv' 
df_train = pd.read_csv(file_path_train)

df_train.head() 

df_train.info() 

# Séparation des variables 

categorical_features = ['State', 'International plan', 'Voice mail plan', 'Area code'] 
numeric_features = ['Account length', 'Number vmail messages', 'Total day minutes', 'Total day calls', 'Total day charge', 
                    'Total eve minutes', 'Total eve calls', 'Total eve charge', 'Total night minutes', 'Total night calls', 'Total night charge', 
                    'Total intl minutes', 'Total intl calls', 'Total intl charge', 'Customer service calls'] 

# Encodage et standardisation 

from sklearn.preprocessing import OneHotEncoder, StandardScaler 
from sklearn.compose import ColumnTransformer 

# Application 
preprocessor = ColumnTransformer(
    transformers=[
    ('num', StandardScaler(), numeric_features), 
    ('cat', OneHotEncoder(), categorical_features)
    ]) 

# Application du prétraitement aux données de train

# Création d'un nouveau Dataframe pour les caratérisques sans la cible 
X_train = df_train.drop(columns=["Churn"]) 

# Extraction de la colonne cible 
y_train = df_train['Churn'] 

# Application de la transformation 
X_train = preprocessor.fit_transform(X_train) 

# Affichage des dimensions après prétraitement 

print(X_train.shape) 
print(y_train.shape)  

# Chargeons les données de test 
file_path_test = 'C:/Users/HP PROBOOK/Documents/churn-bigml-20.csv' 

df_test = pd.read_csv(file_path_test) 

df_test.head() 

# Verification des dimensions 
print(f'les dimensions du jeu de données de test sont : {df_test.shape}') 

# Prétraitement des données de test 

# Création d'un nouveau Dataframe pour les caratérisques sans la cible 
X_test = df_test.drop(columns=["Churn"]) 

# Extraction de la colonne cible 
y_test = df_test['Churn'] 

# Application de la transformation 
X_test = preprocessor.fit_transform(X_test) 

# Affichage des dimensions après prétraitement 

print(X_test.shape) 
print(y_test.shape) 

# Implémentation et entraînement des modèles 

from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier 
from sklearn.metrics import roc_auc_score, classification_report 

# Definir les modèles à entraîner 

models = {
'Regression Logistique' : LogisticRegression(random_state=42),
'Suport Vecteur Machine' : SVC(probability=True, random_state=42),
'K-N' : KNeighborsClassifier(), 
'GBoost' : GradientBoostingClassifier(random_state=42), 
'Forêt Aléatoire': RandomForestClassifier(random_state=42)
}
   
   # Entraînement et Evaluation des modèles 

for name, model in models.items(): 
    model.fit(X_train, y_train) 
    y_pred = model.predict(X_test) 
    auc_score = roc_auc_score(y_test, y_pred) 
    print(f"Model: {name}") 
    print(classification_report(y_test, y_pred)) 
    print(f"AUC: {auc_score}\n")  
    