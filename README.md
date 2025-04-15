# Détection d'événements dans des textes journalistiques avec BERT

## Objectif du projet
Ce projet vise à développer un modèle capable de détecter automatiquement des *événements significatifs* dans des articles de presse en anglais, tels que :
-  *Avancée technologique*
-  *Décision gouvernementale*
-  *Décès*
-  *Conflit*
-  *Événement culturel*

Les étiquettes ont été définies dans le cadre d'une annotation humaine avec [INCEpTION](https://inception-project.github.io/), puis exploitable dans un modèle BERT pour effectuer de la reconnaissance d'entités nommées (NER).

---

##  Méthodologie globale

### 1. *Annotation et export*
Les textes journalistiques sont annotés à la main via Inception, puis exportés au format *WebAnno TSV*. Chaque ligne contient un token et éventuellement une entité annotée.

### 2. *Prétraitement des données*
- Nettoyage des labels : suppression des suffixes [1], fusion des doublons, homogénéisation des noms.
- Conversion vers le format standard *BIO* (Begin / Inside / Outside) pour la NER.
- Création d'un dataset tokens + labels utilisable par le modèle.

### 3. *Tokenisation et alignement*
- Utilisation de BertTokenizerFast pour découper les mots en sous-tokens (wordpieces).
- Alignement des labels avec ces sous-tokens, les sous-tokens non initiaux étant ignorés (-100).

### 4. *Entraînement du modèle*
- Modèle choisi : bert-base-uncased (pré-entrainé en anglais).
- Fine-tuning avec Trainer de Hugging Face.
- Métriques suivies : *f1, **précision, **rappel*.

### 5. *Évaluation et visualisations*
- Affichage de la matrice de confusion
- F1-score par entité
- Représentation visuelle des entités dans les phrases

---

## Fonctions principales du projet

### Traitement des fichiers .tsv
- Extraction des phrases + labels depuis les fichiers exportés Inception
- Nettoyage des labels non pertinents ou ambigus

###  Statistiques et visualisations
- Distribution des labels
- Longueurs de phrases (histogramme)
- Nombre de phrases par type d'événement

###  Tokenisation & alignement BIO
- Utilisation de word_ids() pour aligner les labels correctement avec les sous-tokens BERT

###  Entraînement
- Entraînement du modèle via Trainer
- Split train/test automatique (90/10)
- Sauvegarde du modèle et du tokenizer

###  Évaluation
- Rapport classification_report de sklearn
- Matrice de confusion + visualisation
- F1-score par entité (barplot)

---

##  Résultats

| Métrique   | Valeur |
|-------------|--------|
| F1-score    | 00000  |
| Précision  | 00000 |
| Rappel      | 00000|
| Loss (test) | 00000  |

Ces résultats sont encourageants étant donné le volume de données limité et l'équilibrage parfois imparfait.

---

##  Perspectives
- Ajouter plus de données annotées
- Gérer les entités discontinues
- Tester des modèles multilingues (ex : camembert, xlm-roberta)
- Déploiement en démo ou API Flask / Streamlit / HuggingFace Spaces

---

##  Public cible
Ce projet s'adresse aux personnes souhaitant détecter automatiquement des événements dans des textes journalistiques, en particulier dans des contextes d'analyse médiatique, de veille stratégique, ou d'analyse politique / sociétale
