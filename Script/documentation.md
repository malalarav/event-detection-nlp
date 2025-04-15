# Documentation du Pipeline de Reconnaissance Automatique d'evenements

## Introduction

Ce document presente une documentation complète du pipeline de reconnaissance automatique d'evenements dans des textes que nous avons developpe. Ce pipeline utilise des techniques avancees de traitement du langage naturel (NLP) et d'apprentissage profond pour identifier et classifier automatiquement differents types d'evenements dans des textes annotes.

Le système est conçu pour reconnaître cinq types d'evenements :
- Conflit
- Decision gouvernementale
- Decès
- Avancee technologique
- evenement culturel

## Architecture Globale

Le pipeline complet est structure en plusieurs modules interconnectes :

1. **Pretraitement des donnees** : Extraction et transformation des annotations au format WebAnno TSV 3.3
2. **Tokenisation et encodage** : Conversion des textes en tokens BERT et alignement des annotations
3. **Modelisation** : Configuration d'un modèle BERT adapte à la tâche de reconnaissance d'evenements
4. **Fine-tuning** : Entraînement du modèle sur les donnees annotees
5. **evaluation** : Mesure des performances et analyse des erreurs

Chaque module est implemente dans un fichier Python distinct, et un script principal (`main.py`) intègre tous ces modules pour offrir une interface complète.

## Pretraitement des Donnees

### Format des Donnees d'Entree

Les donnees d'entree sont au format WebAnno TSV 3.3, avec deux fichiers par article :
- `INITIAL_CAS.tsv` : Contient le texte brut tokenise
- `MBY3.tsv` : Contient les annotations d'evenements

Les annotations sont representees comme des spans (segments de texte) avec des etiquettes correspondant aux types d'evenements. Certaines annotations peuvent s'etendre sur plusieurs tokens, indiquees par des indices (ex: "Decision gouvernementale[1]").

### Extraction et Transformation

Le module `preprocessing_pipeline.py` implemente trois classes principales :

1. **TSVAnnotationParser** : Parse les fichiers TSV et extrait les annotations
   - Gère les annotations multi-tokens
   - Identifie les types d'evenements

2. **BIOConverter** : Convertit les annotations en format BIO (Beginning, Inside, Outside)
   - "B-" indique le debut d'un evenement
   - "I-" indique la continuation d'un evenement
   - "O" indique l'absence d'evenement

3. **DatasetBuilder** : Construit un dataset à partir des fichiers annotes
   - Divise les donnees en ensembles d'entraînement, de validation et de test
   - Sauvegarde les donnees pretraitees

Le format BIO est particulièrement adapte aux tâches de reconnaissance d'entites nommees (NER), dont la reconnaissance d'evenements est un cas particulier.

## Tokenisation et Encodage

### Choix du Tokenizer

Le module `bert_tokenization.py` implemente la tokenisation et l'encodage des textes pour BERT. Nous avons choisi d'utiliser principalement le tokenizer de CamemBERT, un modèle BERT specifiquement entraîne sur des textes français, mais notre implementation supporte egalement d'autres modèles comme BERT multilingue.

### Alignement des Annotations

L'alignement des annotations BIO avec les tokens BERT est un defi technique important, car le tokenizer BERT peut diviser un mot en plusieurs sous-mots. Notre classe `BERTTokenizerForEventRecognition` implemente une solution robuste :

1. Tokenise le texte avec le tokenizer BERT
2. Trouve la correspondance entre les tokens originaux et les tokens BERT
3. Propage les annotations BIO aux tokens BERT correspondants
4. Gère les cas speciaux comme les tokens de debut/fin de sequence ([CLS], [SEP])

### Dataset PyTorch

La classe `EventRecognitionDataset` cree un dataset PyTorch à partir des donnees encodees, ce qui permet d'utiliser les fonctionnalites avancees de PyTorch comme les DataLoaders pour l'entraînement par batch.

## Modelisation

### Architecture du Modèle

Le module `bert_model.py` implemente deux architectures de modèle :

1. **BERTForEventRecognition** : Modèle BERT standard avec une couche de classification
   - Utilise la dernière couche cachee de BERT pour la classification de tokens
   - Implemente une fonction de perte adaptee à la classification multi-classes

2. **BERTCRFForEventRecognition** : Modèle BERT avec une couche CRF (Conditional Random Field)
   - Ajoute une couche CRF pour modeliser les dependances entre les tags successifs
   - Ameliore la coherence des predictions (evite les sequences invalides comme "O" → "I-")

### Selection du Modèle

La classe `BERTModelSelector` facilite la selection et la configuration du modèle. Nous recommandons plusieurs options :

1. **CamemBERT** : Modèle BERT français, excellent pour les textes en français
2. **BERT multilingue** : Adapte si le corpus contient plusieurs langues
3. **FlauBERT** : Alternative française avec une architecture optimisee
4. **XLM-RoBERTa** : Modèle multilingue avec des performances superieures

Le choix depend de la langue dominante du corpus et des ressources computationnelles disponibles.

## Fine-tuning

### Hyperparamètres

Le module `bert_fine_tuning.py` implemente le processus de fine-tuning avec des hyperparamètres optimises :

- **Learning rate** : 3e-5 (avec decroissance adaptative)
- **Batch size** : 16
- **Epochs** : 10 (avec early stopping)
- **Weight decay** : 0.01
- **Warmup steps** : 500

### Optimisation

Nous utilisons l'optimiseur AdamW avec un scheduler de learning rate qui combine :
- Warmup lineaire au debut de l'entraînement
- Decroissance lineaire pour le reste de l'entraînement
- Reduction du learning rate sur plateau (quand les performances stagnent)

### Early Stopping

La classe `EarlyStopping` implemente l'arrêt precoce de l'entraînement lorsque les performances sur l'ensemble de validation ne s'ameliorent plus pendant un certain nombre d'epoques, ce qui evite le surapprentissage.

### Suivi des Metriques

La classe `MetricsTracker` suit l'evolution des metriques pendant l'entraînement et genère des visualisations pour analyser la progression.

## evaluation

### Metriques

Le module `evaluation_methodology.py` implemente une evaluation complète avec plusieurs niveaux de metriques :

1. **Metriques au niveau des tokens** :
   - Precision, rappel et F1-score pour chaque classe
   - Matrices de confusion

2. **Metriques au niveau des entites** :
   - evaluation des evenements complets (pas seulement des tokens individuels)
   - Metriques micro et macro-moyennees

### Validation Croisee

La classe `CrossValidationEvaluator` implemente la validation croisee pour une evaluation plus robuste :
- Divise les donnees en k folds
- Entraîne k modèles differents
- Calcule les moyennes et ecarts-types des performances
- Genère des intervalles de confiance

### Analyse d'Erreurs

La classe `ErrorAnalyzer` fournit une analyse detaillee des erreurs du modèle :
- Identifie les types d'erreurs les plus frequents
- Analyse les erreurs par type d'evenement
- Fournit des exemples concrets d'erreurs avec leur contexte

### Tests Statistiques

La classe `StatisticalTester` implemente des tests statistiques pour comparer les performances de differents modèles :
- Test ANOVA pour la comparaison globale
- Tests post-hoc pour les comparaisons par paires
- Visualisations des resultats des tests

## Script Principal

Le script `main.py` intègre tous les modules et offre une interface complète pour l'entraînement, l'evaluation et l'analyse des modèles. Il accepte de nombreux arguments en ligne de commande pour personnaliser le processus :

- Choix du modèle et de l'architecture
- Hyperparamètres d'entraînement
- Mode d'execution (entraînement, evaluation, prediction)
- Options d'evaluation (validation croisee, analyse d'erreurs)

## Choix Techniques et Justifications

### Pourquoi BERT ?

BERT (Bidirectional Encoder Representations from Transformers) est particulièrement adapte à cette tâche pour plusieurs raisons :

1. **Contextualisation bidirectionnelle** : BERT prend en compte le contexte gauche et droit de chaque mot, ce qui est crucial pour la reconnaissance d'evenements qui dependent souvent du contexte.

2. **Transfert d'apprentissage** : BERT est pre-entraîne sur de vastes corpus, ce qui permet de transferer des connaissances linguistiques generales à notre tâche specifique.

3. **Performance etat de l'art** : Les modèles bases sur BERT ont demontre des performances superieures sur de nombreuses tâches de NLP, y compris la reconnaissance d'entites nommees.

### Pourquoi le Format BIO ?

Le format BIO (Beginning, Inside, Outside) est un standard pour les tâches de sequence labelling comme la reconnaissance d'entites nommees :

1. **Gestion des entites multi-tokens** : Permet de representer des evenements qui s'etendent sur plusieurs mots.

2. **Compatibilite avec les architectures NER** : De nombreux modèles et metriques sont conçus pour ce format.

3. **Distinction debut/continuation** : Permet de differencier le debut d'un nouvel evenement de la continuation d'un evenement existant.

### Pourquoi CRF ?

L'ajout d'une couche CRF (Conditional Random Field) au-dessus de BERT presente plusieurs avantages :

1. **Modelisation des dependances sequentielles** : Le CRF prend en compte les transitions entre les etiquettes successives.

2. **Contraintes structurelles** : Permet d'imposer des contraintes sur les sequences d'etiquettes (par exemple, un tag "I-" doit être precede d'un tag "B-" ou "I-" du même type).

3. **Amelioration des performances** : Les etudes empiriques montrent que l'ajout d'un CRF ameliore generalement les performances sur les tâches de sequence labelling.

## Limites et Perspectives d'Amelioration

### Limites Actuelles

1. **Taille du corpus** : Les performances dependent fortement de la quantite de donnees annotees disponibles.

2. **evenements imbriques** : Le format BIO ne permet pas de representer facilement des evenements imbriques.

3. **Contexte long** : BERT est limite à 512 tokens, ce qui peut être insuffisant pour des documents longs.

### Perspectives d'Amelioration

1. **Modèles plus recents** : Experimenter avec des modèles comme RoBERTa, BART ou T5 qui ont montre des performances superieures à BERT sur certaines tâches.

2. **Augmentation de donnees** : Utiliser des techniques d'augmentation de donnees pour enrichir le corpus d'entraînement.

3. **Apprentissage multi-tâches** : Combiner la reconnaissance d'evenements avec d'autres tâches connexes (comme la detection de relations) pour ameliorer les performances.

4. **Modèles de langage plus grands** : Experimenter avec des modèles comme GPT-3 ou GPT-4 pour la generation d'annotations ou la reconnaissance zero-shot/few-shot.

## Conclusion

Ce pipeline de reconnaissance automatique d'evenements represente une solution complète et robuste pour l'identification et la classification d'evenements dans des textes. Il combine des techniques etat de l'art en NLP et en apprentissage profond, avec une attention particulière à l'evaluation rigoureuse et à l'analyse des erreurs.

La modularite de l'implementation permet une grande flexibilite et facilite l'adaptation à differents corpus et types d'evenements. Les performances peuvent être optimisees en ajustant les hyperparamètres et en choisissant le modèle de base le plus adapte au corpus.

Cette solution peut être utilisee dans divers domaines comme l'analyse de medias, la veille strategique, ou l'extraction d'informations à partir de documents non structures.
