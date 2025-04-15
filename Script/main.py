"""
Script principal pour l'exécution de l'entraînement et de l'évaluation du modèle de reconnaissance d'événements.

Ce script intègre tous les modules développés pour:
1. Charger et prétraiter les données
2. Tokeniser et encoder les textes
3. Configurer et entraîner le modèle BERT
4. Évaluer les performances du modèle
"""

import os
import argparse
import torch
import numpy as np
import json
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union, Any

# Importer les modules développés
from preprocessing_pipeline import TSVAnnotationParser, BIOConverter, DatasetBuilder
from bert_tokenization import BERTTokenizerForEventRecognition, EventRecognitionDataset, BERTDataProcessor
from bert_model import BERTModelSelector, BERTForEventRecognition, BERTCRFForEventRecognition
from bert_fine_tuning import BERTFineTuner, HyperparameterTuner, get_recommended_hyperparameters
from evaluation_methodology import CrossValidationEvaluator, EventRecognitionEvaluator, ErrorAnalyzer, StatisticalTester

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("main.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Définition des types d'événements
EVENT_TYPES = [
    "conflit",
    "décision gouvernementale",
    "décès",
    "avancée technologique",
    "événement culturel"
]

def parse_arguments():
    """
    Parse les arguments de la ligne de commande.
    
    Returns:
        Arguments parsés
    """
    parser = argparse.ArgumentParser(description="Entraînement et évaluation d'un modèle de reconnaissance d'événements")
    
    # Arguments pour les données
    parser.add_argument("--data_dir", type=str, default="./data", help="Répertoire contenant les données")
    parser.add_argument("--output_dir", type=str, default="./output", help="Répertoire pour sauvegarder les résultats")
    
    # Arguments pour le modèle
    parser.add_argument("--model_name", type=str, default="camembert-base", 
                       help="Nom du modèle BERT à utiliser (camembert-base, bert-base-multilingual-cased, etc.)")
    parser.add_argument("--use_crf", action="store_true", help="Utiliser une couche CRF")
    
    # Arguments pour l'entraînement
    parser.add_argument("--batch_size", type=int, default=16, help="Taille des batchs")
    parser.add_argument("--epochs", type=int, default=10, help="Nombre d'époques d'entraînement")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Taux d'apprentissage")
    parser.add_argument("--max_length", type=int, default=128, help="Longueur maximale des séquences")
    parser.add_argument("--train_test_split", type=float, default=0.2, help="Proportion de données pour le test")
    
    # Arguments pour l'évaluation
    parser.add_argument("--cross_validation", action="store_true", help="Utiliser la validation croisée")
    parser.add_argument("--n_splits", type=int, default=5, help="Nombre de folds pour la validation croisée")
    
    # Arguments pour le mode d'exécution
    parser.add_argument("--mode", type=str, choices=["train", "evaluate", "predict", "full"], default="full",
                       help="Mode d'exécution (train, evaluate, predict, full)")
    parser.add_argument("--model_path", type=str, help="Chemin vers un modèle pré-entraîné pour l'évaluation ou la prédiction")
    
    return parser.parse_args()

def load_and_preprocess_data(data_dir: str, event_types: List[str] = EVENT_TYPES) -> Tuple[List[Dict], Dict[str, int]]:
    """
    Charge et prétraite les données.
    
    Args:
        data_dir: Répertoire contenant les données
        event_types: Liste des types d'événements à reconnaître
        
    Returns:
        Tuple contenant les données prétraitées et le mapping des labels
    """
    logger.info("Chargement et prétraitement des données...")
    
    # Créer le builder de dataset
    builder = DatasetBuilder(event_types)
    
    # Construire le dataset à partir des fichiers TSV
    dataset = builder.build_from_directory(data_dir, pattern="*/MBY3.tsv")
    
    logger.info(f"Nombre d'exemples chargés: {len(dataset)}")
    
    # Créer le mapping des labels
    label_map = {"O": 0}
    idx = 1
    for event_type in event_types:
        label_map[f"B-{event_type}"] = idx
        idx += 1
        label_map[f"I-{event_type}"] = idx
        idx += 1
    
    id_to_label = {v: k for k, v in label_map.items()}
    
    logger.info(f"Mapping des labels: {label_map}")
    
    return dataset, label_map, id_to_label

def tokenize_and_encode_data(dataset: List[Dict], label_map: Dict[str, int], model_name: str, max_length: int, 
                           batch_size: int, train_test_split: float) -> Tuple[torch.utils.data.DataLoader, 
                                                                            torch.utils.data.DataLoader, 
                                                                            torch.utils.data.DataLoader]:
    """
    Tokenise et encode les données.
    
    Args:
        dataset: Données prétraitées
        label_map: Mapping des labels
        model_name: Nom du modèle BERT à utiliser
        max_length: Longueur maximale des séquences
        batch_size: Taille des batchs
        train_test_split: Proportion de données pour le test
        
    Returns:
        Tuple contenant les dataloaders d'entraînement, de validation et de test
    """
    logger.info("Tokenisation et encodage des données...")
    
    # Créer le processeur de données
    processor = BERTDataProcessor(model_name, max_length, EVENT_TYPES)
    
    # Diviser le dataset
    train_data, val_data, test_data = processor.process_bio_data(dataset)
    
    logger.info(f"Nombre d'exemples d'entraînement: {len(train_data)}")
    logger.info(f"Nombre d'exemples de validation: {len(val_data)}")
    logger.info(f"Nombre d'exemples de test: {len(test_data)}")
    
    # Créer les dataloaders
    train_dataloader, val_dataloader, test_dataloader = processor.create_data_loaders(
        train_data, val_data, test_data, batch_size
    )
    
    return train_dataloader, val_dataloader, test_dataloader, processor.tokenizer_wrapper.tokenizer

def configure_model(model_name: str, use_crf: bool, num_labels: int) -> torch.nn.Module:
    """
    Configure le modèle BERT.
    
    Args:
        model_name: Nom du modèle BERT à utiliser
        use_crf: Utiliser une couche CRF
        num_labels: Nombre de labels (classes) pour la classification
        
    Returns:
        Modèle BERT configuré
    """
    logger.info(f"Configuration du modèle {model_name} (CRF: {use_crf})...")
    
    # Créer le sélecteur de modèle
    selector = BERTModelSelector(num_labels, use_crf)
    
    # Sélectionner le modèle
    model = selector.select_model(model_name)
    
    return model

def train_model(model: torch.nn.Module, train_dataloader: torch.utils.data.DataLoader, 
               val_dataloader: torch.utils.data.DataLoader, id_to_label: Dict[int, str], 
               learning_rate: float, epochs: int, output_dir: str) -> BERTFineTuner:
    """
    Entraîne le modèle.
    
    Args:
        model: Modèle à entraîner
        train_dataloader: DataLoader pour les données d'entraînement
        val_dataloader: DataLoader pour les données de validation
        id_to_label: Dictionnaire mappant les indices numériques aux tags BIO
        learning_rate: Taux d'apprentissage
        epochs: Nombre d'époques d'entraînement
        output_dir: Répertoire pour sauvegarder les résultats
        
    Returns:
        Fine-tuner avec le modèle entraîné
    """
    logger.info("Entraînement du modèle...")
    
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    # Créer le fine-tuner
    fine_tuner = BERTFineTuner(model, id_to_label)
    
    # Entraîner le modèle
    train_params = {
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": 0.01,
        "warmup_steps": 500,
        "max_grad_norm": 1.0
    }
    
    metrics = fine_tuner.train(
        train_dataloader,
        val_dataloader,
        output_dir=os.path.join(output_dir, "model"),
        **train_params
    )
    
    return fine_tuner

def evaluate_model(fine_tuner: BERTFineTuner, test_dataloader: torch.utils.data.DataLoader, 
                 output_dir: str) -> Dict[str, Any]:
    """
    Évalue le modèle.
    
    Args:
        fine_tuner: Fine-tuner avec le modèle entraîné
        test_dataloader: DataLoader pour les données de test
        output_dir: Répertoire pour sauvegarder les résultats
        
    Returns:
        Résultats de l'évaluation
    """
    logger.info("Évaluation du modèle...")
    
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    # Évaluer le modèle
    eval_results = fine_tuner.evaluate(
        test_dataloader,
        output_dir=os.path.join(output_dir, "evaluation")
    )
    
    return eval_results

def analyze_errors(model: torch.nn.Module, test_dataloader: torch.utils.data.DataLoader, 
                 id_to_label: Dict[int, str], tokenizer, output_dir: str) -> Dict[str, Any]:
    """
    Analyse les erreurs du modèle.
    
    Args:
        model: Modèle à analyser
        test_dataloader: DataLoader pour les données de test
        id_to_label: Dictionnaire mappant les indices numériques aux tags BIO
        tokenizer: Tokenizer utilisé pour le modèle
        output_dir: Répertoire pour sauvegarder les résultats
        
    Returns:
        Résultats de l'analyse
    """
    logger.info("Analyse des erreurs...")
    
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    # Créer l'analyseur d'erreurs
    analyzer = ErrorAnalyzer(model, id_to_label, tokenizer)
    
    # Analyser les erreurs
    error_analysis = analyzer.analyze(
        test_dataloader,
        output_dir=os.path.join(output_dir, "error_analysis")
    )
    
    return error_analysis

def cross_validate_model(dataset: torch.utils.data.Dataset, model_builder, fine_tuner_class, 
                       id_to_label: Dict[int, str], n_splits: int, batch_size: int, 
                       train_params: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """
    Évalue le modèle avec validation croisée.
    
    Args:
        dataset: Dataset complet
        model_builder: Fonction pour construire un nouveau modèle
        fine_tuner_class: Classe pour fine-tuner le modèle
        id_to_label: Dictionnaire mappant les indices numériques aux tags BIO
        n_splits: Nombre de folds pour la validation croisée
        batch_size: Taille des batchs
        train_params: Paramètres pour l'entraînement
        output_dir: Répertoire pour sauvegarder les résultats
        
    Returns:
        Résultats de la validation croisée
    """
    logger.info(f"Validation croisée avec {n_splits} folds...")
    
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    # Créer l'évaluateur avec validation croisée
    evaluator = CrossValidationEvaluator(
        dataset,
        model_builder,
        fine_tuner_class,
        id_to_label,
        n_splits=n_splits,
        batch_size=batch_size
    )
    
    # Évaluer le modèle avec validation croisée
    cv_results = evaluator.evaluate(
        train_params,
        output_dir=os.path.join(output_dir, "cross_validation")
    )
    
    return cv_results

def main():
    """
    Fonction principale.
    """
    # Parser les arguments
    args = parse_arguments()
    
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Créer un sous-répertoire avec la date et l'heure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Sauvegarder les arguments
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    
    # Charger et prétraiter les données
    dataset, label_map, id_to_label = load_and_preprocess_data(args.data_dir, EVENT_TYPES)
    
    # Tokeniser et encoder les données
    train_dataloader, val_dataloader, test_dataloader, tokenizer = tokenize_and_encode_data(
        dataset,
        label_map,
        args.model_name,
        args.max_length,
        args.batch_size,
        args.train_test_split
    )
    
    # Nombre de labels
    num_labels = len(label_map)
    
    if args.mode in ["train", "full"]:
        # Configurer le modèle
        model = configure_model(args.model_name, args.use_crf, num_labels)
        
        # Entraîner le modèle
        fine_tuner = train_model(
            model,
            train_dataloader,
            val_dataloader,
            id_to_label,
            args.learning_rate,
            args.epochs,
            output_dir
        )
        
        # Sauvegarder le modèle final
        model_path = os.path.join(output_dir, "final_model")
        os.makedirs(model_path, exist_ok=True)
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        logger.info(f"Modèle final sauvegardé dans {model_path}")
    
    if args.mode in ["evaluate", "full"]:
        # Charger le modèle si nécessaire
        if args.mode == "evaluate" and args.model_path:
            if args.use_crf:
                model = BERTCRFForEventRecognition.from_pretrained(args.model_path, num_labels=num_labels)
            else:
                model = BERTForEventRecognition.from_pretrained(args.model_path, num_labels=num_labels)
            
            fine_tuner = BERTFineTuner(model, id_to_label)
        
        # Évaluer le modèle
        eval_results = evaluate_model(
            fine_tuner,
            test_dataloader,
            output_dir
        )
        
        # Analyser les erreurs
        error_analysis = analyze_errors(
            model,
            test_dataloader,
            id_to_label,
            tokenizer,
            output_dir
        )
    
    if args.cross_validation:
        # Définir une fonction pour construire un nouveau modèle
        def model_builder():
            return configure_model(args.model_name, args.use_crf, num_labels)
        
        # Définir les paramètres d'entraînement
        train_params = {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": 0.01,
            "warmup_steps": 500,
            "max_grad_norm": 1.0
        }
        
        # Évaluer le modèle avec validation croisée
        cv_results = cross_validate_model(
            dataset,
            model_builder,
            BERTFineTuner,
            id_to_label,
            args.n_splits,
            args.batch_size,
            train_params,
            output_dir
        )
    
    logger.info("Terminé!")

if __name__ == "__main__":
    main()