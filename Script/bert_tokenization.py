"""
Module de tokenisation et d'encodage pour BERT adapté à la reconnaissance d'événements.

Ce script permet de:
1. Tokeniser les textes avec un tokenizer BERT
2. Aligner les annotations BIO avec les tokens BERT
3. Encoder les textes et les annotations pour l'entraînement
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, BertTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Définition des types d'événements
EVENT_TYPES = [
    "conflit",
    "décision gouvernementale",
    "décès",
    "avancée technologique",
    "événement culturel"
]

class BERTTokenizerForEventRecognition:
    """
    Classe pour tokeniser les textes et aligner les annotations BIO avec les tokens BERT.
    """
    
    def __init__(self, model_name: str = "bert-base-multilingual-cased", max_length: int = 128):
        """
        Initialise le tokenizer BERT.
        
        Args:
            model_name: Nom du modèle BERT à utiliser
            max_length: Longueur maximale des séquences
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        
    def tokenize_and_align_labels(self, text: str, tokens: List[str], bio_tags: List[str]) -> Dict:
        """
        Tokenise le texte et aligne les annotations BIO avec les tokens BERT.
        
        Args:
            text: Texte brut
            tokens: Liste des tokens originaux
            bio_tags: Liste des tags BIO correspondant aux tokens originaux
            
        Returns:
            Dictionnaire contenant les tokens BERT et les tags BIO alignés
        """
        tokenized_inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            is_split_into_words=False
        )
        
        # Créer une liste de tags BIO alignés avec les tokens BERT
        aligned_labels = []
        
        # Obtenir les positions des tokens originaux dans le texte
        token_positions = []
        current_pos = 0
        for token in tokens:
            start = text.find(token, current_pos)
            if start == -1:  # Si le token n'est pas trouvé, essayer avec une recherche moins stricte
                for i in range(current_pos, len(text)):
                    if text[i:i+len(token)].lower() == token.lower():
                        start = i
                        break
            
            if start != -1:
                end = start + len(token)
                token_positions.append((start, end))
                current_pos = end
            else:
                # Si le token n'est toujours pas trouvé, utiliser la position actuelle
                token_positions.append((current_pos, current_pos + len(token)))
                current_pos += len(token)
        
        # Aligner les tags BIO avec les tokens BERT
        offset_mapping = tokenized_inputs.pop("offset_mapping")[0].numpy()
        special_tokens_mask = tokenized_inputs.pop("special_tokens_mask")[0].numpy()
        
        for i, (offset_start, offset_end) in enumerate(offset_mapping):
            # Ignorer les tokens spéciaux ([CLS], [SEP], etc.)
            if special_tokens_mask[i] == 1:
                aligned_labels.append("O")
                continue
            
            # Trouver le token original qui correspond à ce token BERT
            token_idx = None
            for j, (token_start, token_end) in enumerate(token_positions):
                if offset_start >= token_start and offset_end <= token_end:
                    token_idx = j
                    break
            
            # Si un token original est trouvé, utiliser son tag BIO
            if token_idx is not None:
                # Pour les tokens BERT qui sont des sous-mots (commençant par ##),
                # utiliser "I-" au lieu de "B-" s'ils ne sont pas le premier sous-mot
                current_tag = bio_tags[token_idx]
                
                # Si ce n'est pas le premier token BERT pour ce token original
                # et que le tag commence par "B-", le remplacer par "I-"
                if i > 0 and offset_mapping[i-1][1] > 0 and token_idx == self._find_token_idx(offset_mapping[i-1], token_positions):
                    if current_tag.startswith("B-"):
                        current_tag = "I-" + current_tag[2:]
                
                aligned_labels.append(current_tag)
            else:
                # Si aucun token original ne correspond, utiliser "O"
                aligned_labels.append("O")
        
        # Tronquer ou remplir la liste des labels pour qu'elle ait la même longueur que les tokens BERT
        if len(aligned_labels) < self.max_length:
            aligned_labels.extend(["O"] * (self.max_length - len(aligned_labels)))
        else:
            aligned_labels = aligned_labels[:self.max_length]
        
        return {
            "input_ids": tokenized_inputs["input_ids"][0],
            "attention_mask": tokenized_inputs["attention_mask"][0],
            "token_type_ids": tokenized_inputs["token_type_ids"][0],
            "labels": aligned_labels
        }
    
    def _find_token_idx(self, offset: Tuple[int, int], token_positions: List[Tuple[int, int]]) -> Optional[int]:
        """
        Trouve l'indice du token original qui correspond à un offset donné.
        
        Args:
            offset: Tuple (start, end) représentant l'offset du token BERT
            token_positions: Liste de tuples (start, end) représentant les positions des tokens originaux
            
        Returns:
            Indice du token original ou None si aucun ne correspond
        """
        offset_start, offset_end = offset
        
        for i, (token_start, token_end) in enumerate(token_positions):
            if offset_start >= token_start and offset_end <= token_end:
                return i
        
        return None


class EventRecognitionDataset(Dataset):
    """
    Dataset PyTorch pour la reconnaissance d'événements.
    """
    
    def __init__(self, encoded_data: List[Dict], label_map: Dict[str, int]):
        """
        Initialise le dataset.
        
        Args:
            encoded_data: Liste de dictionnaires contenant les données encodées
            label_map: Dictionnaire mappant les tags BIO aux indices numériques
        """
        self.encoded_data = encoded_data
        self.label_map = label_map
        
    def __len__(self) -> int:
        """
        Retourne la taille du dataset.
        
        Returns:
            Nombre d'exemples dans le dataset
        """
        return len(self.encoded_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retourne un exemple du dataset.
        
        Args:
            idx: Indice de l'exemple
            
        Returns:
            Dictionnaire contenant les tenseurs pour l'entraînement
        """
        item = self.encoded_data[idx]
        
        # Convertir les labels en indices numériques
        labels = [self.label_map.get(label, 0) for label in item["labels"]]
        
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "token_type_ids": item["token_type_ids"],
            "labels": torch.tensor(labels, dtype=torch.long)
        }


class BERTDataProcessor:
    """
    Classe pour traiter les données et les préparer pour l'entraînement avec BERT.
    """
    
    def __init__(self, model_name: str = "bert-base-multilingual-cased", max_length: int = 128, 
                event_types: List[str] = EVENT_TYPES):
        """
        Initialise le processeur de données.
        
        Args:
            model_name: Nom du modèle BERT à utiliser
            max_length: Longueur maximale des séquences
            event_types: Liste des types d'événements à reconnaître
        """
        self.tokenizer_wrapper = BERTTokenizerForEventRecognition(model_name, max_length)
        self.event_types = event_types
        
        # Créer le mapping des labels
        self.label_map = {"O": 0}
        idx = 1
        for event_type in event_types:
            self.label_map[f"B-{event_type}"] = idx
            idx += 1
            self.label_map[f"I-{event_type}"] = idx
            idx += 1
        
        self.id_to_label = {v: k for k, v in self.label_map.items()}
    
    def process_bio_data(self, bio_data: List[Dict]) -> Tuple[EventRecognitionDataset, EventRecognitionDataset, EventRecognitionDataset]:
        """
        Traite les données BIO et crée les datasets pour l'entraînement, la validation et le test.
        
        Args:
            bio_data: Liste de dictionnaires contenant les phrases et leurs annotations BIO
            
        Returns:
            Tuple contenant les datasets d'entraînement, de validation et de test
        """
        # Encoder les données
        encoded_data = []
        
        for item in bio_data:
            encoded_item = self.tokenizer_wrapper.tokenize_and_align_labels(
                item["text"],
                item["tokens"],
                item["bio_tags"]
            )
            encoded_data.append(encoded_item)
        
        # Diviser les données en ensembles d'entraînement, de validation et de test
        train_data, temp_data = train_test_split(encoded_data, test_size=0.3, random_state=42)
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
        
        # Créer les datasets
        train_dataset = EventRecognitionDataset(train_data, self.label_map)
        val_dataset = EventRecognitionDataset(val_data, self.label_map)
        test_dataset = EventRecognitionDataset(test_data, self.label_map)
        
        return train_dataset, val_dataset, test_dataset
    
    def create_data_loaders(self, train_dataset: EventRecognitionDataset, val_dataset: EventRecognitionDataset, 
                           test_dataset: EventRecognitionDataset, batch_size: int = 16) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Crée les data loaders pour l'entraînement, la validation et le test.
        
        Args:
            train_dataset: Dataset d'entraînement
            val_dataset: Dataset de validation
            test_dataset: Dataset de test
            batch_size: Taille des batchs
            
        Returns:
            Tuple contenant les data loaders d'entraînement, de validation et de test
        """
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        return train_loader, val_loader, test_loader
    
    def get_label_map(self) -> Dict[str, int]:
        """
        Retourne le mapping des labels.
        
        Returns:
            Dictionnaire mappant les tags BIO aux indices numériques
        """
        return self.label_map
    
    def get_id_to_label(self) -> Dict[int, str]:
        """
        Retourne le mapping inverse des labels.
        
        Returns:
            Dictionnaire mappant les indices numériques aux tags BIO
        """
        return self.id_to_label


def main():
    """
    Fonction principale pour tester le module de tokenisation et d'encodage.
    """
    # Exemple d'utilisation
    processor = BERTDataProcessor()
    
    # Exemple de données BIO
    bio_data = [
        {
            "text": "Democratic Senator Harry Reid has been elected the new leader of the Democratic party in the Senate.",
            "tokens": ["Democratic", "Senator", "Harry", "Reid", "has", "been", "elected", "the", "new", "leader", "of", "the", "Democratic", "party", "in", "the", "Senate", "."],
            "bio_tags": ["O", "O", "O", "O", "O", "O", "B-décision gouvernementale", "O", "B-décision gouvernementale", "I-décision gouvernementale", "O", "O", "O", "O", "O", "O", "O", "O"]
        }
    ]
    
    # Encoder les données
    encoded_item = processor.tokenizer_wrapper.tokenize_and_align_labels(
        bio_data[0]["text"],
        bio_data[0]["tokens"],
        bio_data[0]["bio_tags"]
    )
    
    # Afficher les résultats
    print("Input IDs:", encoded_item["input_ids"])
    print("Attention Mask:", encoded_item["attention_mask"])
    print("Token Type IDs:", encoded_item["token_type_ids"])
    print("Labels:", encoded_item["labels"])


if __name__ == "__main__":
    main()
