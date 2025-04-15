"""
Pipeline de prétraitement pour la reconnaissance d'événements dans des textes annotés
au format WebAnno TSV 3.3.

Ce script permet de:
1. Extraire le texte et les annotations des fichiers TSV
2. Convertir les annotations en format BIO (Beginning, Inside, Outside)
3. Préparer les données pour l'entraînement d'un modèle BERT
"""

import os
import re
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Set, Optional
import glob

# Définition des types d'événements
EVENT_TYPES = [
    "conflit",
    "décision gouvernementale",
    "décès",
    "avancée technologique",
    "événement culturel"
]

class TSVAnnotationParser:
    """
    Classe pour parser les fichiers d'annotation au format WebAnno TSV 3.3
    et extraire les annotations d'événements.
    """
    
    def __init__(self, event_types: List[str] = EVENT_TYPES):
        """
        Initialise le parser avec les types d'événements à reconnaître.
        
        Args:
            event_types: Liste des types d'événements à reconnaître
        """
        self.event_types = event_types
        
    def parse_tsv_file(self, file_path: str) -> Dict:
        """
        Parse un fichier TSV et extrait le texte et les annotations.
        
        Args:
            file_path: Chemin vers le fichier TSV
            
        Returns:
            Un dictionnaire contenant le texte et les annotations
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Extraire l'en-tête et le format
        header = []
        i = 0
        while i < len(lines) and not lines[i].startswith('#Text='):
            header.append(lines[i].strip())
            i += 1
        
        # Initialiser les structures de données
        sentences = []
        current_sentence = {'text': '', 'tokens': [], 'annotations': []}
        
        # Parser le contenu
        while i < len(lines):
            line = lines[i].strip()
            
            # Nouvelle phrase
            if line.startswith('#Text='):
                if current_sentence['tokens']:
                    sentences.append(current_sentence)
                    current_sentence = {'text': '', 'tokens': [], 'annotations': []}
                
                current_sentence['text'] = line[6:]  # Extraire le texte après '#Text='
            
            # Ligne de token
            elif line and not line.startswith('#'):
                parts = line.split('\t')
                if len(parts) >= 3:  # Au moins l'ID, les positions et le texte
                    token_id = parts[0]
                    token_span = parts[1]
                    token_text = parts[2]
                    
                    # Extraire les annotations d'événements si présentes
                    event_annotation = None
                    if len(parts) >= 4 and parts[3] != '_':
                        # Vérifier si l'annotation contient un indice (ex: "Décision gouvernementale[1]")
                        match = re.match(r'(.+?)(?:\[(\d+)\])?$', parts[3])
                        if match:
                            event_type = match.group(1)
                            event_index = match.group(2)
                            
                            if event_type in self.event_types:
                                event_annotation = {
                                    'type': event_type,
                                    'index': int(event_index) if event_index else None
                                }
                    
                    # Ajouter le token et son annotation
                    token_info = {
                        'id': token_id,
                        'span': token_span,
                        'text': token_text,
                        'event': event_annotation
                    }
                    current_sentence['tokens'].append(token_info)
                    
                    # Si le token a une annotation, l'ajouter à la liste des annotations
                    if event_annotation:
                        start, end = map(int, token_span.split('-'))
                        annotation = {
                            'token_id': token_id,
                            'start': start,
                            'end': end,
                            'text': token_text,
                            'type': event_annotation['type'],
                            'index': event_annotation['index']
                        }
                        current_sentence['annotations'].append(annotation)
            
            i += 1
        
        # Ajouter la dernière phrase
        if current_sentence['tokens']:
            sentences.append(current_sentence)
        
        # Regrouper les annotations multi-tokens
        for sentence in sentences:
            self._group_multi_token_annotations(sentence)
        
        return {
            'header': header,
            'sentences': sentences
        }
    
    def _group_multi_token_annotations(self, sentence: Dict) -> None:
        """
        Regroupe les annotations qui s'étendent sur plusieurs tokens.
        
        Args:
            sentence: Dictionnaire contenant les informations d'une phrase
        """
        # Regrouper par index d'annotation
        grouped_annotations = defaultdict(list)
        
        for annotation in sentence['annotations']:
            if annotation['index'] is not None:
                key = (annotation['type'], annotation['index'])
                grouped_annotations[key].append(annotation)
            else:
                # Les annotations sans index sont déjà des entités complètes
                key = (annotation['type'], f"single_{annotation['token_id']}")
                grouped_annotations[key].append(annotation)
        
        # Créer des annotations multi-tokens
        multi_token_annotations = []
        
        for (event_type, _), annotations in grouped_annotations.items():
            if len(annotations) > 0:
                # Trier par position de début
                sorted_annotations = sorted(annotations, key=lambda x: x['start'])
                
                # Créer une annotation multi-tokens
                start = sorted_annotations[0]['start']
                end = sorted_annotations[-1]['end']
                text = ' '.join(a['text'] for a in sorted_annotations)
                token_ids = [a['token_id'] for a in sorted_annotations]
                
                multi_token_annotation = {
                    'token_ids': token_ids,
                    'start': start,
                    'end': end,
                    'text': text,
                    'type': event_type
                }
                
                multi_token_annotations.append(multi_token_annotation)
        
        # Remplacer les annotations
        sentence['multi_token_annotations'] = multi_token_annotations


class BIOConverter:
    """
    Classe pour convertir les annotations en format BIO (Beginning, Inside, Outside).
    """
    
    def __init__(self, event_types: List[str] = EVENT_TYPES):
        """
        Initialise le convertisseur avec les types d'événements à reconnaître.
        
        Args:
            event_types: Liste des types d'événements à reconnaître
        """
        self.event_types = event_types
        
    def convert_to_bio(self, parsed_data: Dict) -> List[Dict]:
        """
        Convertit les annotations en format BIO.
        
        Args:
            parsed_data: Données parsées par TSVAnnotationParser
            
        Returns:
            Liste de dictionnaires contenant les tokens et leurs tags BIO
        """
        bio_sentences = []
        
        for sentence in parsed_data['sentences']:
            bio_sentence = {
                'text': sentence['text'],
                'tokens': [],
                'bio_tags': []
            }
            
            # Initialiser tous les tokens comme "O" (Outside)
            for token in sentence['tokens']:
                bio_sentence['tokens'].append(token['text'])
                bio_sentence['bio_tags'].append('O')
            
            # Mettre à jour les tags pour les annotations multi-tokens
            for annotation in sentence.get('multi_token_annotations', []):
                event_type = annotation['type']
                token_ids = annotation['token_ids']
                
                # Convertir les IDs de tokens en indices (0-based)
                token_indices = []
                for token_id in token_ids:
                    for i, token in enumerate(sentence['tokens']):
                        if token['id'] == token_id:
                            token_indices.append(i)
                            break
                
                # Assigner les tags B et I
                for i, idx in enumerate(token_indices):
                    if i == 0:
                        bio_sentence['bio_tags'][idx] = f'B-{event_type}'
                    else:
                        bio_sentence['bio_tags'][idx] = f'I-{event_type}'
            
            bio_sentences.append(bio_sentence)
        
        return bio_sentences


class DatasetBuilder:
    """
    Classe pour construire un dataset à partir des annotations au format BIO.
    """
    
    def __init__(self, event_types: List[str] = EVENT_TYPES):
        """
        Initialise le builder avec les types d'événements à reconnaître.
        
        Args:
            event_types: Liste des types d'événements à reconnaître
        """
        self.event_types = event_types
        self.parser = TSVAnnotationParser(event_types)
        self.converter = BIOConverter(event_types)
        
    def build_from_directory(self, directory: str, pattern: str = "*/MBY3.tsv") -> List[Dict]:
        """
        Construit un dataset à partir des fichiers TSV dans un répertoire.
        
        Args:
            directory: Chemin vers le répertoire contenant les fichiers TSV
            pattern: Pattern pour trouver les fichiers TSV (par défaut: "*/MBY3.tsv")
            
        Returns:
            Liste de dictionnaires contenant les phrases et leurs annotations BIO
        """
        dataset = []
        
        # Trouver tous les fichiers TSV correspondant au pattern
        tsv_files = glob.glob(os.path.join(directory, pattern))
        
        for tsv_file in tsv_files:
            # Parser le fichier TSV
            parsed_data = self.parser.parse_tsv_file(tsv_file)
            
            # Convertir en format BIO
            bio_sentences = self.converter.convert_to_bio(parsed_data)
            
            # Ajouter au dataset
            for bio_sentence in bio_sentences:
                dataset.append({
                    'file': tsv_file,
                    'text': bio_sentence['text'],
                    'tokens': bio_sentence['tokens'],
                    'bio_tags': bio_sentence['bio_tags']
                })
        
        return dataset
    
    def split_dataset(self, dataset: List[Dict], train_ratio: float = 0.7, val_ratio: float = 0.15, 
                     test_ratio: float = 0.15, random_seed: int = 42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Divise le dataset en ensembles d'entraînement, de validation et de test.
        
        Args:
            dataset: Liste de dictionnaires contenant les phrases et leurs annotations BIO
            train_ratio: Proportion de données pour l'entraînement
            val_ratio: Proportion de données pour la validation
            test_ratio: Proportion de données pour le test
            random_seed: Graine aléatoire pour la reproductibilité
            
        Returns:
            Tuple contenant les ensembles d'entraînement, de validation et de test
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "Les ratios doivent sommer à 1"
        
        # Mélanger le dataset
        np.random.seed(random_seed)
        indices = np.random.permutation(len(dataset))
        
        # Calculer les indices de séparation
        train_idx = int(len(dataset) * train_ratio)
        val_idx = train_idx + int(len(dataset) * val_ratio)
        
        # Diviser le dataset
        train_data = [dataset[i] for i in indices[:train_idx]]
        val_data = [dataset[i] for i in indices[train_idx:val_idx]]
        test_data = [dataset[i] for i in indices[val_idx:]]
        
        return train_data, val_data, test_data
    
    def save_to_file(self, dataset: List[Dict], output_file: str) -> None:
        """
        Sauvegarde le dataset dans un fichier.
        
        Args:
            dataset: Liste de dictionnaires contenant les phrases et leurs annotations BIO
            output_file: Chemin vers le fichier de sortie
        """
        # Convertir en DataFrame
        rows = []
        
        for item in dataset:
            for token, tag in zip(item['tokens'], item['bio_tags']):
                rows.append({
                    'text': item['text'],
                    'token': token,
                    'tag': tag
                })
            
            # Ajouter une ligne vide pour séparer les phrases
            rows.append({
                'text': '',
                'token': '',
                'tag': ''
            })
        
        df = pd.DataFrame(rows)
        
        # Sauvegarder au format CSV
        df.to_csv(output_file, index=False)


def main():
    """
    Fonction principale pour tester le pipeline de prétraitement.
    """
    # Exemple d'utilisation
    parser = TSVAnnotationParser()
    converter = BIOConverter()
    builder = DatasetBuilder()
    
    # Parser un fichier TSV
    parsed_data = parser.parse_tsv_file('/home/ubuntu/upload/MBY3.tsv')
    
    # Convertir en format BIO
    bio_sentences = converter.convert_to_bio(parsed_data)
    
    # Afficher les résultats
    for i, sentence in enumerate(bio_sentences):
        print(f"Phrase {i+1}: {sentence['text']}")
        for token, tag in zip(sentence['tokens'], sentence['bio_tags']):
            print(f"{token}: {tag}")
        print()


if __name__ == "__main__":
    main()
