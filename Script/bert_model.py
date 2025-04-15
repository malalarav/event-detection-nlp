"""
Module de sélection et configuration du modèle BERT pour la reconnaissance d'événements.

Ce script permet de:
1. Sélectionner un modèle BERT adapté à la tâche de reconnaissance d'événements
2. Configurer le modèle pour la classification de tokens (NER)
3. Implémenter une architecture avec CRF pour améliorer les prédictions
"""

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertPreTrainedModel, BertConfig
from torchcrf import CRF
from typing import List, Dict, Tuple, Optional, Union

class BERTForEventRecognition(BertPreTrainedModel):
    """
    Modèle BERT pour la reconnaissance d'événements basé sur une architecture de type NER.
    """
    
    def __init__(self, config, num_labels: int):
        """
        Initialise le modèle BERT pour la reconnaissance d'événements.
        
        Args:
            config: Configuration du modèle BERT
            num_labels: Nombre de labels (classes) pour la classification
        """
        super().__init__(config)
        self.num_labels = num_labels
        
        # Modèle BERT de base
        self.bert = BertModel(config)
        
        # Dropout pour la régularisation
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Couche de classification
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        
        # Initialisation des poids
        self.init_weights()
    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, 
               position_ids=None, head_mask=None, inputs_embeds=None, labels=None):
        """
        Passe avant du modèle.
        
        Args:
            input_ids: Indices des tokens d'entrée
            attention_mask: Masque d'attention
            token_type_ids: Indices des types de tokens
            position_ids: Indices de position
            head_mask: Masque pour les têtes d'attention
            inputs_embeds: Embeddings d'entrée
            labels: Labels pour le calcul de la perte
            
        Returns:
            Tuple contenant la perte et les logits
        """
        # Obtenir les embeddings de BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
        )
        
        # Extraire les embeddings de la dernière couche cachée
        sequence_output = outputs[0]
        
        # Appliquer le dropout
        sequence_output = self.dropout(sequence_output)
        
        # Calculer les logits
        logits = self.classifier(sequence_output)
        
        # Calculer la perte si les labels sont fournis
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            
            # Masquer les positions de padding
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss, 
                labels.view(-1), 
                torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            
            loss = loss_fct(active_logits, active_labels)
        
        return (loss, logits) if loss is not None else logits


class BERTCRFForEventRecognition(BertPreTrainedModel):
    """
    Modèle BERT avec CRF pour la reconnaissance d'événements.
    """
    
    def __init__(self, config, num_labels: int):
        """
        Initialise le modèle BERT avec CRF pour la reconnaissance d'événements.
        
        Args:
            config: Configuration du modèle BERT
            num_labels: Nombre de labels (classes) pour la classification
        """
        super().__init__(config)
        self.num_labels = num_labels
        
        # Modèle BERT de base
        self.bert = BertModel(config)
        
        # Dropout pour la régularisation
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Couche de classification
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        
        # Couche CRF
        self.crf = CRF(num_labels, batch_first=True)
        
        # Initialisation des poids
        self.init_weights()
    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, 
               position_ids=None, head_mask=None, inputs_embeds=None, labels=None):
        """
        Passe avant du modèle.
        
        Args:
            input_ids: Indices des tokens d'entrée
            attention_mask: Masque d'attention
            token_type_ids: Indices des types de tokens
            position_ids: Indices de position
            head_mask: Masque pour les têtes d'attention
            inputs_embeds: Embeddings d'entrée
            labels: Labels pour le calcul de la perte
            
        Returns:
            Tuple contenant la perte et les logits
        """
        # Obtenir les embeddings de BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
        )
        
        # Extraire les embeddings de la dernière couche cachée
        sequence_output = outputs[0]
        
        # Appliquer le dropout
        sequence_output = self.dropout(sequence_output)
        
        # Calculer les logits
        emissions = self.classifier(sequence_output)
        
        # Calculer la perte si les labels sont fournis
        loss = None
        if labels is not None:
            # Masquer les positions de padding pour le CRF
            mask = attention_mask.type(torch.bool)
            
            # Calculer la log-vraisemblance négative
            loss = -self.crf(emissions, labels, mask=mask, reduction='mean')
        
        return (loss, emissions) if loss is not None else emissions
    
    def decode(self, emissions, mask=None):
        """
        Décode les émissions pour obtenir les meilleurs chemins (séquences de labels).
        
        Args:
            emissions: Émissions du modèle
            mask: Masque pour les positions valides
            
        Returns:
            Liste des meilleurs chemins
        """
        return self.crf.decode(emissions, mask=mask)


class BERTModelSelector:
    """
    Classe pour sélectionner et configurer un modèle BERT adapté à la tâche de reconnaissance d'événements.
    """
    
    def __init__(self, num_labels: int, use_crf: bool = True):
        """
        Initialise le sélecteur de modèle.
        
        Args:
            num_labels: Nombre de labels (classes) pour la classification
            use_crf: Utiliser une couche CRF pour améliorer les prédictions
        """
        self.num_labels = num_labels
        self.use_crf = use_crf
    
    def select_model(self, model_name: str = "bert-base-multilingual-cased") -> Union[BERTForEventRecognition, BERTCRFForEventRecognition]:
        """
        Sélectionne et configure un modèle BERT adapté à la tâche de reconnaissance d'événements.
        
        Args:
            model_name: Nom du modèle BERT à utiliser
            
        Returns:
            Modèle BERT configuré pour la reconnaissance d'événements
        """
        # Charger la configuration du modèle
        config = BertConfig.from_pretrained(model_name)
        
        # Créer le modèle
        if self.use_crf:
            model = BERTCRFForEventRecognition.from_pretrained(
                model_name,
                config=config,
                num_labels=self.num_labels
            )
        else:
            model = BERTForEventRecognition.from_pretrained(
                model_name,
                config=config,
                num_labels=self.num_labels
            )
        
        return model
    
    @staticmethod
    def get_recommended_models() -> List[Dict[str, str]]:
        """
        Retourne une liste de modèles BERT recommandés pour la tâche de reconnaissance d'événements.
        
        Returns:
            Liste de dictionnaires contenant les informations sur les modèles recommandés
        """
        return [
            {
                "name": "bert-base-multilingual-cased",
                "description": "Modèle BERT multilingue (cased) pré-entraîné sur 104 langues, dont le français",
                "advantages": "Bonne performance sur les langues non-anglaises, adapté aux textes français",
                "disadvantages": "Moins performant que les modèles spécifiques au français sur certaines tâches"
            },
            {
                "name": "camembert-base",
                "description": "Modèle BERT spécifique au français, pré-entraîné sur un large corpus français",
                "advantages": "Excellente performance sur les textes français, meilleure compréhension des nuances linguistiques",
                "disadvantages": "Limité au français, peut être moins adapté si le corpus contient d'autres langues"
            },
            {
                "name": "flaubert/flaubert_base_cased",
                "description": "Modèle BERT français alternatif, pré-entraîné sur un corpus français diversifié",
                "advantages": "Bonne performance sur les textes français, architecture optimisée",
                "disadvantages": "Limité au français, peut nécessiter plus de ressources computationnelles"
            },
            {
                "name": "xlm-roberta-base",
                "description": "Modèle RoBERTa multilingue, pré-entraîné sur 100 langues avec une architecture améliorée",
                "advantages": "Performances supérieures à BERT multilingue sur de nombreuses tâches, robuste aux variations linguistiques",
                "disadvantages": "Plus lourd en termes de ressources computationnelles"
            }
        ]


def main():
    """
    Fonction principale pour tester la sélection et la configuration du modèle BERT.
    """
    # Exemple d'utilisation
    num_labels = 11  # O + 2 * 5 types d'événements (B- et I- pour chaque type)
    
    # Afficher les modèles recommandés
    print("Modèles BERT recommandés pour la reconnaissance d'événements:")
    for model_info in BERTModelSelector.get_recommended_models():
        print(f"- {model_info['name']}: {model_info['description']}")
        print(f"  Avantages: {model_info['advantages']}")
        print(f"  Inconvénients: {model_info['disadvantages']}")
        print()
    
    # Sélectionner un modèle
    selector = BERTModelSelector(num_labels, use_crf=True)
    
    # Pour CamemBERT (spécifique au français)
    try:
        model = selector.select_model("camembert-base")
        print("Modèle CamemBERT sélectionné avec succès!")
    except Exception as e:
        print(f"Erreur lors de la sélection du modèle CamemBERT: {e}")
        print("Utilisation du modèle BERT multilingue par défaut...")
        model = selector.select_model()
    
    # Afficher la structure du modèle
    print(f"Structure du modèle: {model.__class__.__name__}")
    print(f"Nombre de labels: {model.num_labels}")
    print(f"Utilisation de CRF: {isinstance(model, BERTCRFForEventRecognition)}")


if __name__ == "__main__":
    main()
