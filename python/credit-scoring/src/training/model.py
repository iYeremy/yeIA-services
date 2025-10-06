import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

class CreditScoringModel(nn.Module):

#num_features = caracteristicas de la red neuronal
#hidden_layers = numero de capas que se quiere y el numero de neuronas por cada capa
#dropout = apagar neuronas de manera aleatoria para combatir el over-fitting
#use_batch_norm = toma la salida de la neurona y la normaliza
#activation_fn = funcion de activacion

    def __init__(self, num_features: int, hidden_layers: List[int], dropout_rate: float = 0.1, use_batch_norm: bool = True, activation_fn: str = "ReLU"):
        super(CreditScoringModel, self).__init__()
        
        self.num_features = num_features
        self.hidden_layers_config = hidden_layers
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.activation_fn_name = activation_fn

        layers = []
        input_size = num_features

        # arquitectura dinamica del server
        for i, layer_size in enumerate(hidden_layers):

            # agregar cada una de las neuronas
            layers.append(nn.Linear(input_size, layer_size))

            # normalizacion de la salida
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(layer_size))

            #f uncion de activacion (agregar mas despues)

            if activation_fn == "ReLU":
                layers.append(nn.ReLU())

            #dropout
            layers.append(nn.Dropout(dropout_rate))


            # la salida de la neurona anterior es la entrada de las neuronas siguientes
            input_size = layer_size

        # output layer
        layers.append(nn.Linear(input_size, 1))
        self.network = nn.Sequential(*layers) # pasar todas las capas a la funcion secuencial, para que las capas esten ordenadas


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward Pass
        x: input tensor (batch_size, nun_features)
        -toma el tensor de entrada y lo pasa por toda la red neuronal (Forward pass)
        Retorna: tensor de entrada (batch_size, 1) con probabilidades
        """
        return self.network(x)

    #funciones opcionales para hacer predicciones

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Obtiene la probabilidades de prediccion
        x: input tensor
        Retorna: probabilidades para ambas clases (bueno=1, malo=0)
        """
        with torch.no_grad():
            logits = self.forward(x)
            prob_good = torch.sigmoid(logits)
            prob_bad = 1 - prob_good
            return torch.cat([prob_bad, prob_good], dim = 1)
        
    def predict(self, x: torch.Tensor, threshould: float = 0.5) -> torch.Tensor:
        """
        Obtiene las prediciones en binario (umbral)
        x: input tensor , threshould: clasificacion de thereshould
        Retorna: una prediccion en binario (1=bueno, 0=malo)
        """
        with torch.no_grad():
            logits = self.forward(x)
            probabilidades = torch.signoid(logits)
            return (probabilidades > threshould).int() #Umbral (se puede hacer de manera dinamica tambien)


    def get_model_info(self) -> dict:
        return {
                    "model_type": "CalificacionCrediticiaModel",
                    "num_features": self.num_features,
                    "dropout_rate": self.dropout_rate,
                    "use_batch_norm": self.use_batch_norm,
                    "activation_fn": self.activation_fn_name,
                    "architecture": {
                        "input_layer": self.num_features,
                        "hidden_layers": self.hidden_layers_config,
                        "output_layer": 1
                    },
                    "total_parameters": sum(p.numel() for p in self.parameters()),
                    "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad)
                    
        }