import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import mode

class HEOMImputer(BaseEstimator, TransformerMixin):
    """
    Imputer que utiliza a métrica HEOM (Heterogeneous Euclidean-Overlap Metric)
    para calcular distâncias entre instâncias com atributos numéricos e categóricos,
    realizando imputação baseada nos k vizinhos mais próximos.
    """
    def __init__(self, k=5, categorical_features=None, normalize=True):
        """
        Inicializa o HEOMImputer.

        Parâmetros:
            k (int): Número de vizinhos mais próximos para usar na imputação
            categorical_features (list): Lista dos índices de colunas categóricas
            normalize (bool): Se deve normalizar características numéricas
        """
        self.k = k
        self.categorical_features = categorical_features if categorical_features is not None else []
        self.normalize = normalize
        self.scaler = MinMaxScaler() if normalize else None
        self.numerical_features = None
        self.X_fitted = None
        
    def fit(self, X, y=None):
        """
        Memoriza o conjunto de dados para ser usado na imputação posteriormente.
        
        Parâmetros:
            X (pandas.DataFrame): Conjunto de dados de treinamento
            y: Ignorado
        
        Retorna:
            self
        """
        # Converter para DataFrame se for numpy array
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
            
        # Identificar colunas numéricas (as que não são categóricas)
        self.numerical_features = [i for i in range(X.shape[1]) if i not in self.categorical_features]
        
        # Armazenar os dados originais
        self.X_fitted = X.copy()
        
        # Calcular os intervalos de valores para normalização na distância
        if self.normalize:
            numeric_data = X.iloc[:, self.numerical_features].values
            self.scaler.fit(numeric_data)
            
        return self
    
    def _heom_distance(self, instance1, instance2):
        """
        Calcula a distância HEOM entre duas instâncias.
        
        HEOM trata valores numéricos e categóricos de forma diferente:
        - Para numéricos: Distância euclidiana normalizada
        - Para categóricos: 0 se iguais, 1 se diferentes ou se um valor está faltando
        
        Parâmetros:
            instance1 (array-like): Primeira instância
            instance2 (array-like): Segunda instância
            
        Retorna:
            float: Distância HEOM entre as instâncias
        """
        squared_sum = 0.0
        
        # Processar atributos categóricos
        for i in self.categorical_features:
            # Verificar se algum dos valores é nulo
            if pd.isna(instance1[i]) or pd.isna(instance2[i]):
                squared_sum += 1.0  # Distância máxima se um é nulo
            elif instance1[i] == instance2[i]:
                squared_sum += 0.0  # Mesma categoria
            else:
                squared_sum += 1.0  # Categorias diferentes
        
        # Processar atributos numéricos
        for i in self.numerical_features:
            # Verificar se algum dos valores é nulo
            if pd.isna(instance1[i]) or pd.isna(instance2[i]):
                squared_sum += 1.0  # Distância máxima se um é nulo
            else:
                # Normalização no intervalo [0,1] se self.normalize for True
                if self.normalize:
                    val1 = self.scaler.transform([[instance1[i]]])[0][0]
                    val2 = self.scaler.transform([[instance2[i]]])[0][0]
                    squared_sum += (val1 - val2) ** 2
                else:
                    squared_sum += ((instance1[i] - instance2[i]) ** 2)
        
        return np.sqrt(squared_sum)
    
    def _find_k_neighbors(self, instance, X_without_nan):
        """
        Encontra os k vizinhos mais próximos para uma instância com base na distância HEOM.
        
        Parâmetros:
            instance (array-like): Instância para encontrar vizinhos
            X_without_nan (pandas.DataFrame): Conjunto de dados sem valores NaN
            
        Retorna:
            list: Índices dos k vizinhos mais próximos
        """
        distances = []
        
        for i, row in X_without_nan.iterrows():
            distance = self._heom_distance(instance, row)
            distances.append((i, distance))
            
        # Ordenar por distância e retornar os k mais próximos
        distances.sort(key=lambda x: x[1])
        neighbors = [idx for idx, _ in distances[:self.k]]
        
        return neighbors
    
    def _impute_value(self, instance, feature_idx, X_without_nan):
        """
        Imputa um valor para um atributo específico com base nos k vizinhos mais próximos.
        
        Parâmetros:
            instance (array-like): Instância com valor faltante
            feature_idx (int): Índice do atributo a ser imputado
            X_without_nan (pandas.DataFrame): Conjunto de dados sem valores NaN
            
        Retorna:
            valor imputado para o atributo
        """
        # Encontrar os k vizinhos mais próximos
        neighbors = self._find_k_neighbors(instance, X_without_nan)
        
        # Obter valores dos vizinhos para o atributo específico
        neighbor_values = X_without_nan.iloc[neighbors, feature_idx].values
        
        # Imputar diferentemente para atributos categóricos e numéricos
        if feature_idx in self.categorical_features:
            # Para categórico, usar o valor mais frequente (moda)
            return mode(neighbor_values).mode[0]
        else:
            # Para numérico, usar a média
            return np.mean(neighbor_values)
    
    def transform(self, X):
        """
        Imputa valores ausentes usando a métrica HEOM e vizinhos mais próximos.
        
        Parâmetros:
            X (pandas.DataFrame): Conjunto de dados com valores ausentes
            
        Retorna:
            pandas.DataFrame: Conjunto de dados com valores imputados
        """
        # Converter para DataFrame se for numpy array
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
            
        # Criar uma cópia dos dados para não modificar o original
        X_imputed = X.copy()
        
        # Encontrar instâncias sem valores NaN do conjunto de treinamento para usar como referência
        X_without_nan = self.X_fitted.dropna()
        
        # Verificar se temos exemplos suficientes sem NaN
        if len(X_without_nan) < self.k:
            raise ValueError(f"Apenas {len(X_without_nan)} instâncias sem NaN disponíveis, mas k={self.k}. Reduza k ou forneça mais dados.")
        
        # Para cada linha com valores ausentes
        for idx, row in X.iterrows():
            if row.isna().any():
                # Identifique colunas com valores ausentes
                na_columns = row.index[row.isna()]
                
                for col_name in na_columns:
                    col_idx = X.columns.get_loc(col_name)
                    # Impute o valor ausente
                    X_imputed.loc[idx, col_name] = self._impute_value(row, col_idx, X_without_nan)
                    
        return X_imputed
    
    def fit_transform(self, X, y=None):
        """
        Método de conveniência para ajustar o imputer e aplicar transformação.
        
        Parâmetros:
            X (pandas.DataFrame): Conjunto de dados com valores ausentes
            y: Ignorado
            
        Retorna:
            pandas.DataFrame: Conjunto de dados com valores imputados
        """
        return self.fit(X).transform(X)
