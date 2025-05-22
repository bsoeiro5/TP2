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
        self.k = k
        self.categorical_features = categorical_features if categorical_features is not None else []
        self.normalize = normalize
        self.numerical_features = None
        self.X_fitted = None
        # Armazenar min e max para cada coluna numérica para normalização manual
        self.min_values = {}
        self.max_values = {}
        
    def fit(self, X, y=None):
        # Converter para DataFrame se for numpy array
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
            
        # Identificar colunas numéricas (as que não são categóricas)
        self.numerical_features = [i for i in range(X.shape[1]) if i not in self.categorical_features]
        
        # Armazenar os dados originais
        self.X_fitted = X.copy()
        
        # Calcular min e max para cada coluna numérica para normalização manual
        if self.normalize and len(self.numerical_features) > 0:
            for i in self.numerical_features:
                col_data = X.iloc[:, i].dropna()
                self.min_values[i] = col_data.min()
                self.max_values[i] = col_data.max()
            
        return self
    
    def _normalize_value(self, value, feature_idx):
        """Normaliza um valor único com base no min e max da feature."""
        min_val = self.min_values.get(feature_idx, 0)
        max_val = self.max_values.get(feature_idx, 1)
        range_val = max_val - min_val
        
        if range_val == 0:
            return 0  # Evitar divisão por zero
        
        return (value - min_val) / range_val
    
    def _heom_distance(self, instance1, instance2):
        squared_sum = 0.0
        n_features = 0  # Contador para normalizar a distância
        
        # Processar atributos categóricos
        for i in self.categorical_features:
            n_features += 1
            # Verificar se algum dos valores é nulo
            if pd.isna(instance1.iloc[i]) or pd.isna(instance2.iloc[i]):
                squared_sum += 1.0  # Distância máxima se um é nulo
            elif instance1.iloc[i] == instance2.iloc[i]:
                squared_sum += 0.0  # Mesma categoria
            else:
                squared_sum += 1.0  # Categorias diferentes
        
        # Processar atributos numéricos
        for i in self.numerical_features:
            n_features += 1
            # Verificar se algum dos valores é nulo
            if pd.isna(instance1.iloc[i]) or pd.isna(instance2.iloc[i]):
                squared_sum += 1.0  # Distância máxima se um é nulo
            else:
                try:
                    val1 = float(instance1.iloc[i])
                    val2 = float(instance2.iloc[i])
                    
                    if self.normalize:
                        # Normalizar manualmente cada valor
                        val1_norm = self._normalize_value(val1, i)
                        val2_norm = self._normalize_value(val2, i)
                        squared_sum += (val1_norm - val2_norm) ** 2
                    else:
                        squared_sum += (val1 - val2) ** 2
                except:
                    squared_sum += 1.0  # Usar distância máxima em caso de erro
        
        # Normalizar a distância pelo número de atributos
        if n_features > 0:
            return np.sqrt(squared_sum / n_features)
        return 0.0
    
    def _find_k_neighbors(self, instance, X_without_nan):
        """
        Encontra os k vizinhos mais próximos para uma instância com base na distância HEOM.
        """
        distances = []
        
        for idx, row in X_without_nan.iterrows():
            try:
                distance = self._heom_distance(instance, row)
                distances.append((idx, distance))
            except:
                pass  # Ignorar erros silenciosamente
        
        # Ordenar por distância e retornar os k mais próximos
        distances.sort(key=lambda x: x[1])
        k_adjusted = min(self.k, len(distances))  # Ajustar k se não houver vizinhos suficientes
        neighbors = [idx for idx, _ in distances[:k_adjusted]]
        
        return neighbors
    
    def _impute_value(self, instance, feature_idx, X_without_nan):
        """
        Imputa um valor para um atributo específico com base nos k vizinhos mais próximos.
        """
        try:
            # Encontrar os k vizinhos mais próximos
            neighbors = self._find_k_neighbors(instance, X_without_nan)
            
            if not neighbors:
                # Retornar a média/moda global como fallback
                if feature_idx in self.categorical_features:
                    return X_without_nan.iloc[:, feature_idx].mode().iloc[0]
                else:
                    return X_without_nan.iloc[:, feature_idx].mean()
            
            # Obter valores dos vizinhos para o atributo específico
            neighbor_values = X_without_nan.iloc[neighbors, feature_idx].values
            
            # Imputar diferentemente para atributos categóricos e numéricos
            if feature_idx in self.categorical_features:
                # Para categórico, usar o valor mais frequente (moda)
                try:
                    # Tente usar a versão mais recente (scipy >= 1.9.0)
                    return mode(neighbor_values, keepdims=False)[0]
                except:
                    # Fallback para versões mais antigas do scipy
                    return mode(neighbor_values)[0][0]
            else:
                # Para numérico, usar a média
                return np.mean(neighbor_values)
        except:
            # Retornar a média/moda global como fallback
            if feature_idx in self.categorical_features:
                return X_without_nan.iloc[:, feature_idx].mode().iloc[0]
            else:
                return X_without_nan.iloc[:, feature_idx].mean()
    
    def transform(self, X):
        """
        Imputa valores ausentes usando a métrica HEOM e vizinhos mais próximos.
        """
        # Converter para DataFrame se for numpy array
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
            
        # Criar uma cópia dos dados para não modificar o original
        X_imputed = X.copy()
        
        # Encontrar instâncias sem valores NaN do conjunto de treinamento para usar como referência
        X_without_nan = self.X_fitted.dropna()
        
        # Verificar se temos exemplos suficientes sem NaN
        if len(X_without_nan) < 1:
            raise ValueError("Não há instâncias sem NaN disponíveis para referência na imputação.")
        
        # Para cada linha com valores ausentes
        for idx, row in X.iterrows():
            if row.isna().any():
                # Identifique colunas com valores ausentes
                na_columns = row.index[row.isna()]
                
                for col_name in na_columns:
                    try:
                        col_idx = X.columns.get_loc(col_name)
                        # Impute o valor ausente
                        X_imputed.loc[idx, col_name] = self._impute_value(row, col_idx, X_without_nan)
                    except Exception as e:
                        pass  # Ignorar erros silenciosamente
                    
        return X_imputed
    
    def fit_transform(self, X, y=None):
        """
        Método de conveniência para ajustar o imputer e aplicar transformação.
        """
        return self.fit(X).transform(X)