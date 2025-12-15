import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import pickle
from scipy.sparse import csr_matrix
import logging
from tqdm import tqdm
import warnings
from typing import Dict, List, Optional, Tuple, Union

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProfileAnalyzer:
    def __init__(self, n_components=50, n_neighbors=20, min_interactions=5):
        """
        Inicializa o analisador de perfis
        
        Args:
            n_components: Número de componentes para SVD
            n_neighbors: Número de vizinhos para KNN
            min_interactions: Mínimo de interações para considerar usuário
        """
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_interactions = min_interactions
        self.svd = None
        self.knn = None
        self.scaler = StandardScaler()
        self.user_encoder = None
        self.game_encoder = None
        self.user_profiles = None
        self.game_profiles = None
        self.user_game_matrix = None
        
    def train_collaborative_model(self, interactions_df, user_col='user_id', 
                                 game_col='appid', rating_col='implicit_rating'):
        """
        Treina modelo de filtragem colaborativa com nova estrutura de dados
        
        Args:
            interactions_df: DataFrame com interações usuário-jogo
            user_col: Nome da coluna de usuário
            game_col: Nome da coluna de jogo (appid)
            rating_col: Nome da coluna de rating
        
        Returns:
            self: Modelo treinado
        """
        logger.info("Treinando modelo colaborativo...")
        
        # Filtrar usuários com interações suficientes
        user_counts = interactions_df[user_col].value_counts()
        active_users = user_counts[user_counts >= self.min_interactions].index
        
        if len(active_users) == 0:
            raise ValueError("Nenhum usuário com interações suficientes")
        
        interactions_filtered = interactions_df[interactions_df[user_col].isin(active_users)]
        
        logger.info(f"Usuários ativos: {len(active_users)}")
        logger.info(f"Interações após filtro: {len(interactions_filtered)}")
        
        # Criar mapeamentos
        users = interactions_filtered[user_col].astype('category')
        games = interactions_filtered[game_col].astype('category')
        
        self.user_encoder = {cat: i for i, cat in enumerate(users.cat.categories)}
        self.game_encoder = {cat: i for i, cat in enumerate(games.cat.categories)}
        
        # Mapear para índices numéricos
        user_indices = users.cat.codes.values
        game_indices = games.cat.codes.values
        
        # Criar matriz esparsa
        n_users = len(self.user_encoder)
        n_games = len(self.game_encoder)
        
        ratings = interactions_filtered[rating_col].values
        
        # Normalizar ratings por usuário
        user_rating_means = {}
        for user_id, user_idx in self.user_encoder.items():
            user_mask = (interactions_filtered[user_col] == user_id)
            if user_mask.any():
                user_rating_means[user_idx] = interactions_filtered.loc[user_mask, rating_col].mean()
        
        # Ajustar ratings (subtrair média do usuário)
        adjusted_ratings = ratings.copy()
        for i, (user_idx, game_idx) in enumerate(zip(user_indices, game_indices)):
            if user_idx in user_rating_means:
                adjusted_ratings[i] = ratings[i] - user_rating_means[user_idx]
        
        self.user_game_matrix = csr_matrix(
            (adjusted_ratings, (user_indices, game_indices)),
            shape=(n_users, n_games)
        )
        
        logger.info(f"Matriz usuário-jogo: {self.user_game_matrix.shape}")
        logger.info(f"Densidade: {self.user_game_matrix.nnz / (n_users * n_games):.4%}")
        
        # Aplicar SVD (Matrix Factorization)
        logger.info(f"Aplicando SVD ({self.n_components} componentes)...")
        self.svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        self.user_profiles = self.svd.fit_transform(self.user_game_matrix)
        self.game_profiles = self.svd.components_.T
        
        # Escalar perfis de usuário
        self.user_profiles = self.scaler.fit_transform(self.user_profiles)
        
        # Treinar KNN para encontrar usuários similares
        logger.info(f"Treinando KNN ({self.n_neighbors} vizinhos)...")
        self.knn = NearestNeighbors(n_neighbors=min(self.n_neighbors, n_users-1), 
                                   metric='cosine', 
                                   n_jobs=-1)
        self.knn.fit(self.user_profiles)
        
        explained_variance = self.svd.explained_variance_ratio_.sum()
        logger.info(f"Modelo treinado: {n_users} usuários, {n_games} jogos")
        logger.info(f"Variância explicada: {explained_variance:.2%}")
        
        return self
    
    def get_user_profile(self, user_id: Union[str, int]) -> Optional[np.ndarray]:
        """
        Obtém o perfil embedding de um usuário
        
        Args:
            user_id: ID do usuário
            
        Returns:
            profile: Perfil do usuário ou None se não encontrado
        """
        if self.user_encoder is None or user_id not in self.user_encoder:
            logger.warning(f"Usuário {user_id} não encontrado no encoder")
            return None
        
        user_idx = self.user_encoder[user_id]
        return self.user_profiles[user_idx]
    
    def get_similar_users(self, user_id: Union[str, int], n_similar: int = 5, 
                         min_similarity: float = 0.0) -> List[Tuple[str, float]]:
        """
        Encontra usuários similares
        
        Args:
            user_id: ID do usuário
            n_similar: Número de usuários similares a retornar
            min_similarity: Similaridade mínima
            
        Returns:
            similar_users: Lista de tuplas (user_id, similarity)
        """
        if user_id not in self.user_encoder:
            logger.warning(f"Usuário {user_id} não encontrado")
            return []
        
        user_idx = self.user_encoder[user_id]
        user_profile = self.user_profiles[user_idx].reshape(1, -1)
        
        # Encontrar vizinhos
        n_neighbors = min(n_similar + 1, len(self.user_profiles))
        distances, indices = self.knn.kneighbors(user_profile, n_neighbors=n_neighbors)
        
        # Converter índices para IDs de usuário
        idx_to_user = {v: k for k, v in self.user_encoder.items()}
        similar_users = []
        
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx != user_idx:  # Excluir o próprio usuário
                similar_user_id = idx_to_user[idx]
                similarity = 1 - dist  # Converter distância para similaridade
                
                if similarity >= min_similarity:
                    similar_users.append((similar_user_id, similarity))
        
        return similar_users[:n_similar]
    
    def recommend_from_profile(self, user_id: Union[str, int], games_df: pd.DataFrame,
                             top_n: int = 10, exclude_played: bool = True,
                             user_library: Optional[List] = None) -> pd.DataFrame:
        """
        Recomenda jogos baseado no perfil do usuário
        
        Args:
            user_id: ID do usuário
            games_df: DataFrame com informações dos jogos
            top_n: Número de recomendações
            exclude_played: Se True, exclui jogos já jogados
            user_library: Lista de appids já jogados (opcional)
            
        Returns:
            recommendations: DataFrame com recomendações
        """
        if user_id not in self.user_encoder:
            logger.warning(f"Usuário {user_id} não encontrado")
            return pd.DataFrame()
        
        user_idx = self.user_encoder[user_id]
        
        # Obter jogos já jogados pelo usuário
        if exclude_played:
            if user_library is None:
                # Extrair da matriz esparsa
                played_indices = self.user_game_matrix[user_idx].nonzero()[1]
                played_games = [list(self.game_encoder.keys())[list(self.game_encoder.values()).index(idx)] 
                              for idx in played_indices]
            else:
                played_games = user_library
        
        # Calcular scores para todos os jogos
        user_profile = self.user_profiles[user_idx]
        game_scores = user_profile @ self.game_profiles.T
        
        # Adicionar bias baseado na popularidade do jogo
        if self.user_game_matrix is not None:
            game_popularity = np.array(self.user_game_matrix.sum(axis=0)).flatten()
            game_popularity = np.log1p(game_popularity)  # Suavizar
            game_popularity = game_popularity / game_popularity.max()
            
            # Combinar scores
            combined_scores = 0.7 * game_scores + 0.3 * game_popularity
        else:
            combined_scores = game_scores
        
        # Ordenar jogos por score
        sorted_indices = np.argsort(combined_scores)[::-1]
        
        # Coletar recomendações
        recommendations = []
        for game_idx in sorted_indices:
            if len(recommendations) >= top_n * 2:  # Buscar mais para depois filtrar
                break
            
            game_id = list(self.game_encoder.keys())[list(self.game_encoder.values()).index(game_idx)]
            
            # Pular jogos já jogados
            if exclude_played and game_id in played_games:
                continue
            
            # Encontrar informações do jogo
            game_info = games_df[games_df['appid'] == game_id]
            if len(game_info) == 0:
                continue
            
            game_info = game_info.iloc[0]
            
            recommendations.append({
                'appid': game_id,
                'name': game_info.get('name', 'Desconhecido'),
                'score': float(combined_scores[game_idx]),
                'collaborative_score': float(game_scores[game_idx]),
                'genres': game_info.get('genres', ''),
                'positive_ratio': game_info.get('positive_ratio', 0),
                'price': game_info.get('price', 0)
            })
        
        # Converter para DataFrame e retornar top_n
        if recommendations:
            df = pd.DataFrame(recommendations).head(top_n)
            
            return df
        
        return pd.DataFrame()
    
    def get_user_stats(self, user_id: Union[str, int], interactions_df: pd.DataFrame) -> Dict:
        """
        Obtém estatísticas do usuário
        
        Args:
            user_id: ID do usuário
            interactions_df: DataFrame com interações
            
        Returns:
            Dicionário com estatísticas
        """
        user_interactions = interactions_df[interactions_df['user_id'] == user_id]
        
        if len(user_interactions) == 0:
            return {}
        
        # Calcular estatísticas básicas
        total_hours = user_interactions['hours_played'].sum()
        avg_hours = user_interactions['hours_played'].mean()
        num_games = user_interactions['appid'].nunique()
        
        # Encontrar gêneros mais jogados (se disponível)
        # Esta parte depende da estrutura dos seus dados
        
        return {
            'total_hours': total_hours,
            'avg_hours_per_game': avg_hours,
            'num_games_played': num_games,
            'preferred_playstyle': 'hardcore' if avg_hours > 50 else 'casual' if avg_hours < 10 else 'moderate'
        }
    
    def save_model(self, filepath: str):
        """
        Salva o modelo treinado
        
        Args:
            filepath: Caminho para salvar
        """
        model_data = {
            'svd': self.svd,
            'knn': self.knn,
            'scaler': self.scaler,
            'user_encoder': self.user_encoder,
            'game_encoder': self.game_encoder,
            'user_profiles': self.user_profiles,
            'game_profiles': self.game_profiles,
            'user_game_matrix': self.user_game_matrix,
            'n_components': self.n_components,
            'n_neighbors': self.n_neighbors,
            'min_interactions': self.min_interactions
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Modelo salvo em: {filepath}")
    
    def load_model(self, filepath: str):
        """
        Carrega modelo salvo
        
        Args:
            filepath: Caminho do arquivo
            
        Returns:
            self: Modelo carregado
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.svd = model_data['svd']
        self.knn = model_data['knn']
        self.scaler = model_data['scaler']
        self.user_encoder = model_data['user_encoder']
        self.game_encoder = model_data['game_encoder']
        self.user_profiles = model_data['user_profiles']
        self.game_profiles = model_data['game_profiles']
        self.user_game_matrix = model_data['user_game_matrix']
        self.n_components = model_data['n_components']
        self.n_neighbors = model_data['n_neighbors']
        self.min_interactions = model_data['min_interactions']
        
        logger.info(f"Modelo carregado: {len(self.user_encoder)} usuários, {len(self.game_encoder)} jogos")
        return self