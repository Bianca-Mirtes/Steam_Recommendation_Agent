import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import pickle
from scipy.sparse import csr_matrix
import logging
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import json
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============ CONFIGURAÇÃO COMPLETA ============
@dataclass
class ModelConfig:
    """Configuração completa do modelo Transformer"""
    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 3
    ff_dim: int = 512
    dropout: float = 0.2
    n_neighbors: int = 20
    min_interactions: int = 5
    use_attention: bool = True

# ============ TRANSFORMER COMPLETO  ============
class MultiHeadAttention(nn.Module):
    """Mecanismo de atenção multi-head"""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        context = torch.matmul(attention, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        
        return self.out_proj(context)

class TransformerEncoderLayer(nn.Module):
    """Camada do encoder Transformer"""
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention
        attn_out = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x

class GameTransformer(nn.Module):
    """Modelo Transformer completo"""
    def __init__(self, num_users, num_games, config: ModelConfig):
        super().__init__()
        
        self.config = config
        
        # Embeddings básicos
        self.user_embedding = nn.Embedding(num_users, config.embed_dim)
        self.game_embedding = nn.Embedding(num_games, config.embed_dim)
        
        # Positional encoding (opcional para sequências)
        self.position_embedding = nn.Embedding(1000, config.embed_dim)
        
        # Camadas Transformer (se usar atenção)
        if config.use_attention:
            self.transformer_layers = nn.ModuleList([
                TransformerEncoderLayer(config.embed_dim, config.num_heads, config.ff_dim, config.dropout)
                for _ in range(config.num_layers)
            ])
        else:
            self.transformer_layers = None
        
        # Camada de saída
        self.output_layer = nn.Sequential(
            nn.Linear(config.embed_dim, config.ff_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ff_dim // 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Normalizações
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.norm2 = nn.LayerNorm(config.embed_dim)
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Inicialização
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, user_ids, game_ids, sequence_ids=None, sequence_mask=None):
        # Embeddings básicos
        user_emb = self.user_embedding(user_ids).unsqueeze(1)  # [batch, 1, embed_dim]
        game_emb = self.game_embedding(game_ids).unsqueeze(1)  # [batch, 1, embed_dim]
        
        # Se tivermos sequência (para histórico futuro)
        if sequence_ids is not None:
            batch_size, seq_len = sequence_ids.shape
            game_seq_emb = self.game_embedding(sequence_ids)  # [batch, seq_len, embed_dim]
            
            # Adicionar positional encoding
            positions = torch.arange(seq_len, device=user_ids.device).unsqueeze(0)
            pos_emb = self.position_embedding(positions)
            game_seq_emb = game_seq_emb + pos_emb
            
            # Aplicar transformer
            x = self.dropout(game_seq_emb)
            
            if self.transformer_layers:
                for layer in self.transformer_layers:
                    x = layer(x, sequence_mask)
            
            # Pooling da sequência
            seq_representation = x.mean(dim=1)  # [batch, embed_dim]
            
            # Combinação
            combined = user_emb.squeeze(1) + seq_representation + game_emb.squeeze(1)
        else:
            # Modo simples - combinar user e game embeddings
            combined = user_emb.squeeze(1) + game_emb.squeeze(1)
            
            # Aplicar atenção se configurado
            if self.transformer_layers:
                # Stack embeddings
                stacked = torch.stack([user_emb, game_emb], dim=1).squeeze(2)  # [batch, 2, embed_dim]
                
                # Aplicar transformer
                x = self.dropout(stacked)
                for layer in self.transformer_layers:
                    x = layer(x)
                
                # Pooling
                combined = x.mean(dim=1)
        
        # Normalização final
        combined = self.norm2(combined)
        
        # Saída
        output = self.output_layer(combined)
        
        return torch.sigmoid(output) * 5.0, user_emb.squeeze(1), game_emb.squeeze(1)

# ============ CLASSE PRINCIPAL COMPLETA ============
class DeepProfileAnalyzer:
    def __init__(self, config: ModelConfig = None):
        """
        Analisador de perfis com Transformer completo
        """
        self.config = config or ModelConfig()
        
        # Componentes do modelo
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        
        # Encoders e mapeamentos
        self.user_encoder = None
        self.game_encoder = None
        self.user_profiles = None
        self.game_profiles = None
        self.user_game_matrix = None
        
        # Componentes auxiliares
        self.scaler = StandardScaler()
        self.knn = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Inicializado DeepProfileAnalyzer com Transformer no dispositivo: {self.device}")
    
    def train_collaborative_model(self, interactions_df, user_col='user_id',
                                 game_col='appid', rating_col='implicit_rating',
                                 epochs=20, batch_size=256):
        """
        Treina modelo Transformer
        """
        logger.info(f"Treinando modelo Transformer (attention={self.config.use_attention})...")
        
        # 1. Filtragem de usuários ativos
        user_counts = interactions_df[user_col].value_counts()
        active_users = user_counts[user_counts >= self.config.min_interactions].index
        
        if len(active_users) == 0:
            raise ValueError("Nenhum usuário com interações suficientes")
        
        interactions_filtered = interactions_df[interactions_df[user_col].isin(active_users)].copy()
        
        # 2. Criar mapeamentos
        self.user_encoder = {user: idx for idx, user in enumerate(interactions_filtered[user_col].unique())}
        self.game_encoder = {game: idx for idx, game in enumerate(interactions_filtered[game_col].unique())}
        
        # 3. Preparar dados
        data = self._prepare_data(interactions_filtered, user_col, game_col, rating_col)
        
        # 4. Criar modelo Transformer
        n_users = len(self.user_encoder)
        n_games = len(self.game_encoder)
        
        self.model = GameTransformer(
            num_users=n_users,
            num_games=n_games,
            config=self.config
        ).to(self.device)
        
        # 5. Otimizador
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-3,
            weight_decay=1e-5
        )
        
        # 6. Scheduler para learning rate
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5
        )
        
        # 7. Treinamento
        best_loss = float('inf')
        
        for epoch in range(epochs):
            # Treino
            train_loss = self._train_epoch(data, batch_size)
            
            logger.info(f"Época {epoch+1}/{epochs}: Loss = {train_loss:.4f}")
            
            # Scheduler step
            self.scheduler.step(train_loss)
            
            # Salvar melhor modelo
            if train_loss < best_loss:
                best_loss = train_loss
                self.best_model_state = self.model.state_dict().copy()
        
        # 8. Usar melhor modelo
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"Melhor modelo carregado (loss={best_loss:.4f})")
        
        # 9. Extrair embeddings
        self._extract_embeddings(n_users, n_games)
        
        # 10. Criar KNN para similaridade
        if self.user_profiles is not None and len(self.user_profiles) > 1:
            n_neighbors = min(self.config.n_neighbors, len(self.user_profiles) - 1)
            self.knn = NearestNeighbors(
                n_neighbors=n_neighbors,
                metric='cosine',
                n_jobs=-1
            )
            self.knn.fit(self.user_profiles)
            logger.info(f"KNN treinado com {n_neighbors} vizinhos")
        
        logger.info(f"✅ Transformer treinado: {n_users} usuários, {n_games} jogos")
        logger.info(f"   Config: embed_dim={self.config.embed_dim}, "
                   f"heads={self.config.num_heads}, "
                   f"layers={self.config.num_layers}")
        
        return self
    
    def _prepare_data(self, interactions_df, user_col, game_col, rating_col):
        """Prepara dados para treinamento"""
        data = []
        
        for _, row in interactions_df.iterrows():
            user_id = row[user_col]
            game_id = row[game_col]
            
            if user_id in self.user_encoder and game_id in self.game_encoder:
                data.append({
                    'user_idx': self.user_encoder[user_id],
                    'game_idx': self.game_encoder[game_id],
                    'rating': float(row[rating_col])
                })
        
        logger.info(f"Dados preparados: {len(data)} samples")
        return data
    
    def _train_epoch(self, data, batch_size):
        """Treina por uma época"""
        self.model.train()
        total_loss = 0
        np.random.shuffle(data)
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            
            # Preparar batch - SIMPLES
            user_indices = torch.tensor([s['user_idx'] for s in batch], dtype=torch.long).to(self.device)
            game_indices = torch.tensor([s['game_idx'] for s in batch], dtype=torch.long).to(self.device)
            ratings = torch.tensor([s['rating'] for s in batch], dtype=torch.float).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions, _, _ = self.model(user_indices, game_indices)
            predictions = predictions.squeeze()
            
            # Loss
            loss = self.criterion(predictions, ratings)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / max(1, (len(data) / batch_size))
    
    def _extract_embeddings(self, n_users, n_games):
        """Extrai embeddings do modelo treinado"""
        self.model.eval()
        
        # Extrair embeddings de usuários
        user_indices = torch.arange(n_users, dtype=torch.long).to(self.device)
        dummy_game = torch.zeros(n_users, dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            _, user_embeddings, _ = self.model(user_indices, dummy_game)
            self.user_profiles = user_embeddings.cpu().numpy()
        
        # Extrair embeddings de jogos
        game_indices = torch.arange(n_games, dtype=torch.long).to(self.device)
        dummy_user = torch.zeros(n_games, dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            _, _, game_embeddings = self.model(dummy_user, game_indices)
            self.game_profiles = game_embeddings.cpu().numpy()
        
        # Normalizar perfis de usuário
        self.user_profiles = self.scaler.fit_transform(self.user_profiles)
        
        logger.info(f"Embeddings extraídos: Usuários {self.user_profiles.shape}, Jogos {self.game_profiles.shape}")
    
    
    def get_user_profile(self, user_id: Union[str, int]) -> Optional[np.ndarray]:
        """Obtém perfil do usuário"""
        if self.user_encoder is None or user_id not in self.user_encoder:
            return None
        
        user_idx = self.user_encoder[user_id]
        return self.user_profiles[user_idx]
    
    def get_similar_users(self, user_id: Union[str, int], n_similar: int = 5,
                         min_similarity: float = 0.0) -> List[Tuple[str, float]]:
        """Encontra usuários similares"""
        if user_id not in self.user_encoder:
            return []
        
        user_idx = self.user_encoder[user_id]
        user_profile = self.user_profiles[user_idx].reshape(1, -1)
        
        if self.knn is None:
            # Calcular similaridade manualmente
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(user_profile, self.user_profiles)[0]
            similar_indices = np.argsort(similarities)[::-1][1:n_similar+1]
            
            idx_to_user = {v: k for k, v in self.user_encoder.items()}
            similar_users = []
            
            for idx in similar_indices:
                if idx != user_idx:
                    similarity = similarities[idx]
                    if similarity >= min_similarity:
                        similar_users.append((idx_to_user[idx], similarity))
            
            return similar_users
        else:
            # Usar KNN
            n_neighbors = min(n_similar + 1, len(self.user_profiles))
            distances, indices = self.knn.kneighbors(user_profile, n_neighbors=n_neighbors)
            
            idx_to_user = {v: k for k, v in self.user_encoder.items()}
            similar_users = []
            
            for idx, dist in zip(indices[0], distances[0]):
                if idx != user_idx:
                    similarity = 1 - dist
                    if similarity >= min_similarity:
                        similar_users.append((idx_to_user[idx], similarity))
            
            return similar_users[:n_similar]
    
    def recommend_from_profile(self, user_id: str, games_df: pd.DataFrame,
                             top_n: int = 10, exclude_played: bool = True,
                             user_library: Optional[List] = None) -> pd.DataFrame:
        """Gera recomendações usando Transformer"""
        if user_id not in self.user_encoder:
            return pd.DataFrame()
        
        user_idx = self.user_encoder[user_id]
        
        # Jogos já jogados
        played_games = set()
        if exclude_played:
            if user_library is not None:
                played_games = set(user_library)
            elif self.user_game_matrix is not None:
                played_indices = self.user_game_matrix[user_idx].nonzero()[1]
                idx_to_game = {v: k for k, v in self.game_encoder.items()}
                played_games = {idx_to_game[idx] for idx in played_indices if idx in idx_to_game}
        
        # Preparar candidatos
        candidate_games = []
        candidate_indices = []
        
        for _, game_row in games_df.iterrows():
            game_id = game_row['appid']
            
            if game_id in played_games:
                continue
            
            if game_id in self.game_encoder:
                candidate_games.append(game_id)
                candidate_indices.append(self.game_encoder[game_id])
        
        if not candidate_games:
            return pd.DataFrame()
        
        # Fazer predições
        self.model.eval()
        
        user_indices = torch.tensor([user_idx] * len(candidate_games), dtype=torch.long).to(self.device)
        game_indices = torch.tensor(candidate_indices, dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            predictions, _, _ = self.model(user_indices, game_indices)
            scores = predictions.squeeze().cpu().numpy()
        
        # Coletar recomendações
        recommendations = []
        sorted_indices = np.argsort(scores)[::-1]
        
        for rank, idx in enumerate(sorted_indices[:top_n]):
            game_id = candidate_games[idx]
            game_info = games_df[games_df['appid'] == game_id].iloc[0]
            
            recommendations.append({
                'appid': game_id,
                'name': game_info.get('name', 'Desconhecido'),
                'score': float(scores[idx]),
                'transformer_score': float(scores[idx]),
                'genres': game_info.get('genres', ''),
                'positive_ratio': game_info.get('positive_ratio', 0),
                'price': game_info.get('price', 0),
                'rank': rank + 1
            })
        
        return pd.DataFrame(recommendations)
    
    def get_user_stats(self, user_id: Union[str, int], interactions_df: pd.DataFrame) -> Dict:
        """Obtém estatísticas do usuário"""
        user_interactions = interactions_df[interactions_df['user_id'] == user_id]
        
        if len(user_interactions) == 0:
            return {}
        
        # Estatísticas básicas
        total_hours = user_interactions['hours_played'].sum()
        avg_hours = user_interactions['hours_played'].mean()
        num_games = user_interactions['appid'].nunique()
        
        # Análise de perfil Transformer
        profile_analysis = {}
        if user_id in self.user_encoder:
            profile = self.get_user_profile(user_id)
            if profile is not None:
                from sklearn.metrics.pairwise import cosine_similarity
                if self.user_profiles is not None and len(self.user_profiles) > 1:
                    similarities = cosine_similarity([profile], self.user_profiles)[0]
                    avg_similarity = np.mean(similarities[similarities < 0.999])
                else:
                    avg_similarity = 0
                
                # Determinar tipo de perfil baseado no embedding
                profile_norm = np.linalg.norm(profile)
                if profile_norm > 2.0:
                    profile_type = 'enthusiast'
                elif profile_norm < 0.5:
                    profile_type = 'casual'
                else:
                    profile_type = 'balanced'
                
                profile_analysis = {
                    'embedding_norm': float(profile_norm),
                    'avg_user_similarity': float(avg_similarity),
                    'profile_type': profile_type,
                    'transformer_embedding': True,
                    'embedding_dim': len(profile)
                }
        
        # Determinar estilo de jogo
        if avg_hours > 100:
            playstyle = 'hardcore'
        elif avg_hours < 10:
            playstyle = 'casual'
        else:
            playstyle = 'moderate'
        
        return {
            'total_hours': total_hours,
            'avg_hours_per_game': avg_hours,
            'num_games_played': num_games,
            'preferred_playstyle': playstyle,
            'profile_analysis': profile_analysis,
            'transformer_model': True
        }
    
    def save_model(self, filepath: str):
        """Salva o modelo Transformer"""
        model_data = {
            'model_state_dict': self.model.state_dict() if self.model else None,
            'config': self.config,
            'user_encoder': self.user_encoder,
            'game_encoder': self.game_encoder,
            'user_profiles': self.user_profiles,
            'game_profiles': self.game_profiles,
            'user_game_matrix': self.user_game_matrix,
            'scaler': self.scaler,
            'device': str(self.device)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Modelo Transformer salvo em: {filepath}")
    
    def load_model(self, filepath: str):
        """Carrega modelo Transformer salvo"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Carregar configuração
        self.config = model_data['config']
        
        # Recriar modelo
        n_users = len(model_data['user_encoder'])
        n_games = len(model_data['game_encoder'])
        
        self.model = GameTransformer(
            num_users=n_users,
            num_games=n_games,
            config=self.config
        ).to(self.device)
        
        # Carregar pesos
        if model_data['model_state_dict']:
            self.model.load_state_dict(model_data['model_state_dict'])
        
        # Carregar outros dados
        self.user_encoder = model_data['user_encoder']
        self.game_encoder = model_data['game_encoder']
        self.user_profiles = model_data['user_profiles']
        self.game_profiles = model_data['game_profiles']
        self.user_game_matrix = model_data['user_game_matrix']
        self.scaler = model_data['scaler']
        
        # Recriar KNN
        if self.user_profiles is not None and len(self.user_profiles) > 1:
            n_neighbors = min(self.config.n_neighbors, len(self.user_profiles) - 1)
            self.knn = NearestNeighbors(
                n_neighbors=n_neighbors,
                metric='cosine',
                n_jobs=-1
            )
            self.knn.fit(self.user_profiles)
        
        logger.info(f"Modelo Transformer carregado: {n_users} usuários, {n_games} jogos")
        logger.info(f"   Config: embed_dim={self.config.embed_dim}, "
                   f"heads={self.config.num_heads}, "
                   f"layers={self.config.num_layers}")
        
        return self