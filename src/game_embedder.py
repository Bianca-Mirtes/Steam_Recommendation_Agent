import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Dict, List, Optional, Tuple, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GameEmbedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", use_gpu=True):
        """
        Inicializa o embedder de jogos
        
        Args:
            model_name: Nome do modelo Sentence-BERT
            use_gpu: Se True, usa GPU quando disponível
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        
        logger.info(f"Carregando modelo {model_name} no dispositivo: {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)
        
        # Cache para embeddings já calculados
        self.embedding_cache = {}
        self.game_id_to_idx = {}  # Mapeamento appid -> índice
        self.idx_to_game_id = {}  # Mapeamento índice -> appid
        self.faiss_index = None
        
    def create_game_embeddings(self, games_df, text_column='combined_text', batch_size=32):
        """
        Cria embeddings para todos os jogos baseados no texto
        
        Args:
            games_df: DataFrame com informações dos jogos
            text_column: Nome da coluna com texto para embedding
            batch_size: Tamanho do batch para processamento
            
        Returns:
            Dict com embeddings e mapeamentos
        """
        if text_column not in games_df.columns:
            raise ValueError(f"DataFrame deve conter coluna '{text_column}'")
        
        if 'appid' not in games_df.columns:
            raise ValueError("DataFrame deve conter coluna 'appid'")
        
        texts = games_df[text_column].tolist()
        game_ids = games_df['appid'].tolist()
        
        # Criar mapeamentos
        self.game_id_to_idx = {appid: idx for idx, appid in enumerate(game_ids)}
        self.idx_to_game_id = {idx: appid for idx, appid in enumerate(game_ids)}
        
        logger.info(f"Criando embeddings para {len(texts)} jogos...")
        
        # Criar embeddings em batches
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Criando embeddings"):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self.model.encode(
                batch_texts,
                convert_to_tensor=True,
                show_progress_bar=False,
                normalize_embeddings=True  # Normalizar para similaridade cosseno
            )
            
            # Mover para CPU se estiver na GPU
            if self.device == "cuda":
                batch_embeddings = batch_embeddings.cpu()
            
            embeddings.append(batch_embeddings.numpy())
        
        # Concatenar todos os embeddings
        text_embeddings = np.vstack(embeddings)
        
        logger.info(f"Embeddings de texto criados: {text_embeddings.shape}")
        
        # Salvar no cache
        for game_id, embedding in zip(game_ids, text_embeddings):
            self.embedding_cache[game_id] = {
                'text_embedding': embedding,
                'idx': self.game_id_to_idx[game_id]
            }
        
        return {
            'text_embeddings': text_embeddings,
            'game_ids': game_ids,
            'game_id_to_idx': self.game_id_to_idx,
            'idx_to_game_id': self.idx_to_game_id
        }
    
    def build_faiss_index(self, embeddings, index_type="IVFFlat", nlist=100, metric='l2'):
        """
        Constrói índice FAISS para busca eficiente
        
        Args:
            embeddings: Array numpy com embeddings
            index_type: Tipo de índice FAISS ('IVFFlat', 'Flat', 'IVFPQ')
            nlist: Número de clusters para índice IVF
            metric: Métrica de distância ('l2', 'ip' para produto interno)
            
        Returns:
            index: Índice FAISS
        """
        dimension = embeddings.shape[1]
        
        if metric == 'l2':
            if index_type == "IVFFlat":
                quantizer = faiss.IndexFlatL2(dimension)
                index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            elif index_type == "Flat":
                index = faiss.IndexFlatL2(dimension)
            elif index_type == "IVFPQ":
                # Para embeddings maiores, usar compressão
                quantizer = faiss.IndexFlatL2(dimension)
                m = 8  # Número de subquantizers
                index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, 8)
            else:
                raise ValueError(f"Tipo de índice não suportado: {index_type}")
        
        elif metric == 'ip':
            if index_type == "IVFFlat":
                quantizer = faiss.IndexFlatIP(dimension)
                index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            elif index_type == "Flat":
                index = faiss.IndexFlatIP(dimension)
            else:
                raise ValueError(f"Tipo de índice não suportado para metric='ip': {index_type}")
        else:
            raise ValueError(f"Métrica não suportada: {metric}")
        
        # Para índices IVF, treinar
        if index_type.startswith('IVF'):
            logger.info("Treinando índice FAISS...")
            index.train(embeddings)
        
        # Adicionar embeddings ao índice
        logger.info("Adicionando embeddings ao índice...")
        index.add(embeddings)
        
        # Configurar parâmetros de busca
        if index_type.startswith('IVF'):
            index.nprobe = min(20, nlist // 2)  # Número de clusters a verificar
        
        logger.info(f"Índice FAISS construído: {index.ntotal} vetores")
        logger.info(f"Parâmetros: tipo={index_type}, métrica={metric}, nprobe={index.nprobe if hasattr(index, 'nprobe') else 'N/A'}")

        self.faiss_index = index
        
        return index
    
    def search_similar_games(self, query, games_df, index, top_k=10, 
                       search_by='text', threshold=0.15, return_dataframe=True):
        """
        Busca jogos similares - VERSÃO MELHORADA
        """
        if 'appid' not in games_df.columns:
            raise ValueError("games_df deve conter coluna 'appid'")
        
        # MELHORIA: Expandir a query para termos relacionados
        query_expanded = query
        query_lower = query.lower()
        
        # Adicionar termos relacionados baseados na query
        related_terms = {
            'terror': ['horror', 'assustador', 'medo', 'macabro', 'sobrenatural'],
            'multijogador': ['coop', 'cooperativo', 'online', 'amigos', 'grupo'],
            'coop': ['cooperativo', 'multijogador', 'equipe', 'juntos'],
            'ação': ['tiro', 'fps', 'combate', 'batalha', 'adrenalina'],
            'rpg': ['role playing', 'personagem', 'level', 'experiência'],
        }
        
        for term, related_list in related_terms.items():
            if term in query_lower:
                for related in related_list[:2]:  # Adicionar 2 termos relacionados
                    if related not in query_lower:
                        query_expanded += " " + related
        
        # Criar embedding da query
        if search_by == 'text':
            if isinstance(query_expanded, str):
                query_embedding = self.model.encode(
                    query_expanded,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                    normalize_embeddings=True
                )
                
                if self.device == "cuda":
                    query_embedding = query_embedding.cpu()
                
                query_embedding = query_embedding.numpy()
            else:
                raise ValueError("Para search_by='text', query deve ser string")
        
        elif search_by == 'embedding':
            if isinstance(query, np.ndarray):
                query_embedding = query
            else:
                raise ValueError("Para search_by='embedding', query deve ser numpy array")
        else:
            raise ValueError(f"search_by deve ser 'text' ou 'embedding', recebido: {search_by}")
        
        # Garantir shape correto
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

                # Verificar se a dimensão do embedding da query é igual à dimensão do índice
        if query_embedding.shape[1] != index.d:
            raise ValueError(f"Dimensão do embedding da query ({query_embedding.shape[1]}) não coincide com a dimensão do índice ({index.d})")
        
        # Buscar mais resultados para depois filtrar
        search_k = min(top_k * 5, index.ntotal)
        distances, indices = index.search(query_embedding, search_k)
        
        # Coletar resultados acima do threshold
        results = []
        for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
            if idx < len(games_df):
                # Converter para appid
                game_id = self.idx_to_game_id.get(idx, idx)
                
                # Encontrar jogo no DataFrame
                game_data = games_df[games_df['appid'] == game_id]
                if len(game_data) == 0:
                    continue
                
                game_info = game_data.iloc[0]
                
                # similaridade baseada na métrica l2 (padrão do FAISS)
                similarity = 1 / (1 + distance)  # Converte para 0-1
                
                # BOOST para jogos populares
                if 'positive_ratings' in game_info:
                    positive = float(game_info['positive_ratings'])
                    if positive > 10000:
                        similarity *= 1.3  # +30% boost
                    elif positive > 1000:
                        similarity *= 1.15  # +15% boost
                
                # Coletar informações
                result = {
                    'rank': len(results) + 1,
                    'appid': game_id,
                    'name': game_info.get('name', 'Desconhecido'),
                    'similarity': float(similarity),
                    'distance': float(distance),
                }
                
                # Adicionar informações adicionais se disponíveis
                additional_fields = ['genres', 'categories', 'positive_ratio', 
                                'average_playtime', 'price', 'developer']
                
                for field in additional_fields:
                    if field in game_info:
                        result[field] = game_info[field]
                
                # Descrição curta
                if 'short_description' in game_info:
                    desc = str(game_info['short_description'])
                    result['description'] = desc[:150] + '...' if len(desc) > 150 else desc
                elif 'combined_text' in game_info:
                    desc = str(game_info['combined_text'])
                    result['description'] = desc[:150] + '...' if len(desc) > 150 else desc
                
                results.append(result)
                
                if len(results) >= top_k:
                    break
        
        if return_dataframe and results:
            df = pd.DataFrame(results)
            # Ordenar por similaridade
            df = df.sort_values('similarity', ascending=False).reset_index(drop=True)
            return df
        
        return results
    
    def search_by_game_id(self, game_id, games_df, index, top_k=10, embedding_type='text'):
        """
        Busca jogos similares a um jogo específico
        
        Args:
            game_id: appid do jogo de referência
            games_df: DataFrame com jogos
            index: Índice FAISS
            top_k: Número de resultados
            embedding_type: Tipo de embedding a usar ('text', 'hybrid')
            
        Returns:
            DataFrame com resultados
        """
        if game_id not in self.embedding_cache:
            raise ValueError(f"Game ID {game_id} não encontrado no cache")
        
        # Obter embedding do jogo
        if embedding_type == 'text':
            query_embedding = self.embedding_cache[game_id]['text_embedding']
        elif embedding_type == 'hybrid' and 'hybrid_embedding' in self.embedding_cache[game_id]:
            query_embedding = self.embedding_cache[game_id]['hybrid_embedding']
        else:
            query_embedding = self.embedding_cache[game_id]['text_embedding']
        
        return self.search_similar_games(
            query_embedding, 
            games_df, 
            index, 
            top_k=top_k, 
            search_by='embedding'
        )
    
    def search_with_filters(self, query_text, games_df, index, top_k=10,
                          min_positive_ratio=0, max_price=None, genres=None):
        """
        Busca jogos similares com filtros
        
        Args:
            query_text: Texto de consulta
            games_df: DataFrame com jogos
            index: Índice FAISS
            top_k: Número de resultados
            min_positive_ratio: Filtro mínimo de avaliações positivas
            max_price: Preço máximo
            genres: Lista de gêneros para filtrar
            
        Returns:
            DataFrame com resultados filtrados
        """
        # Primeiro, buscar sem filtros
        results = self.search_similar_games(
            query_text, games_df, index, top_k=top_k * 3, return_dataframe=True
        )
        
        if len(results) == 0:
            return pd.DataFrame()
        
        # Aplicar filtros
        filtered_results = []
        
        for _, row in results.iterrows():
            game_id = row['appid']
            game_data = games_df[games_df['appid'] == game_id]
            
            if len(game_data) == 0:
                continue
            
            game_info = game_data.iloc[0]
            
            # Verificar filtro de positive_ratio
            if min_positive_ratio > 0:
                if 'positive_ratio' not in game_info or game_info['positive_ratio'] < min_positive_ratio:
                    continue
            
            # Verificar filtro de preço
            if max_price is not None:
                if 'price' in game_info and game_info['price'] > max_price:
                    continue
            
            # Verificar filtro de gêneros
            if genres:
                game_genres = str(game_info.get('genres', '')).lower()
                if not any(genre.lower() in game_genres for genre in genres):
                    continue
            
            filtered_results.append(row.to_dict())
            
            if len(filtered_results) >= top_k:
                break
        
        return pd.DataFrame(filtered_results)
    
    def save_all(self, save_dir="embeddings"):
        """
        Salva todos os dados do embedder
        
        Args:
            save_dir: Diretório para salvar
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Salvar embeddings
        embeddings_data = {
            'cache': self.embedding_cache,
            'game_id_to_idx': self.game_id_to_idx,
            'idx_to_game_id': self.idx_to_game_id,
            'model_name': self.model_name
        }
        
        with open(f"{save_dir}/embeddings.pkl", 'wb') as f:
            pickle.dump(embeddings_data, f)
        
        logger.info(f"Dados salvos em: {save_dir}/")
    
    def load_all(self, save_dir="embeddings"):
        """
        Carrega todos os dados do embedder
        
        Args:
            save_dir: Diretório com dados salvos
        """
        embeddings_path = f"{save_dir}/embeddings.pkl"
        
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {embeddings_path}")
        
        with open(embeddings_path, 'rb') as f:
            data = pickle.load(f)
        
        self.embedding_cache = data['cache']
        self.game_id_to_idx = data['game_id_to_idx']
        self.idx_to_game_id = data['idx_to_game_id']
        
        faiss_index = faiss.read_index(f"{save_dir}/game_index.faiss")
        self.faiss_index = faiss_index
        
        logger.info(f"Dados carregados: {len(self.embedding_cache)} jogos")
    
    def get_game_embedding(self, game_id, embedding_type='text'):
        """
        Obtém embedding de um jogo específico
        
        Args:
            game_id: appid do jogo
            embedding_type: Tipo de embedding ('text', 'hybrid')
            
        Returns:
            embedding ou None se não encontrado
        """
        if game_id not in self.embedding_cache:
            return None
        
        cache_entry = self.embedding_cache[game_id]
        
        if embedding_type == 'hybrid' and 'hybrid_embedding' in cache_entry:
            return cache_entry['hybrid_embedding']
        elif 'text_embedding' in cache_entry:
            return cache_entry['text_embedding']
        
        return None


# Função utilitária para criar embeddings completo
def create_game_embeddings_pipeline(games_df, output_dir="embeddings", 
                                  numeric_features=None, use_hybrid=False):
    """
    Pipeline completo para criação de embeddings
    
    Args:
        games_df: DataFrame com jogos
        output_dir: Diretório de saída
        numeric_features: Features numéricas para embeddings híbridos
        use_hybrid: Se True, cria embeddings híbridos
        
    Returns:
        embedder, index, resultados
    """
    # Inicializar embedder
    embedder = GameEmbedder()
    
    # Criar embeddings
    if use_hybrid and numeric_features:
        result = embedder.create_hybrid_embeddings(
            games_df, 
            numeric_features=numeric_features,
            weight_text=0.7,
            weight_numeric=0.3
        )
        embeddings = result['hybrid_embeddings']
    else:
        result = embedder.create_game_embeddings(games_df)
        embeddings = result['text_embeddings']
    
    # Construir índice FAISS
    index = embedder.build_faiss_index(
        embeddings, 
        index_type="IVFFlat", 
        nlist=min(100, len(embeddings) // 10),
        metric='l2'
    )
    
    # Salvar índice FAISS
    faiss_path = f"{output_dir}/game_index.faiss"
    faiss.write_index(index, faiss_path)
    
    # Salvar todos os dados
    embedder.save_all(output_dir)
    
    return embedder, index, result