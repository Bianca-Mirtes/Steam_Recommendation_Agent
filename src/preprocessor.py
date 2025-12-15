import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from tqdm import tqdm
import ast
import joblib
import os

class SteamDataPreprocessor:
    def __init__(self, unified_data):
        """
        unified_data: dicionário com:
            - 'games': DataFrame com informações dos jogos
            - 'user_interactions': DataFrame com interações usuário-jogo
            - 'reviews': DataFrame com reviews (opcional)
        """
        self.games_df = unified_data['games']
        self.interactions_df = unified_data['user_interactions']
        self.reviews_df = unified_data.get('reviews', None)
        
    def preprocess_games(self, min_positive_ratio=60, min_total_reviews=500):
        """Pré-processa dados dos jogos com filtros de qualidade"""
        print("Pré-processando dados dos jogos...")
        games_clean = self.games_df.copy()
        
        # 1. Garantir que temos colunas essenciais
        essential_cols = ['appid', 'name']
        missing_cols = [col for col in essential_cols if col not in games_clean.columns]
        if missing_cols:
            print(f"Aviso: Colunas faltando: {missing_cols}")
        
        # 2. Filtrar por qualidade mínima
        if 'positive_ratings' in games_clean.columns and 'negative_ratings' in games_clean.columns:
            games_clean['total_reviews'] = games_clean['positive_ratings'] + games_clean['negative_ratings']
            games_clean['positive_ratio'] = (games_clean['positive_ratings'] / games_clean['total_reviews'] * 100).fillna(0)

            games_clean = games_clean[
                (games_clean['total_reviews'] >= min_total_reviews) |
                (games_clean['positive_ratio'] >= 85)  # Jogos muito bem avaliados mesmo com poucas reviews
            ]
            print(f"  Filtrados para {len(games_clean)} jogos com >{min_total_reviews} reviews e >{min_positive_ratio}% positivos")
        
        # 3. Processar gêneros e categorias (GARANTIR QUE EXISTAM PARA DEEP LEARNING)
        if 'genres' in games_clean.columns:
            games_clean['genres'] = games_clean['genres'].fillna('').astype(str)
            games_clean['genres_list'] = games_clean['genres'].apply(
                lambda x: x.split(';') if pd.notna(x) and x != '' else []
            )
        else:
            games_clean['genres'] = ''
            games_clean['genres_list'] = []
            print("  Aviso: Coluna 'genres' não encontrada - criando coluna vazia")
        
        if 'categories' in games_clean.columns:
            games_clean['categories'] = games_clean['categories'].fillna('').astype(str)
            games_clean['categories_list'] = games_clean['categories'].apply(
                lambda x: x.split(';') if pd.notna(x) and x != '' else []
            )
        
        # 4. Garantir que temos colunas necessárias para Deep Learning
        # positive_ratio (já calculado acima)
        if 'positive_ratio' not in games_clean.columns:
            games_clean['positive_ratio'] = 70  # Valor padrão
        
        # price
        if 'price' not in games_clean.columns:
            if 'price_final' in games_clean.columns:
                games_clean['price'] = games_clean['price_final']
            elif 'price_original' in games_clean.columns:
                games_clean['price'] = games_clean['price_original']
            else:
                games_clean['price'] = 0
                print("  Aviso: Coluna 'price' não encontrada - definindo como 0")
        
        # Garantir tipos numéricos
        games_clean['positive_ratio'] = pd.to_numeric(games_clean['positive_ratio'], errors='coerce').fillna(70)
        games_clean['price'] = pd.to_numeric(games_clean['price'], errors='coerce').fillna(0)
        
        # 5. Extrair ano de lançamento
        if 'release_date' in games_clean.columns:
            games_clean['release_year'] = pd.to_datetime(
                games_clean['release_date'], errors='coerce'
            ).dt.year
        
        # 6. Criar texto combinado para embeddings
        games_clean['combined_text'] = games_clean.apply(self._combine_game_text, axis=1)
        
        # 7. Criar features numéricas normalizadas
        numeric_features = []
        
        if 'positive_ratio' in games_clean.columns:
            games_clean['positive_ratio_norm'] = games_clean['positive_ratio'] / 100
            numeric_features.append('positive_ratio_norm')
        
        if 'average_playtime' in games_clean.columns:
            # Normalizar playtime (log scale devido à distribuição long tail)
            games_clean['avg_playtime_norm'] = np.log1p(games_clean['average_playtime'])
            games_clean['avg_playtime_norm'] = MinMaxScaler().fit_transform(
                games_clean[['avg_playtime_norm']]
            )
            numeric_features.append('avg_playtime_norm')
        
        if 'price' in games_clean.columns:
            games_clean['price_norm'] = MinMaxScaler().fit_transform(
                games_clean[['price']].fillna(0)
            )
            numeric_features.append('price_norm')
        
        # 8. Criar one-hot encoding para gêneros principais
        if 'genres_list' in games_clean.columns and len(games_clean['genres_list'].iloc[0]) > 0:
            # Encontrar gêneros mais comuns
            all_genres = []
            for genres in games_clean['genres_list']:
                if isinstance(genres, list):
                    all_genres.extend(genres)
            
            if all_genres:
                genre_counts = pd.Series(all_genres).value_counts()
                top_genres = genre_counts.head(20).index.tolist()
                
                for genre in top_genres:
                    col_name = f'genre_{genre.replace(" ", "_").lower()}'
                    games_clean[col_name] = games_clean['genres_list'].apply(
                        lambda x: 1 if isinstance(x, list) and genre in x else 0
                    )
                    numeric_features.append(col_name)
        
        print(f"✓ Jogos pré-processados: {len(games_clean)}")
        print(f"  Features numéricas: {len(numeric_features)}")
        print(f"  Metadados para Deep Learning: genres, positive_ratio, price")
        
        return {
            'games': games_clean,
            'numeric_features': numeric_features,
            'game_ids': games_clean['appid'].tolist(),
            # Adicionar metadados específicos para Deep Learning
            'metadata_columns': ['genres', 'positive_ratio', 'price']
        }
    
    def _combine_game_text(self, row):
        """Combina informações textuais para embeddings"""
        text_parts = []
        
        # Nome
        if pd.notna(row.get('name')):
            text_parts.append(str(row['name']))
        
        # Descritores
        descriptors = []
        
        if pd.notna(row.get('short_description')):
            descriptors.append(str(row['short_description']))
        
        if 'genres_list' in row and isinstance(row['genres_list'], list):
            descriptors.extend([f"Genre: {g}" for g in row['genres_list']])
        
        if 'categories_list' in row and isinstance(row['categories_list'], list):
            descriptors.extend([f"Category: {c}" for c in row['categories_list']])
        
        # Developer/Publisher
        if pd.notna(row.get('developer')):
            descriptors.append(f"Developer: {row['developer']}")
        
        if pd.notna(row.get('publisher')):
            descriptors.append(f"Publisher: {row['publisher']}")
        
        # Juntar tudo
        if descriptors:
            text_parts.append(". ".join(descriptors))
        
        return " ".join(text_parts)
    
    def preprocess_interactions(self, min_user_interactions=5, min_game_interactions=10):
        """Pré-processa interações usuário-jogo"""
        print("Pré-processando interações...")
        
        interactions = self.interactions_df.copy()
        
        # 1. Filtrar por jogos que temos informações
        valid_games = set(self.games_df['appid'].unique())
        interactions = interactions[interactions['appid'].isin(valid_games)]
        
        # 2. Filtrar usuários e jogos com interações suficientes
        user_counts = interactions['user_id'].value_counts()
        game_counts = interactions['appid'].value_counts()
        
        active_users = user_counts[user_counts >= min_user_interactions].index
        popular_games = game_counts[game_counts >= min_game_interactions].index
        
        interactions = interactions[
            interactions['user_id'].isin(active_users) &
            interactions['appid'].isin(popular_games)
        ]
        
        print(f"  Após filtragem: {len(interactions)} interações")
        print(f"  Usuários ativos: {len(active_users)}")
        print(f"  Jogos populares: {len(popular_games)}")
        
        # 3. Criar implicit_rating para Deep Learning
        # Baseado em horas jogadas, compras, etc.
        if 'hours_played' in interactions.columns:
            # Normalizar horas jogadas para rating 1-5
            max_hours = interactions['hours_played'].max()
            if max_hours > 0:
                # Log scale para normalizar distribuição long tail
                interactions['implicit_rating'] = interactions['hours_played'].apply(
                    lambda x: 1.0 + 4.0 * (np.log1p(x) / np.log1p(max_hours)) if x > 0 else 1.0
                )
            else:
                interactions['implicit_rating'] = 2.5  # Valor médio
        elif 'purchase' in interactions.columns:
            # Se temos dados de compra
            interactions['implicit_rating'] = interactions['purchase'].apply(
                lambda x: 5.0 if x == 1 else 1.0
            )
        else:
            # Se não temos horas ou compras, criar rating baseado em contagem
            interactions['implicit_rating'] = 3.0  # Valor padrão
        
        # Garantir que o rating esteja entre 1-5
        interactions['implicit_rating'] = interactions['implicit_rating'].clip(1.0, 5.0)
        
        # 4. Criar matriz de interações
        user_encoder = LabelEncoder()
        game_encoder = LabelEncoder()
        
        interactions['user_encoded'] = user_encoder.fit_transform(interactions['user_id'])
        interactions['game_encoded'] = game_encoder.fit_transform(interactions['appid'])
        
        # 5. Criar matriz esparsa de interações (com ratings)
        n_users = len(user_encoder.classes_)
        n_games = len(game_encoder.classes_)
        
        # Para Deep Learning, vamos criar uma matriz com os ratings
        interaction_matrix = np.zeros((n_users, n_games), dtype=np.float32)
        
        for _, row in tqdm(interactions.iterrows(), total=len(interactions), desc="Criando matriz"):
            u = row['user_encoded']
            g = row['game_encoded']
            rating = row['implicit_rating']
            
            interaction_matrix[u, g] = rating
        
        print(f"✓ Matriz de interações: {interaction_matrix.shape}")
        print(f"  Densidade: {(interaction_matrix > 0).sum() / (n_users * n_games):.4%}")
        print(f"  Ratings: min={interaction_matrix[interaction_matrix > 0].min():.2f}, "
              f"max={interaction_matrix.max():.2f}, "
              f"avg={interaction_matrix[interaction_matrix > 0].mean():.2f}")
        
        return {
            'matrix': interaction_matrix,
            'user_encoder': user_encoder,
            'game_encoder': game_encoder,
            'raw_interactions': interactions,
            'n_users': n_users,
            'n_games': n_games
        }
    
    def create_training_pairs(self, interactions_processed, n_negatives=4):
        """Cria pares positivos/negativos para treinamento"""
        print("Criando pares de treinamento...")
        
        interaction_matrix = interactions_processed['matrix']
        n_users, n_games = interaction_matrix.shape
        
        positive_pairs = []
        negative_pairs = []
        
        # Encontrar pares positivos (interações existentes)
        user_indices, game_indices = np.where(interaction_matrix > 0)
        
        for u, g in tqdm(zip(user_indices, game_indices), 
                        total=len(user_indices), 
                        desc="Coletando positivos"):
            weight = interaction_matrix[u, g]
            positive_pairs.append({
                'user_idx': u,
                'game_idx': g,
                'label': 1,
                'weight': weight,
                'rating': weight  # Para Deep Learning
            })
        
        # Amostrar pares negativos
        for u in tqdm(range(n_users), desc="Amostrando negativos"):
            # Jogos com os quais o usuário interagiu
            positive_games = np.where(interaction_matrix[u] > 0)[0]
            
            if len(positive_games) == 0:
                continue
            
            # Todos os jogos possíveis
            all_games = np.arange(n_games)
            
            # Jogos negativos (não interagidos)
            negative_mask = ~np.isin(all_games, positive_games)
            negative_games = all_games[negative_mask]
            
            if len(negative_games) > 0:
                # Amostrar negativos proporcional ao número de positivos
                n_sample = min(len(negative_games), len(positive_games) * n_negatives)
                sampled_negatives = np.random.choice(
                    negative_games, n_sample, replace=False
                )
                
                for g in sampled_negatives:
                    negative_pairs.append({
                        'user_idx': u,
                        'game_idx': g,
                        'label': 0,
                        'weight': 0.1,  # Peso menor para negativos
                        'rating': 1.0  # Rating mínimo para negativos
                    })
        
        # Combinar e embaralhar
        all_pairs = positive_pairs + negative_pairs
        df_pairs = pd.DataFrame(all_pairs)
        df_pairs = df_pairs.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"✓ Pares de treinamento criados: {len(df_pairs)}")
        print(f"  Positivos: {len(positive_pairs)}")
        print(f"  Negativos: {len(negative_pairs)}")
        
        return df_pairs

# Função para executar todo o pipeline
def create_training_dataset(unified_data, output_dir="processed"):
    """Pipeline completo de pré-processamento"""
    
    preprocessor = SteamDataPreprocessor(unified_data)
    
    # 1. Processar jogos
    games_processed = preprocessor.preprocess_games(
        min_positive_ratio=60,
        min_total_reviews=1000
    )
    
    # 2. Processar interações
    interactions_processed = preprocessor.preprocess_interactions(
        min_user_interactions=5,
        min_game_interactions=10
    )
    
    # 3. Criar pares de treinamento
    training_pairs = preprocessor.create_training_pairs(
        interactions_processed,
        n_negatives=4
    )
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Salvar dados processados
    joblib.dump(games_processed, f'{output_dir}/games_processed.joblib')
    joblib.dump(interactions_processed, f'{output_dir}/interactions_processed.joblib')
    
    training_pairs.to_csv(f'{output_dir}/training_pairs.csv', index=False)
    
    print(f"\n✅ Dados salvos em: {output_dir}/")
    
    return {
        'games_processed': games_processed,
        'interactions_processed': interactions_processed,
        'training_pairs': training_pairs
    }