import os
import kaggle
import pandas as pd
from pathlib import Path
import zipfile
from dotenv import load_dotenv
import yaml

class DataLoader:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Configurar paths
        self.raw_path = Path(self.config['paths']['data_raw'])
        self.raw_path.mkdir(parents=True, exist_ok=True)
        
        # Configurar Kaggle
        # Suas credenciais
        os.environ['KAGGLE_USERNAME'] = 'etidorfa'
        os.environ['KAGGLE_KEY'] = 'KGAT_0d416c88ccddc5f40387cb65e190441c'
    
    def download_steam_datasets(self):
        """Baixa todos os datasets relevantes do Steam"""
        datasets = [
            {
                'dataset': 'nikdavis/steam-store-games',
                'files': ['steam.csv']
            },
            {
                'dataset': 'tamber/steam-video-games',
                'files': ['steam-200k.csv']
            },
            {
                'dataset': 'antonkozyriev/game-recommendations-on-steam',
                'files': ['games.csv', 'users.csv', 'recommendations.csv']
            }
        ]
        
        for ds in datasets:
            print(f"Baixando {ds['dataset']}...")
            try:
                kaggle.api.dataset_download_files(
                    ds['dataset'],
                    path=self.raw_path,
                    unzip=True
                )
                print(f"✓ {ds['dataset']} baixado com sucesso!")
            except Exception as e:
                print(f"✗ Erro ao baixar {ds['dataset']}: {e}")
    
    def load_games_data(self):
        """Carrega dados dos jogos"""
        games_path = self.raw_path / "steam.csv"
        if games_path.exists():
            df = pd.read_csv(games_path, encoding='utf-8')
            print(f"Jogos carregados: {len(df)} registros")
            return df
        else:
            print("Arquivo de jogos não encontrado")
            return None
    
    def create_unified_dataset(self):
        """Cria um dataset unificado para treino do modelo"""
        
        # 1. Carregar dados básicos dos jogos
        steam_df = self.load_games_data()
        if steam_df is None:
            print("Erro: steam.csv não encontrado")
            return None
        
        # 2. Carregar e mesclar games.csv para mais informações
        games_df = self._load_games_extra()
        if games_df is not None:
            # Mesclar com base no appid
            steam_df = pd.merge(
                steam_df, 
                games_df,
                left_on='appid',
                right_on='app_id',
                how='left'
            )
            print(f"✓ Dados de games.csv mesclados")
        
        # 3. Processar interações usuário-jogo (steam-200k.csv)
        interactions_df = self.load_user_data()
        if interactions_df is not None:
            # Normalizar nomes dos jogos para mapear com appids
            interactions_df = self._map_game_names(interactions_df, steam_df)
            print(f"✓ Interações mapeadas: {len(interactions_df)} registros")
        
        # 4. Carregar reviews (recommendations.csv)
        reviews_df = self._load_recommendations()
        
        # 5. Combinar interações e reviews
        user_game_matrix = self._create_user_game_matrix(interactions_df, reviews_df, steam_df)
        
        # 6. Criar dataset final
        unified_data = {
            'games': steam_df,
            'user_interactions': user_game_matrix,
            'interactions_raw': interactions_df,
            'reviews': reviews_df
        }
        
        print(f"\n✅ Dataset unificado criado:")
        print(f"   - Jogos: {len(steam_df)}")
        print(f"   - Interações únicas: {len(user_game_matrix)}")
        if reviews_df is not None:
            print(f"   - Reviews: {len(reviews_df)}")
        
        return unified_data
    
    def _load_games_extra(self):
        """Carrega games.csv para informações extras"""
        games_path = self.raw_path / "games.csv"
        if games_path.exists():
            df = pd.read_csv(games_path)
            # Renomear colunas para consistência
            df = df.rename(columns={'app_id': 'app_id'})
            print(f"Games extras carregados: {len(df)}")
            return df
        return None
        
    def load_user_data(self):
        """Carrega dados de usuários"""
        # Dataset steam-200k.csv tem: user_id, game, behavior, value
        user_path = self.raw_path / "steam-200k.csv"
        if user_path.exists():
            df = pd.read_csv(user_path, header=None, 
                           names=['user_id', 'game', 'behavior', 'value', 'null'])
            df = df.drop('null', axis=1)
            print(f"Interações usuário-jogo: {len(df)} registros")
            return df
        else:
            print("Arquivo de interações não encontrado")
            return None
    
    def _map_game_names(self, interactions_df, games_df):
        """Mapeia nomes de jogos para appids"""
        # Criar dicionário de mapeamento nome->appid
        name_to_appid = {}
        for _, row in games_df.iterrows():
            name_to_appid[row['name'].lower()] = row['appid']
        
        # Tentar mapear cada jogo
        mapped_count = 0
        for idx, row in interactions_df.iterrows():
            game_name_lower = row['game'].lower()
            if game_name_lower in name_to_appid:
                interactions_df.at[idx, 'appid'] = name_to_appid[game_name_lower]
                mapped_count += 1
            else:
                # Tentar matching parcial para nomes similares
                for game_name, appid in name_to_appid.items():
                    if game_name in game_name_lower or game_name_lower in game_name:
                        interactions_df.at[idx, 'appid'] = appid
                        mapped_count += 1
                        break
        
        print(f"Mapeados {mapped_count}/{len(interactions_df)} jogos para appids")
        return interactions_df
    
    def _load_recommendations(self):
        """Carrega recommendations.csv"""
        rec_path = self.raw_path / "recommendations.csv"
        if rec_path.exists():
            # Carregar apenas um subset para não sobrecarregar
            df = pd.read_csv(rec_path, nrows=50000)
            print(f"Recommendations carregadas: {len(df)}")
            return df
        return None
    
    def _create_user_game_matrix(self, interactions_df, reviews_df, games_df):
        """Cria matriz usuário-jogo com features consolidadas"""
        
        # Filtrar apenas jogos que temos informações detalhadas
        valid_appids = set(games_df['appid'].unique())
        
        user_game_data = []
        
        # Processar interações de comportamento (steam-200k)
        if interactions_df is not None and 'appid' in interactions_df.columns:
            for _, row in interactions_df.iterrows():
                if row.get('appid') in valid_appids:
                    user_game_data.append({
                        'user_id': row['user_id'],
                        'appid': row['appid'],
                        'hours_played': row['value'] if row['behavior'] == 'play' else 0,
                        'purchased': 1 if row['behavior'] == 'purchase' else 0,
                        'source': 'interaction'
                    })
        
        # Processar reviews (recommendations.csv)
        if reviews_df is not None:
            for _, row in reviews_df.iterrows():
                if row['app_id'] in valid_appids:
                    user_game_data.append({
                        'user_id': row['user_id'],
                        'appid': row['app_id'],
                        'hours_played': row.get('hours', 0),
                        'is_recommended': 1 if row.get('is_recommended', False) == True else 0,
                        'source': 'review'
                    })
        
        # Criar DataFrame consolidado
        if user_game_data:
            df = pd.DataFrame(user_game_data)
            
            # Agrupar por usuário-jogo (pode ter múltiplas entradas)
            grouped = df.groupby(['user_id', 'appid']).agg({
                'hours_played': 'sum',
                'purchased': 'max',
                'is_recommended': 'max',
                'source': lambda x: ','.join(set(x))
            }).reset_index()
            
            # Adicionar rating implícito baseado em horas jogadas
            def calculate_implicit_rating(hours):
                if hours <= 1:
                    return 0
                elif hours <= 10:
                    return 1
                elif hours <= 50:
                    return 2
                elif hours <= 100:
                    return 3
                else:
                    return 4
            
            grouped['implicit_rating'] = grouped['hours_played'].apply(calculate_implicit_rating)
            
            # Filtrar usuários com poucas interações (ruído)
            user_counts = grouped['user_id'].value_counts()
            active_users = user_counts[user_counts >= 3].index
            grouped = grouped[grouped['user_id'].isin(active_users)]
            
            print(f"Matriz usuário-jogo criada: {len(grouped)} interações únicas")
            return grouped
        
        return pd.DataFrame()
    
    def get_filtered_games(self, min_positive_ratio=70, min_reviews=100):
        """Filtra jogos por popularidade e qualidade"""
        games_df = self.load_games_data()
        
        if games_df is None:
            return None
        
        # Filtrar jogos em inglês (se disponível)
        if 'english' in games_df.columns:
            games_df = games_df[games_df['english'] == 1]
        
        # Filtrar por ratings positivos (se disponível em games.csv)
        if 'positive_ratings' in games_df.columns and 'negative_ratings' in games_df.columns:
            games_df['total_reviews'] = games_df['positive_ratings'] + games_df['negative_ratings']
            games_df['positive_ratio'] = games_df['positive_ratings'] / games_df['total_reviews'] * 100
            games_df = games_df[games_df['total_reviews'] >= min_reviews]
            games_df = games_df[games_df['positive_ratio'] >= min_positive_ratio]
        
        # Ordenar por popularidade
        if 'positive_ratings' in games_df.columns:
            games_df = games_df.sort_values('positive_ratings', ascending=False)
        
        print(f"Jogos filtrados: {len(games_df)} (min {min_reviews} reviews, {min_positive_ratio}% positivos)")
        return games_df