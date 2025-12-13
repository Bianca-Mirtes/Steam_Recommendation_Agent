# Criar um script setup.py
from src.data_loader import DataLoader

if __name__ == "__main__":
    print("🚀 Iniciando download dos datasets do Steam...")
    loader = DataLoader()
    loader.download_steam_datasets()
    
    print("\n📊 Carregando dados...")
    # Criar dataset unificado
    unified_data = loader.create_unified_dataset()

    # Filtar jogos populares para o modelo
    popular_games = loader.get_filtered_games(min_positive_ratio=75, min_reviews=500)

    # Salvar para uso futuro
    unified_data['games'].to_csv('data/processed/games_filtered.csv', index=False)
    unified_data['user_interactions'].to_csv('data/processed/user_game_matrix.csv', index=False)
    
    print(f"\n✅ Setup completo!")