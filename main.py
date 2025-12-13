import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.data_loader import DataLoader
from src.preprocessor import create_training_dataset
from src.game_embedder import create_game_embeddings_pipeline
from src.profile_analyzer import ProfileAnalyzer
import pickle
import json
import faiss

def main():
    print("=" * 60)
    print("🎮 STEAM RECOMMENDATION AGENT - PIPELINE COMPLETO")
    print("=" * 60)
    
    # 1. Carregar dados
    print("\n1. 📥 Carregando dados...")
    loader = DataLoader()
    unified_data = loader.create_unified_dataset()

    if unified_data is None:
        print("❌ Erro ao criar dataset unificado")
        return
    
    # 2. Pré-processamento dos dados
    print("\n2. 🧹 Pré-processando dados...")
    processed_data = create_training_dataset(unified_data, output_dir="processed_data")

    games_processed = processed_data['games_processed']
    games_df = games_processed['games']
    interactions_processed = processed_data['interactions_processed']
    interactions_matrix = interactions_processed['matrix']
    training_pairs = processed_data['training_pairs']
    
    print(f"   • Jogos após pré-processamento: {len(games_df)}")
    print(f"   • Matriz de interações: {interactions_matrix.shape}")
    print(f"   • Pares de treino: {len(training_pairs)}")
    
    # Features numéricas disponíveis
    numeric_features = games_processed.get('numeric_features', [])
    
    # 3. Criar embeddings dos jogos
    print("\n3. 🤖 Criando embeddings dos jogos...")
    embedder, index, embed_result = create_game_embeddings_pipeline(
        games_df,
        output_dir="data/embeddings",
        numeric_features=numeric_features,
        use_hybrid=True
    )
    
    # Verificar qual tipo de embedding foi criado
    if 'hybrid_embeddings' in embed_result:
        embeddings = embed_result['hybrid_embeddings']
        print(f"   • Embeddings híbridos: {embeddings.shape}")
    else:
        embeddings = embed_result['text_embeddings']
        print(f"   • Embeddings de texto: {embeddings.shape}")
    
    print(f"   • Índice FAISS: {index.ntotal} vetores")
    
    # 4. Treinar modelo colaborativo (ProfileAnalyzer)
    print("\n4. 🧠 Treinando modelo de perfil colaborativo...")
    
    analyzer = ProfileAnalyzer(n_components=50, n_neighbors=20, min_interactions=3)
    
    # Usar as interações processadas para treinar
    interactions_df = interactions_processed['raw_interactions'].copy()
    
    # Garantir que temos a coluna de rating
    if 'implicit_rating' not in interactions_df.columns:
        print("   ⚠️ Coluna 'implicit_rating' não encontrada, criando...")
        # Criar rating baseado em horas jogadas
        interactions_df['implicit_rating'] = interactions_df['hours_played'].apply(
            lambda x: min(5.0, max(1.0, np.log1p(x) / 5.0)) if x > 0 else 1.0
        )
    
    # Treinar modelo colaborativo
    try:
        profile_model = analyzer.train_collaborative_model(
            interactions_df,
            user_col='user_id',
            game_col='appid',
            rating_col='implicit_rating'
        )
        print(f"   ✓ Modelo treinado com {len(analyzer.user_encoder)} usuários e {len(analyzer.game_encoder)} jogos")
    except Exception as e:
        print(f"   ❌ Erro ao treinar modelo: {e}")
        return
    
    # 5. Salvar tudo
    print("\n5. 💾 Salvando modelos e dados...")
    
    # Salvar embeddings (usando o método do embedder)
    embedder.save_all("data/embeddings")
    print("   ✓ Embeddings salvos")
    
    # Salvar índice FAISS separadamente
    faiss_path = "data/embeddings/game_index.faiss"
    faiss.write_index(index, faiss_path)
    print(f"   ✓ Índice FAISS salvo: {faiss_path}")
    
    # Salvar modelo colaborativo
    analyzer.save_model("models/profile_model.pkl")
    print("   ✓ Modelo colaborativo salvo")
    
    # Salvar dados processados
    games_df.to_pickle("data/processed/games_processed.pkl")
    
    # Salvar interações processadas
    with open("data/processed/interactions_processed.pkl", "wb") as f:
        pickle.dump(interactions_processed, f)
    
    # Salvar training pairs
    training_pairs.to_pickle("data/processed/training_pairs.pkl")
    
    # Salvar mapeamentos e metadados
    metadata = {
        'num_games': len(games_df),
        'num_users': interactions_processed.get('n_users', 0),
        'num_games_interactions': interactions_processed.get('n_games', 0),
        'embedding_dim': embeddings.shape[1] if embeddings is not None else 0,
        'collaborative_model': {
            'n_components': analyzer.n_components,
            'n_neighbors': analyzer.n_neighbors,
            'num_users': len(analyzer.user_encoder) if analyzer.user_encoder else 0,
            'num_games': len(analyzer.game_encoder) if analyzer.game_encoder else 0
        },
        'data_sources': {
            'steam_csv': 'nikdavis/steam-store-games',
            'steam_200k': 'tamber/steam-video-games',
            'game_recommendations': 'antonkozyriev/game-recommendations-on-steam'
        }
    }
    
    with open("data/processed/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETADO COM SUCESSO!")
    print("=" * 60)
    print(f"   📊 ESTATÍSTICAS FINAIS:")
    print(f"   • Jogos: {len(games_df)}")
    print(f"   • Usuários ativos: {metadata['collaborative_model']['num_users']}")
    print(f"   • Jogos com interações: {metadata['collaborative_model']['num_games']}")
    print(f"   • Dimensão embeddings: {metadata['embedding_dim']}")
    print(f"   • Matriz interações: {interactions_matrix.shape}")
    
    # Calcular densidade da matriz
    if interactions_matrix is not None:
        density = (interactions_matrix > 0).sum() / (interactions_matrix.shape[0] * interactions_matrix.shape[1])
        print(f"   • Densidade matriz: {density:.4%}")
    
    print("\n🎯 Agora execute: streamlit run app.py")
    print("=" * 60)

if __name__ == "__main__":
    main()