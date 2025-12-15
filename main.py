import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.data_loader import DataLoader
from src.preprocessor import create_training_dataset
from src.game_embedder import create_game_embeddings_pipeline
from src.profile_analyzer_deep  import DeepProfileAnalyzer, ModelConfig
import pickle
import json
import faiss
import numpy as np
import pandas as pd
from datetime import datetime
import torch

def main():
    print("=" * 60)
    print("🎮 STEAM RECOMMENDATION AGENT - PIPELINE DEEP LEARNING")
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
    print(f"   • Features numéricas: {numeric_features}")
    
    # 3. Criar embeddings dos jogos (para busca semântica)
    print("\n3. 🤖 Criando embeddings semânticos dos jogos...")
    embedder, index, embed_result = create_game_embeddings_pipeline(
        games_df,
        output_dir="data/embeddings",
        numeric_features=numeric_features,
        use_hybrid=False
    )
    
    embeddings = embed_result['text_embeddings']
    print(f"   • Embeddings de texto: {embeddings.shape}")
    print(f"   • Índice FAISS: {index.ntotal} vetores")
    
    # Preparar interações para treinamento
    interactions_df = interactions_processed['raw_interactions'].copy()
    
    # Garantir que temos a coluna de rating
    if 'implicit_rating' not in interactions_df.columns:
        print("   ⚠️ Coluna 'implicit_rating' não encontrada, criando...")
        # Criar rating baseado em horas jogadas com normalização
        interactions_df['implicit_rating'] = interactions_df['hours_played'].apply(
            lambda x: 1.0 + 4.0 * (np.log1p(x) / np.log1p(interactions_df['hours_played'].max())) 
            if x > 0 else 1.0
        ).clip(1.0, 5.0)
    
    # Verificar tamanho dos dados para ajustar parâmetros
    n_users = interactions_df['user_id'].nunique()
    n_games = interactions_df['appid'].nunique()
    
    print(f"   • Usuários únicos: {n_users}")
    print(f"   • Jogos únicos: {n_games}")
    print(f"   • Total de interações: {len(interactions_df)}")

    # 4. Preparar metadados dos jogos para Deep Learning
    print("\n4. 📊 Preparando metadados para Deep Learning...")
    
    # Na parte do treinamento:
    config = ModelConfig(
        embed_dim=96,           # Ajustado para seu dataset (3160 usuários)
        num_heads=4,            # Multi-head attention
        num_layers=2,           # 2 camadas Transformer
        ff_dim=256,             # Feed-forward dimension
        dropout=0.2,            # Dropout para regularização
        n_neighbors=20,         # Para KNN
        min_interactions=3,     # Filtro mínimo
        use_attention=True      # Usar atenção multi-head (Transformer)
    )

    # Ajustar parâmetros baseado no tamanho dos dados
    if n_users < 1000: # dataset muito pequeno
        config.embed_dim = 64
        config.ff_dim = 128
        config.num_layers = 1
        epochs = 10
        batch_size = 64
    elif n_users < 10000: 
        config.embed_dim = 96
        config.ff_dim = 192
        epochs = 15
        batch_size = 128
    else:
        config.embed_dim = 128
        config.ff_dim = 256
        epochs = 20
        batch_size = 256

    analyzer = DeepProfileAnalyzer(config)
    
    print(f"   • Configuração final:")
    print(f"     - Embedding dim: {config.embed_dim}")
    print(f"     - Nº de camadas: {config.num_layers}")
    print(f"     - Épocas: {epochs}")
    print(f"     - Batch size: {batch_size}")

    # 5. Treinar modelo Deep Learning
    print("\n5. 🧠 Treinando modelo Deep Learning (Transformer)...")
    
   # Treinar modelo TRANSFORMER
    try:
        start_time = datetime.now()
        
        profile_model = analyzer.train_collaborative_model(
            interactions_df=interactions_df,
            user_col='user_id',
            game_col='appid',
            rating_col='implicit_rating',
            epochs=epochs,
            batch_size=batch_size
        )
        
        training_time = (datetime.now() - start_time).total_seconds()
        print(f"   ✓ Modelo treinado em {training_time:.1f} segundos")
        print(f"   ✓ Usuários no modelo: {len(analyzer.user_encoder)}")
        print(f"   ✓ Jogos no modelo: {len(analyzer.game_encoder)}")
        
        # Extrair e salvar estatísticas do treinamento
        if hasattr(analyzer, 'training_history'):
            print(f"   ✓ Loss final: {analyzer.training_history['train_loss'][-1]:.4f}")
            if 'val_loss' in analyzer.training_history:
                print(f"   ✓ Val loss final: {analyzer.training_history['val_loss'][-1]:.4f}")
        
    except Exception as e:
        print(f"   ❌ Erro ao treinar modelo Deep Learning: {e}")
        import traceback
        traceback.print_exc()
        print("\n   ⚠️ Tentando fallback para modelo clássico...")
        
        # Fallback para modelo clássico
        from src.profile_analyzer_basic import ProfileAnalyzer
        analyzer = ProfileAnalyzer(n_components=50, n_neighbors=20, min_interactions=3)
        profile_model = analyzer.train_collaborative_model(
            interactions_df,
            user_col='user_id',
            game_col='appid',
            rating_col='implicit_rating'
        )
        print(f"   ✓ Modelo clássico treinado como fallback")
    
    # 6. Avaliação rápida do modelo
    print("\n6. 📈 Avaliando modelo...")
    
    try:
        # Testar com alguns usuários de exemplo
        test_users = interactions_df['user_id'].unique()[:5]
        
        for user_id in test_users:
            profile = analyzer.get_user_profile(user_id)
            if profile is not None:
                similar = analyzer.get_similar_users(user_id, n_similar=3)
                print(f"   • {user_id}: perfil {profile.shape}, similares: {len(similar)}")
                
        # Verificar qualidade dos embeddings
        if analyzer.user_profiles is not None:
            profile_norms = np.linalg.norm(analyzer.user_profiles, axis=1)
            print(f"   • Norma média dos perfis: {profile_norms.mean():.3f} ± {profile_norms.std():.3f}")
            
    except Exception as e:
        print(f"   ⚠️ Avaliação parcial: {e}")
    
    # 7. Salvar tudo
    print("\n7. 💾 Salvando modelos e dados...")
    
    # Criar diretórios se não existirem
    Path("data/embeddings").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)
    
    # Salvar embeddings semânticos (FAISS)
    faiss_path = "data/embeddings/game_index.faiss"
    faiss.write_index(index, faiss_path)
    print(f"   ✓ Índice FAISS salvo: {faiss_path}")
    
    # Salvar metadados dos embeddings
    embedder.save_all("data/embeddings")
    print("   ✓ Embeddings semânticos salvos")
    
    # Salvar modelo Deep Learning
    model_path = "models/deep_profile_model.pkl"
    analyzer.save_model(model_path)
    print(f"   ✓ Modelo Deep Learning salvo: {model_path}")
    
    # Salvar também em formato PyTorch para facilidade
    torch_path = "models/deep_profile_model.pth"
    if hasattr(analyzer, 'model') and analyzer.model is not None:
        torch.save({
            'model_state_dict': analyzer.model.state_dict(),
            'config': analyzer.config,
            'user_encoder': analyzer.user_encoder,
            'game_encoder': analyzer.game_encoder
        }, torch_path)
        print(f"   ✓ Modelo PyTorch salvo: {torch_path}")
    
    # Salvar dados processados
    games_df.to_pickle("data/processed/games_processed.pkl")
    
    # Salvar interações processadas
    with open("data/processed/interactions_processed.pkl", "wb") as f:
        pickle.dump(interactions_processed, f)
    
    # Salvar training pairs
    training_pairs.to_pickle("data/processed/training_pairs.pkl")
    
    # Salvar configuração do modelo
    config_dict = {
        'embed_dim': config.embed_dim,
        'num_heads': config.num_heads,
        'num_layers': config.num_layers,
        'ff_dim': config.ff_dim,
        'dropout': config.dropout,
        'n_neighbors': config.n_neighbors,
        'min_interactions': config.min_interactions,
        'use_attention': config.use_attention,
        'training_params': {
            'epochs': epochs,
            'batch_size': batch_size
        }
    }
    
    with open("models/model_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    
    # Salvar mapeamentos e metadados completos
    metadata = {
        'num_games': len(games_df),
        'num_users': n_users,
        'num_games_interactions': n_games,
        'embedding_dim': embeddings.shape[1] if embeddings is not None else 0,
        'deep_learning_model': {
            'num_users': len(analyzer.user_encoder) if analyzer.user_encoder else 0,
            'num_games': len(analyzer.game_encoder) if analyzer.game_encoder else 0,
            'user_embedding_dim': analyzer.user_profiles.shape[1] if hasattr(analyzer, 'user_profiles') and analyzer.user_profiles is not None else 0,
            'game_embedding_dim': analyzer.game_profiles.shape[1] if hasattr(analyzer, 'game_profiles') and analyzer.game_profiles is not None else 0,
            'model_type': 'DeepProfileAnalyzer' if hasattr(analyzer, 'model') else 'ProfileAnalyzer'
        },
        'data_sources': {
            'steam_csv': 'nikdavis/steam-store-games',
            'steam_200k': 'tamber/steam-video-games',
            'game_recommendations': 'antonkozyriev/game-recommendations-on-steam'
        },
        'processing_date': datetime.now().isoformat(),
        'training_time_seconds': training_time if 'training_time' in locals() else None
    }
    
    with open("data/processed/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 60)
    print("✅ PIPELINE DEEP LEARNING COMPLETADO COM SUCESSO!")
    print("=" * 60)
    print(f"📊 ESTATÍSTICAS FINAIS:")
    print(f"   • Jogos no catálogo: {len(games_df)}")
    print(f"   • Usuários ativos: {metadata['deep_learning_model']['num_users']}")
    print(f"   • Jogos com interações: {metadata['deep_learning_model']['num_games']}")
    print(f"   • Dimensão embeddings semânticos: {metadata['embedding_dim']}")
    print(f"   • Dimensão embeddings colaborativos: {metadata['deep_learning_model']['user_embedding_dim']}")
    print(f"   • Matriz interações: {interactions_matrix.shape}")
    
    # Calcular densidade da matriz
    if interactions_matrix is not None:
        density = (interactions_matrix > 0).sum() / (interactions_matrix.shape[0] * interactions_matrix.shape[1])
        print(f"   • Densidade matriz: {density:.4%}")
    
    # Informações sobre o modelo
    print(f"\n🤖 MODELO TREINADO:")
    print(f"   • Tipo: {metadata['deep_learning_model']['model_type']}")
    print(f"   • Atenção Multi-head: {'Sim' if config.use_attention else 'Não'}")
    print("=" * 60)

if __name__ == "__main__":
    main()