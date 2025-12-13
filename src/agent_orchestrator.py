import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass
from enum import Enum
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecommendationStrategy(Enum):
    """Estratégias de recomendação"""
    COLLABORATIVE = "collaborative"
    CONTENT_BASED = "content_based"
    HYBRID = "hybrid"
    CONTEXTUAL = "contextual"

@dataclass
class Recommendation:
    """Classe para representar uma recomendação"""
    game_id: Any  # Agora será appid (int)
    game_name: str
    score: float
    rationale: str
    match_features: Dict[str, float]
    source: str
    metadata: Dict[str, Any]

class RecommendationAgent:
    def __init__(self, 
                 profile_analyzer,
                 game_embedder,
                 prompt_interpreter,
                 games_df,
                 strategy: RecommendationStrategy = RecommendationStrategy.HYBRID):
        """
        Agente orquestrador de recomendações
        
        Args:
            profile_analyzer: Instância do ProfileAnalyzer (atualizado)
            game_embedder: Instância do GameEmbedder (atualizado)
            prompt_interpreter: Instância do PromptInterpreter
            games_df: DataFrame com informações dos jogos (com coluna 'appid')
            strategy: Estratégia de recomendação
        """
        self.profile_analyzer = profile_analyzer
        self.game_embedder = game_embedder
        self.prompt_interpreter = prompt_interpreter
        self.games_df = games_df
        self.strategy = strategy
        
        # Configurações
        self.weights = {
            'collaborative': 0.5, 
            'content': 0.5,
            'contextual': 0.2
        }
        
        self.diversity_penalty = 0.3
        self.min_score_threshold = 0.2  # Reduzido para incluir mais recomendações
        
        logger.info(f"Agente inicializado com estratégia: {strategy.value}")
        logger.info(f"Jogos disponíveis: {len(games_df)}")
        
    def recommend(self, 
              user_profile: Dict[str, Any],
              user_prompt: str,
              user_library: List[Any],  # Agora lista de appids
              n_recommendations: int = 5) -> List[Recommendation]:
        """
        Gera recomendações baseadas no perfil e prompt
        
        Args:
            user_profile: Perfil do usuário (com 'user_id')
            user_prompt: Prompt textual do usuário
            user_library: Lista de appids que o usuário já possui
            n_recommendations: Número de recomendações a retornar
        """
        logger.info(f"Gerando recomendações para usuário {user_profile.get('user_id', 'unknown')}")
        logger.info(f"Biblioteca do usuário: {len(user_library)} jogos")
        
        try:
            # 1. Interpretar prompt do usuário
            prompt_query = self.prompt_interpreter.interpret_to_query(user_prompt)
            logger.info(f"Prompt interpretado: {prompt_query['primary_categories']}")
            
            # 2. Gerar recomendações de cada fonte
            all_recommendations = []
            
            # Recomendações colaborativas
            if self.strategy in [RecommendationStrategy.COLLABORATIVE, 
                            RecommendationStrategy.HYBRID]:
                logger.info("Gerando recomendações colaborativas...")
                collab_recs = self._get_collaborative_recommendations(
                    user_profile, user_library
                )
                logger.info(f"Recomendações colaborativas geradas: {len(collab_recs)}")
                all_recommendations.extend(collab_recs)
            
            # Recomendações baseadas em conteúdo
            if self.strategy in [RecommendationStrategy.CONTENT_BASED,
                            RecommendationStrategy.HYBRID,
                            RecommendationStrategy.CONTEXTUAL]:
                logger.info("Gerando recomendações baseadas em conteúdo...")
                content_recs = self._get_content_based_recommendations(
                    prompt_query, user_library
                )
                logger.info(f"Recomendações baseadas em conteúdo geradas: {len(content_recs)}")
                all_recommendations.extend(content_recs)
            
            logger.info(f"Total de recomendações coletadas: {len(all_recommendations)}")
            
            if not all_recommendations:
                logger.warning("Nenhuma recomendação gerada!")
                return []
            
            # DEBUG: Verificar tipos de scores
            for i, rec in enumerate(all_recommendations[:3]):
                logger.info(f"Recomendação {i}: {rec.game_name}, score={rec.score}, source={rec.source}")
            
            # 3. Combinar e rankear recomendações
            if self.strategy == RecommendationStrategy.HYBRID:
                logger.info("Aplicando ranking híbrido...")
                final_recommendations = self._hybrid_ranking(
                    all_recommendations, prompt_query, user_profile
                )
            else:
                logger.info("Aplicando ranking simples...")
                final_recommendations = self._simple_ranking(all_recommendations)
            
            logger.info(f"Ranking aplicado: {len(final_recommendations)} recomendações")
            
            # 4. Aplicar diversidade e filtrar
            final_recommendations = self._apply_diversity(final_recommendations)
            final_recommendations = self._filter_existing(final_recommendations, user_library)
            
            # 5. Limitar número de recomendações
            final_recommendations = final_recommendations[:n_recommendations]
            
            # 6. Gerar explicações para cada recomendação
            logger.info("Gerando explicações...")
            for rec in final_recommendations:
                rec.rationale = self.prompt_interpreter.get_recommendation_rationale(
                    user_prompt,
                    rec.game_name,
                    rec.match_features
                )
            
            logger.info(f"Geradas {len(final_recommendations)} recomendações finais")
            return final_recommendations
        
        except Exception as e:
            logger.error(f"Erro ao gerar recomendações: {str(e)}")
            logger.exception(e)  # Log completo do traceback
            return []
    
    def _get_collaborative_recommendations(self, user_profile, user_library):
        """
        Gera recomendações baseadas em filtragem colaborativa - COM FALLBACK
        """
        recommendations = []
        
        try:
            user_id = user_profile.get('user_id')
            
            # VERIFICAR se o usuário está no encoder
            if hasattr(self.profile_analyzer, 'user_encoder') and user_id in self.profile_analyzer.user_encoder:
                logger.info(f"Usuário {user_id} está no encoder, usando filtragem colaborativa")
                
                if hasattr(self.profile_analyzer, 'recommend_from_profile'):
                    collab_df = self.profile_analyzer.recommend_from_profile(
                        user_id=user_id,
                        games_df=self.games_df,
                        top_n=15,
                        exclude_played=True,
                        user_library=user_library
                    )
                    
                    for _, row in collab_df.iterrows():
                        score = float(row.get('score', row.get('combined_score', 0)))
                        
                        rec = Recommendation(
                            game_id=row['appid'],
                            game_name=row.get('name', 'Desconhecido'),
                            score=score,
                            rationale="",
                            match_features={
                                'collaborative_score': score,
                                'source': 'collaborative'
                            },
                            source="collaborative",
                            metadata=row.to_dict()
                        )
                        recommendations.append(rec)
            
            else:
                # FALLBACK: Usar jogos populares entre usuários similares
                logger.info(f"Usuário {user_id} não está no encoder, usando fallback")
                
                # Buscar jogos populares no geral
                if hasattr(self, 'games_df') and not self.games_df.empty:
                    # Filtrar jogos bem avaliados e populares
                    popular_games = self.games_df.copy()
                    
                    # Garantir que temos as colunas necessárias
                    if 'positive_ratio' in popular_games.columns and 'positive_ratings' in popular_games.columns:
                        # Filtrar por avaliação
                        popular_games = popular_games[
                            (popular_games['positive_ratio'] >= 70) &
                            (popular_games['positive_ratings'] >= 1000)
                        ]
                        
                        # Ordenar por popularidade
                        popular_games = popular_games.sort_values('positive_ratings', ascending=False)
                        
                        # Pegar top 20 (aumentado de 10 para 20)
                        for _, row in popular_games.head(20).iterrows():
                            if row['appid'] not in user_library:
                                score = 0.7  # Aumentado de 0.5 para 0.7 (score base para jogos populares)
                                
                                # Ajustar score baseado na avaliação
                                positive_ratio = row.get('positive_ratio', 0)
                                if positive_ratio > 90:
                                    score = 0.9
                                elif positive_ratio > 80:
                                    score = 0.8
                                elif positive_ratio > 70:
                                    score = 0.7
                                
                                rec = Recommendation(
                                    game_id=row['appid'],
                                    game_name=row.get('name', 'Desconhecido'),
                                    score=score,
                                    rationale="",
                                    match_features={
                                        'collaborative_score': score,
                                        'positive_ratio': positive_ratio,
                                        'source': 'popularity_fallback'
                                    },
                                    source="collaborative",
                                    metadata=row.to_dict()
                                )
                                recommendations.append(rec)
                                
                                if len(recommendations) >= 20:
                                    break
                
                logger.info(f"Fallback colaborativo: {len(recommendations)} jogos populares")
                        
        except Exception as e:
            logger.warning(f"Erro em recomendações colaborativas: {e}")
        
        return recommendations
    
    def _get_content_based_recommendations(self, prompt_query, user_library):
        """
        Gera recomendações baseadas em conteúdo/semântica
        """
        recommendations = []
        
        try:
            # Busca semântica baseada no prompt
            if hasattr(self.game_embedder, 'search_similar_games'):
                
                # Usar FAISS index que já deve estar carregado no embedder
                if not hasattr(self, 'faiss_index'):
                    # Tentar carregar índice se existir
                    import faiss
                    try:
                        self.faiss_index = faiss.read_index("data/embeddings/game_index.faiss")
                        logger.info("Índice FAISS carregado do arquivo")
                    except:
                        # Construir índice a partir dos embeddings do cache
                        logger.info("Construindo índice FAISS a partir dos embeddings...")
                        embeddings_list = []
                        appid_list = []
                        
                        for appid, cache_data in self.game_embedder.embedding_cache.items():
                            if 'text_embedding' in cache_data:
                                embeddings_list.append(cache_data['text_embedding'])
                                appid_list.append(appid)
                        
                        if embeddings_list:
                            embeddings_array = np.vstack(embeddings_list)
                            self.faiss_index = self.game_embedder.build_faiss_index(
                                embeddings_array, index_type="Flat"
                            )
                        else:
                            logger.error("Nenhum embedding encontrado no cache")
                            return []
                
                # CORREÇÃO: Usar parâmetro correto 'query' em vez de 'query_text'
                results_df = self.game_embedder.search_similar_games(
                    query=prompt_query['text_query'],  
                    games_df=self.games_df,
                    index=self.faiss_index,
                    top_k=100,
                    search_by='text',  
                    threshold=0.1      
                )
                
                if results_df.empty:
                    logger.warning("Nenhum resultado encontrado na busca por conteúdo")
                    return []
                
                for _, row in results_df.iterrows():
                    # Obter appid
                    appid = row['appid']

                    # FILTRO 1: Já está na biblioteca do usuário
                    if appid in user_library:
                        continue
                    
                    # FILTRO 2: Qualidade mínima (positive_ratio)
                    positive_ratio = row.get('positive_ratio', 0)
                    if isinstance(positive_ratio, (int, float)):
                        if positive_ratio < 60:  # Mínimo 60% de avaliações positivas
                            continue
                    
                    # FILTRO 3: Tem reviews suficientes?
                    total_reviews = 0
                    if 'positive_ratings' in row and 'negative_ratings' in row:
                        positive = float(row.get('positive_ratings', 0))
                        negative = float(row.get('negative_ratings', 0))
                        total_reviews = positive + negative
                    
                    if total_reviews < 100:  # Mínimo 100 reviews
                        continue
                    # Calcular score contextual
                    context_score = self._calculate_context_score(row, prompt_query)

                    game_name = row['name']
                    similarity = float(row.get('similarity', 0))
                    
                    # Score combinado
                    combined_score = similarity * context_score

                    # BOOST para jogos populares/bem avaliados
                    popularity_boost = 1.0
                    if positive_ratio > 80:
                        popularity_boost *= 1.2  # +20% para jogos muito bem avaliados
                    elif positive_ratio > 70:
                        popularity_boost *= 1.1  # +10% para jogos bem avaliados
                    
                    if total_reviews > 10000:
                        popularity_boost *= 1.15  # +15% para jogos populares
                    elif total_reviews > 1000:
                        popularity_boost *= 1.08  # +8% para jogos conhecidos
                    
                    final_score = combined_score * popularity_boost
                    
                    rec = Recommendation(
                        game_id=appid,
                        game_name=game_name,
                        score=final_score,
                        rationale="",
                        match_features={
                            'semantic_similarity': similarity,
                            'context_match': context_score,
                            'popularity_boost': popularity_boost,
                            'positive_ratio': positive_ratio,
                            'total_reviews': total_reviews,
                            'categories': prompt_query['primary_categories']
                        },
                        source="content_based",
                        metadata=row.to_dict()
                    )
                    recommendations.append(rec)
                    
        except Exception as e:
            logger.warning(f"Erro em recomendações baseadas em conteúdo: {e}")
        
        return recommendations
    
    def _calculate_context_score(self, game_row, prompt_query):
        """
        Calcula score baseado no contexto do prompt - VERSÃO SUPER MELHORADA
        """
        score = 1.0
        
        # 1. MATCH DE GÊNEROS (mais importante)
        game_genres = str(game_row.get('genres', '')).lower()
        game_categories = str(game_row.get('categories', '')).lower()
        
        prompt_features = prompt_query.get('features', {})
        prompt_categories = prompt_query.get('primary_categories', [])
        
        genre_match_score = 0
        
        # Verificar match direto com gêneros
        for category in prompt_categories:
            category_lower = category.lower()
            
            # Match direto no gênero
            if category_lower in game_genres:
                genre_match_score += 1.0
            
            # Match com sinônimos
            elif category_lower in ['ação', 'action']:
                if any(genre in game_genres for genre in ['action', 'shooter', 'fps']):
                    genre_match_score += 0.9
            elif category_lower in ['terror', 'horror']:
                if any(term in game_genres for term in ['horror']):
                    genre_match_score += 0.9
                elif 'horror' in game_categories:
                    genre_match_score += 0.8
            elif category_lower in ['rpg']:
                if 'rpg' in game_genres:
                    genre_match_score += 0.9
            elif category_lower in ['multijogador', 'multiplayer']:
                if 'multi-player' in game_categories or 'online multiplayer' in game_categories:
                    genre_match_score += 0.9
            elif category_lower in ['cooperativo', 'coop']:
                if 'co-op' in game_categories:
                    genre_match_score += 0.9
        
        # Boost para match de gênero
        if genre_match_score > 0:
            score *= 1.0 + (genre_match_score * 0.4)  # Boost significativo!
        
        # 2. MATCH COM FEATURES DO PROMPT NO TEXTO DO JOGO
        game_text = ""
        if 'combined_text' in game_row:
            game_text = str(game_row['combined_text']).lower()
        elif 'short_description' in game_row:
            game_text = str(game_row['short_description']).lower()
        
        game_name_lower = str(game_row.get('name', '')).lower()
        full_game_text = game_name_lower + " " + game_text
        
        feature_match_score = 0
        for feature, feature_value in prompt_features.items():
            if isinstance(feature_value, (int, float)) and feature_value > 0.5:
                feature_lower = str(feature).lower()
                
                # Verificar se feature está no texto do jogo
                if feature_lower in full_game_text:
                    feature_match_score += feature_value * 0.5
        
        if feature_match_score > 0:
            score *= 1.0 + feature_match_score
        
        # 3. BOOST ESPECIAL para tipos específicos de jogos
        # Se busca por terror + multijogador, dar boost para survival horror
        if 'terror' in prompt_categories and 'multijogador' in prompt_categories:
            horror_multi_keywords = ['survival', 'co-op', 'multiplayer', 'horror', 'scary']
            if any(keyword in full_game_text for keyword in horror_multi_keywords[:3]):
                score *= 1.3  # Boost de 30%!
        
        # Se busca por FPS, dar boost para jogos populares de FPS
        if 'ação' in prompt_categories or 'fps' in prompt_categories:
            fps_keywords = ['fps', 'first person', 'shooter', 'call of duty', 'battlefield', 'halo']
            if any(keyword in full_game_text for keyword in fps_keywords):
                score *= 1.25
        
        # 4. BOOST para jogos conhecidos por serem bons no gênero
        known_games_boost = {
            'terror': ['phasmophobia', 'left 4 dead', 'resident evil', 'silent hill', 'dead by daylight', 'devour'],
            'ação': ['call of duty', 'battlefield', 'halo', 'doom', 'overwatch', 'counter-strike'],
            'rpg': ['the witcher', 'skyrim', 'final fantasy', 'dragon age', 'mass effect'],
            'estratégia': ['civilization', 'age of empires', 'starcraft', 'total war'],
            'aventura': ['tomb raider', 'uncharted', 'the last of us', 'god of war'],
            'simulação': ['sims', 'simcity', 'cities skylines', 'farming simulator'],
            'esportes': ['fifa', 'nba 2k', 'forza', 'gran turismo', 'rocket league'],
            'multijogador': ['among us', 'fall guys', 'team fortress', 'left 4 dead']
        }
        
        for genre, known_games in known_games_boost.items():
            if genre in prompt_categories:
                for known_game in known_games:
                    if known_game in full_game_text:
                        score *= 1.4  # Boost de 40% para jogos conhecidos no gênero!
                        break
        
        if 'price' in game_row:
            price = game_row['price']
            positive_ratio = game_row.get('positive_ratio', 0)
            total_reviews = game_row.get('positive_ratings', 0) + game_row.get('negative_ratings', 0)
            
            if isinstance(price, (int, float)) and price > 0 and isinstance(positive_ratio, (int, float)):
                # Boost para jogos baratos e bem avaliados
                if price < 20 and positive_ratio > 70:
                    score *= 1.3
                elif price < 10 and positive_ratio > 60:
                    score *= 1.4
                
                # Boost para jogos muito populares (muitas reviews)
                if total_reviews > 10000:
                    score *= 1.5
                elif total_reviews > 1000:
                    score *= 1.3
        
        # 6. Limitar boost máximo
        return min(score, 4.0)
    
    def _hybrid_ranking(self, recommendations, prompt_query, user_profile):
        # SUBSTITUA todo o cálculo de score por:
        scored_recs = {}
        
        for rec in recommendations:
            # 1. SCORE BASE ALTO para qualquer recomendação
            base_score = 0.5
            
            # 2. BOOST MASSIVO para conteúdo (principal motor)
            if rec.source == "content_based":
                # Similaridade semântica direta (0-1)
                semantic_sim = rec.match_features.get('semantic_similarity', 0)
                base_score += semantic_sim * 0.4  # Até +0.4
                
                # Boost por avaliação positiva
                pos_ratio = rec.match_features.get('positive_ratio', 0)
                if pos_ratio > 90:
                    base_score += 0.3
                elif pos_ratio > 80:
                    base_score += 0.2
                elif pos_ratio > 70:
                    base_score += 0.1
            
            # 3. BOOST para colaborativo
            elif rec.source == "collaborative":
                collab_score = rec.match_features.get('collaborative_score', 0)
                base_score += collab_score * 0.3
            
            # 4. BOOST GIGANTE para jogos POPULARES
            total_reviews = rec.match_features.get('total_reviews', 0)
            if total_reviews > 10000:
                base_score += 0.4  # Muito popular
            elif total_reviews > 1000:
                base_score += 0.2  # Popular
            
            # 5. Match com prompt
            if prompt_query['primary_categories']:
                # Verificar match de gênero
                game_genres = str(rec.metadata.get('genres', '')).lower()
                for cat in prompt_query['primary_categories']:
                    if cat.lower() in game_genres:
                        base_score += 0.25  # Match perfeito!
                        break
            
            # 6. LIMITAR entre 0.5 e 1.0 (scores altos!)
            final_score = min(max(base_score, 0.5), 1.0)
            
            # Atualizar score
            rec.score = final_score
        
        # Ordenar por score
        return sorted(recommendations, key=lambda x: x.score, reverse=True)
    
    def _simple_ranking(self, recommendations):
        """
        Ranking simples baseado apenas no score
        """
        try:
            # Garantir que todos os scores são floats
            filtered_recs = []
            for rec in recommendations:
                if not isinstance(rec.score, (int, float)):
                    try:
                        rec.score = float(rec.score)
                    except (ValueError, TypeError):
                        rec.score = 0.0
                
                if rec.score >= self.min_score_threshold:
                    filtered_recs.append(rec)
            
            # Ordenar por score final
            filtered_recs.sort(key=lambda x: float(x.score), reverse=True)
            return filtered_recs
        except Exception as e:
            logger.error(f"Erro no ranking simples: {e}")
            # Retornar na ordem original em caso de erro
            return recommendations
    
    def _apply_diversity(self, recommendations):
        """
        Aplica penalidade de diversidade para evitar recomendações muito similares
        """
        try:
            if len(recommendations) <= 1:
                return recommendations
            
            # Garantir scores como float
            for rec in recommendations:
                if not isinstance(rec.score, (int, float)):
                    try:
                        rec.score = float(rec.score)
                    except (ValueError, TypeError):
                        rec.score = 0.0
            
            # Agrupar por gêneros similares
            genre_groups = {}
            for i, rec in enumerate(recommendations):
                # Extrair gêneros dos metadados ou do games_df
                genres = []
                if 'genres' in rec.metadata:
                    genres_str = rec.metadata['genres']
                    if isinstance(genres_str, str):
                        genres = [g.strip().lower() for g in genres_str.split(';') if g.strip()]
                
                if genres:
                    primary_genre = genres[0] if genres else 'other'
                    if primary_genre not in genre_groups:
                        genre_groups[primary_genre] = []
                    genre_groups[primary_genre].append((i, rec))
                else:
                    # Se não tem gênero, agrupar como 'other'
                    if 'other' not in genre_groups:
                        genre_groups['other'] = []
                    genre_groups['other'].append((i, rec))
            
            # Aplicar penalidade a jogos no mesmo grupo
            diversified_recs = []
            used_indices = set()
            
            # Pegar o melhor de cada grupo primeiro
            for genre, group in genre_groups.items():
                if group:
                    # Garantir comparação de floats
                    try:
                        best_idx, best_rec = max(group, key=lambda x: float(x[1].score))
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Erro ao comparar scores no grupo {genre}: {e}")
                        best_idx, best_rec = group[0]  # Fallback para primeiro
                    
                    # Reduzir score dos outros no mesmo grupo
                    for idx, rec in group:
                        if idx != best_idx:
                            try:
                                rec.score *= (1.0 - self.diversity_penalty)
                            except (ValueError, TypeError):
                                rec.score = 0.0
                    used_indices.add(best_idx)
                    diversified_recs.append(best_rec)
            
            # Adicionar o restante
            for i, rec in enumerate(recommendations):
                if i not in used_indices:
                    diversified_recs.append(rec)
            
            # Reordenar
            diversified_recs.sort(key=lambda x: float(x.score), reverse=True)
            return diversified_recs
        
        except Exception as e:
            logger.error(f"Erro ao aplicar diversidade: {e}")
            return recommendations
    
    def _filter_existing(self, recommendations, user_library):
        """
        Filtra jogos que o usuário já possui
        user_library agora é lista de appids
        """
        filtered = []
        for rec in recommendations:
            # Verificar por appid
            if rec.game_id not in user_library:
                filtered.append(rec)
            else:
                logger.debug(f"Filtrando jogo {rec.game_name} (appid: {rec.game_id}) - já na biblioteca")
        return filtered
    
    def explain_recommendation(self, recommendation, user_prompt):
        """
        Gera explicação detalhada para uma recomendação
        """
        explanation = {
            'game': recommendation.game_name,
            'appid': recommendation.game_id,
            'final_score': round(recommendation.score, 3),
            'match_features': recommendation.match_features,
            'sources': recommendation.metadata.get('sources', []),
            'num_sources': recommendation.metadata.get('num_sources', 1),
            'rationale': recommendation.rationale,
            'key_factors': []
        }
        
        # Adicionar fatores-chave baseados no score
        if 'collaborative_score' in recommendation.match_features:
            explanation['key_factors'].append({
                'factor': 'Usuários similares',
                'value': f"{recommendation.match_features['collaborative_score']:.2f}",
                'impact': 'alta'
            })
        
        if 'semantic_similarity' in recommendation.match_features:
            explanation['key_factors'].append({
                'factor': 'Similaridade semântica',
                'value': f"{recommendation.match_features['semantic_similarity']:.2f}",
                'impact': 'alta'
            })
        
        if 'context_match' in recommendation.match_features:
            explanation['key_factors'].append({
                'factor': 'Match com contexto',
                'value': f"{recommendation.match_features['context_match']:.2f}",
                'impact': 'média'
            })
        
        # Adicionar informações adicionais se disponíveis
        game_info = self.games_df[self.games_df['appid'] == recommendation.game_id]
        if not game_info.empty:
            game_info = game_info.iloc[0]
            if 'positive_ratio' in game_info and pd.notna(game_info['positive_ratio']):
                explanation['positive_ratio'] = game_info['positive_ratio']
            if 'price' in game_info and pd.notna(game_info['price']):
                explanation['price'] = game_info['price']
        
        return explanation
    
    def save_recommendation_history(self, user_id, prompt, recommendations, filepath):
        """
        Salva histórico de recomendações
        """
        try:
            # Criar entrada de histórico
            history_entry = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'user_id': str(user_id),
                'prompt': str(prompt),
                'recommendations': []
            }
            
            # Processar cada recomendação
            for rec in recommendations:
                # Garantir que todos os valores são serializáveis
                try:
                    rec_dict = {
                        'appid': int(rec.game_id) if isinstance(rec.game_id, (int, np.integer)) else str(rec.game_id),
                        'game_name': str(rec.game_name),
                        'score': float(rec.score) if isinstance(rec.score, (int, float)) else 0.0,
                        'rationale': str(rec.rationale) if rec.rationale else "",
                        'source': rec.source,
                        'match_features': rec.match_features
                    }
                    history_entry['recommendations'].append(rec_dict)
                except Exception as rec_error:
                    logger.warning(f"Erro ao processar recomendação para histórico: {rec_error}")
                    continue
            
            # Carregar histórico existente
            try:
                if os.path.exists(filepath):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        history = json.load(f)
                else:
                    history = []
                    
            except Exception as load_error:
                logger.warning(f"Erro ao carregar histórico: {load_error}. Criando novo.")
                history = []
            
            # Adicionar nova entrada (limitar histórico a 100 entradas)
            history.append(history_entry)
            if len(history) > 100:
                history = history[-100:]
            
            # Salvar com encoding UTF-8
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Histórico salvo em: {filepath} ({len(history)} entradas)")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao salvar histórico: {e}")
            return False
