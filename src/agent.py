# src/improved_agent.py
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import json
import os
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Recommendation:
    """Classe para representar uma recomendação"""
    game_id: Any
    game_name: str
    score: float
    rationale: str
    source: str
    metadata: Dict[str, Any]

class RecommendationAgent:
    def __init__(self, 
                 profile_analyzer,
                 prompt_interpreter,
                 embedder,
                 games_df,
                 interactions_df,
                 steam_api):
        """
        Agente de recomendações de jogos Steam
        """
        self.profile_analyzer = profile_analyzer
        self.prompt_interpreter = prompt_interpreter
        self.embedder = embedder
        self.games_df = games_df
        self.interactions_df = interactions_df
        self.steam_api = steam_api
        
        # Configuração
        self.max_library_hours = 3  # Máximo de horas para considerar jogo "não jogado"
        
        logger.info(f"Agente melhorado inicializado: {len(games_df)} jogos disponíveis")
    
    def recommend_from_prompt(self, 
                            user_id: str,
                            user_prompt: str,
                            n_recommendations: int = 5) -> List[Recommendation]:
        """
        Fluxo principal de recomendação:
        1. Extrai TODAS as features importantes do prompt
        2. Busca primeiro na biblioteca
        3. Se não encontrar, busca no dataset considerando múltiplas features
        4. Usa ProfileAnalyzer para dar boost baseado no perfil
        """
        logger.info(f"Iniciando recomendação para usuário {user_id}")
        logger.info(f"Prompt: {user_prompt}")
        
        try:
            # 1. Obter biblioteca do usuário
            library_data = self.steam_api.get_owned_games(user_id)
            if not library_data:
                logger.error("Não foi possível obter biblioteca do usuário")
                return self._get_popular_fallback([], n_recommendations, user_id)
            
            user_library = library_data['games']
            user_library_appids = [game['appid'] for game in user_library]
            logger.info(f"Biblioteca do usuário: {len(user_library)} jogos")
            
            # 2. Extrair as features importantes do prompt 
            prompt_features = self.prompt_interpreter.interpret_to_query(user_prompt)
            top_features = prompt_features['primary_categories']
            
            if not top_features:
                logger.warning("Não foi possível extrair features do prompt")
                return self._get_popular_fallback(user_library_appids, n_recommendations, user_id)
            
            logger.info(f"Features importantes: {top_features}")
            
            # 3. Buscar na biblioteca do usuário a partir das features 
            library_recs = self._search_in_library_with_features(
                user_library=user_library,
                features=top_features,
                max_hours=self.max_library_hours,
                n_recommendations=n_recommendations
            )
            
            logger.info(f"Encontrados {len(library_recs)} jogos na biblioteca")
            
            # 4. Se encontrou o suficiente, retornar
            if len(library_recs) >= n_recommendations:
                sorted_recs = sorted(library_recs, key=lambda x: x.score, reverse=True)
                final_recs = sorted_recs[:n_recommendations]
                
                for rec in final_recs:
                    rec.rationale = self._generate_library_rationale(rec, user_prompt)
                
                logger.info(f"Retornando {len(final_recs)} recomendações da biblioteca")
                return final_recs


            embedder_dfs = []
            embedder_df = pd.DataFrame()
            if len(library_recs) > 0:
                for rec in library_recs:
                    df = self.embedder.search_by_game_id(
                        rec.game_id,
                        self.games_df,
                        self.embedder.faiss_index,
                        top_k=10
                    )

                    if df is not None and not df.empty:
                        embedder_dfs.append(df)

            # Junta tudo
            if embedder_dfs:
                embedder_df = pd.concat(embedder_dfs, ignore_index=True)
            
            if not embedder_df.empty:
                embedder_df = (
                    embedder_df
                    .groupby("appid", as_index=False)
                    .agg({
                        "similarity": "max",
                        "name": "first",
                        "genres": "first",
                        "categories": "first",
                        "positive_ratio": "first",
                        "price": "first",
                        "developer": "first",
                        "distance": "min"
                    })
                    .sort_values("similarity", ascending=False)
                )
 
            recs_embedder = self.build_similarity_recommendations(
                embedder_df,
                source_name="semantic_search"
            )

            # Remover duplicados
            library_game_ids = {rec.game_id for rec in library_recs}
            recs_embedder = [
                rec for rec in recs_embedder
                if rec.game_id not in library_game_ids
            ]
            
            recs_concat = []
            remaining_one = n_recommendations-len(library_recs)        
            if len(recs_embedder)+len(library_recs) >= n_recommendations:
                sorted_lib_recs = sorted(library_recs, key=lambda x: x.score, reverse=True)
                sorted_emb_recs = sorted(recs_embedder, key=lambda x: x.score, reverse=True)
                recs_concat = sorted_lib_recs +  sorted_emb_recs[:remaining_one]
                
                res_sorted = sorted(recs_concat, key=lambda x: x.score, reverse=True)
                for rec in recs_concat:
                    rec.rationale = self._generate_library_rationale(rec, user_prompt)
                
                logger.info(f"Retornando {len(recs_concat)} recomendações do embedding")
                return recs_concat

            # 5. Se não encontrou o suficiente, busca no dataset
            remaining_two = n_recommendations - len(recs_concat)
            logger.info(f"Buscando {remaining_two} jogos no dataset...")
            
            dataset_recs = self._search_in_dataset_with_features(
                features=top_features,
                user_library=user_library_appids,
                n_recommendations=remaining_two,
                user_id=user_id
            )
            
            # 6. Combinar resultados
            all_recs = library_recs + dataset_recs
            
            # 7. Se ainda não tem o suficiente, adicionar fallback popular
            if len(all_recs) < n_recommendations:
                additional = n_recommendations - len(all_recs)
                fallback_recs = self._get_popular_fallback(user_library_appids, additional, user_id)
                all_recs.extend(fallback_recs)
            
            # Garantir que não passamos do limite
            all_recs = all_recs[:n_recommendations]
            
            # Gerar explicações finais
            for rec in all_recs:
                if rec.rationale == "":
                    rec.rationale = self._generate_enhanced_rationale(rec)
            
            logger.info(f"Recomendações finais: {len(all_recs)} (biblioteca: {len(library_recs)}, dataset: {len(dataset_recs)})")
            
            return all_recs
            
        except Exception as e:
            logger.error(f"Erro no fluxo de recomendação: {str(e)}")
            logger.exception(e)
            # Fallback em caso de erro
            return self._get_popular_fallback([], n_recommendations, user_id)
    
    def build_similarity_recommendations(self,
            similarity_df: pd.DataFrame,
            source_name: str = "semantic_search"
        ) -> List[Recommendation]:
        """
        Converte DataFrame de similaridade em objetos Recommendation
        """
        recommendations = []

        if similarity_df is None or similarity_df.empty:
            return recommendations

        for _, game_row in similarity_df.iterrows():
            similarity = float(game_row.get("similarity", 0.0))

            rec = Recommendation(
                game_id=game_row["appid"],
                game_name=game_row.get("name", "Desconhecido"),
                score=similarity,
                rationale="",
                source=source_name,
                metadata={
                    "similarity": similarity,
                    "distance": game_row.get("distance", None),
                    "genres": game_row.get("genres", None),
                    "categories": game_row.get("categories", None),
                    "positive_ratio": game_row.get("positive_ratio", 0),
                    "average_playtime": game_row.get("average_playtime", None),
                    "price": game_row.get("price", None),
                    "developer": game_row.get("developer", None),
                    "game_data": game_row.to_dict()
                }
            )

            recommendations.append(rec)

        return recommendations


    def _generate_enhanced_rationale(self, recommendation: Recommendation) -> str:
        """
        Gera explicação aprimorada para jogos do dataset
        """
        source = recommendation.source
        metadata = recommendation.metadata
        
        if source == "dataset":
            return self._generate_dataset_rationale(recommendation)
        elif source == "popular_fallback":
            return recommendation.rationale
        else:
            return self._generate_dataset_rationale(recommendation)
    
    def _search_in_library_with_features(self, 
                                        user_library: List[Dict],
                                        features: List[str],
                                        max_hours: float = 3,
                                        n_recommendations: int = 5) -> List[Recommendation]:
        """
        Busca jogos na biblioteca que:
        1. Tenham menos de max_hours de gameplay
        2. Tenham PELO MENOS UMA das features
        """
        recommendations = []
        
        if not user_library or not features:
            return []
        
        # Filtrar jogos com menos de max_hours
        filtered_games = []
        for game in user_library:
            playtime_minutes = game.get('playtime_forever', 0)
            playtime_hours = playtime_minutes / 60
            
            if playtime_hours < max_hours:
                filtered_games.append({
                    'appid': game['appid'],
                    'name': game.get('name', 'Desconhecido'),
                    'playtime_hours': playtime_hours,
                    'metadata': game
                })
        
        logger.info(f"Jogos na biblioteca com menos de {max_hours}h: {len(filtered_games)}")
        
        if not filtered_games:
            return []
        
        # Para cada jogo filtrado, verificar se tem alguma feature
        for game in filtered_games:
            appid = game['appid']
            
            # Buscar informações completas do jogo no dataset
            game_info = self.games_df[self.games_df['appid'] == appid]
            
            if game_info.empty:
                # Se não encontrou no dataset, verificar apenas pelo nome
                best_feature, best_score = self._check_features_in_game_basic(game['name'], features)
            else:
                # Se encontrou no dataset, verificar em todos os campos
                game_row = game_info.iloc[0]
                best_feature, best_score = self._check_features_in_game_complete(game_row, features)
            
            if best_feature:  # Se encontrou alguma feature
                library_bonus = 0.5
                
                # Score final
                final_score = min(best_score + library_bonus, 1.0)
                
                rec = Recommendation(
                    game_id=appid,
                    game_name=game['name'],
                    score=final_score,
                    rationale="",
                    source="user_library",
                    metadata={
                        'playtime_hours': game['playtime_hours'],
                        'best_feature': best_feature,
                        'match_score': best_score,
                        'library_bonus': library_bonus,
                        'all_features': features
                    }
                )
                recommendations.append(rec)
        
        # Ordenar por score (maior primeiro)
        recommendations.sort(key=lambda x: x.score, reverse=True)
        
        return recommendations[:min(len(recommendations), n_recommendations * 2)]
    
    def _check_features_in_game_basic(self, game_name: str, features: List[str]) -> Tuple[str, float]:
        """
        Verifica se o jogo tem alguma das features baseado apenas no nome
        
        Returns:
            Tuple[best_feature, best_score] (melhor feature encontrada e seu score)
        """
        if not game_name or not features:
            return None, 0.0
        
        game_name_lower = game_name.lower()
        
        # Procurar cada feature
        best_feature = None
        best_score = 0.0
        
        for feature in features:
            feature_lower = feature.lower()
            if feature_lower in game_name_lower:
                # Feature encontrada no nome - score alto
                best_feature = feature
                best_score = 0.8
                break
        
        return best_feature, best_score
    
    def _check_features_in_game_complete(self, game_row: pd.Series, features: List[str]) -> Tuple[str, float]:
        """
        Verifica se o jogo tem alguma das features em vários campos
        
        Returns:
            Tuple[best_feature, best_score] (melhor feature encontrada e seu score)
        """
        if not features:
            return None, 0.0
        
        best_feature = None
        best_score = 0.0
        
        # Campos a verificar (com pesos)
        fields_to_check = [
            ('genres', 1.0),           # Gêneros são o mais importante
            ('steamspy_tags', 0.8),    # Tags do SteamSpy
            ('categories', 0.7),       # Categorias Steam
            ('name', 0.5),             # Nome do jogo
            ('short_description', 0.3) # Descrição curta
        ]
        
        for feature in features:
            feature_lower = feature.lower()
            feature_score = 0.0
            total_weight = 0
            
            for field, weight in fields_to_check:
                if field in game_row and pd.notna(game_row[field]):
                    field_value = str(game_row[field]).lower()
                    
                    # Verificar se a feature está neste campo
                    if feature_lower in field_value:
                        feature_score += weight
                    
                    total_weight += weight
            
            # Calcular score final para esta feature (0 a 1)
            if total_weight > 0:
                feature_score = feature_score / total_weight
            
            # Se esta feature tem score melhor que a anterior, atualizar
            if feature_score > best_score:
                best_feature = feature
                best_score = feature_score
        
        return best_feature, best_score
    
    def _search_in_dataset_with_features(self,
                                        features: List[str],
                                        user_library: List[int],
                                        n_recommendations: int,
                                        user_id: str = None) -> List[Recommendation]:
        """
        Busca no dataset por jogos que tenham PELO MENOS UMA das features
        """
        recommendations = []
        
        if not features:
            return self._get_popular_fallback(user_library, n_recommendations, user_id)
        
        logger.info(f"Buscando jogos no dataset com features: {features}")
        
        # Filtrar jogos que não estão na biblioteca do usuário
        available_games = self.games_df[~self.games_df['appid'].isin(user_library)]
        
        if len(available_games) == 0:
            logger.warning("Nenhum jogo disponível no dataset após filtrar biblioteca")
            return self._get_popular_fallback(user_library, n_recommendations, user_id)
        
        # Para cada jogo disponível, verificar se tem alguma feature
        for _, game_row in available_games.iterrows():
            if len(recommendations) >= n_recommendations * 2:
                break
            
            best_feature, best_score = self._check_features_in_game_complete(game_row, features)
            
            if best_feature and best_score > 0.1:  # Threshold baixo para aceitar
                # Adicionar bônus por qualidade (revies positivas)
                quality_bonus = self._calculate_quality_bonus(game_row)

                # Bônus por popularidade (muitas reviews)
                popularity_bonus = self._calculate_popularity_bonus(game_row)
                
                # Bônus baseado no perfil do usuario
                profile_bonus = self._calculate_profile_bonus(game_row['appid'], user_id)
                
                # Score final = match_score + quality_bonus (máximo 1.0)
                final_score = min(best_score + quality_bonus, 1.0)
                
                final_score = min(
                    best_score + quality_bonus + popularity_bonus + profile_bonus,
                    1.0
                )
                
                rec = Recommendation(
                    game_id=game_row['appid'],
                    game_name=game_row.get('name', 'Desconhecido'),
                    score=final_score,
                    rationale="",
                    source="dataset",
                    metadata={
                        'best_feature': best_feature,
                        'match_score': best_score,
                        'quality_bonus': quality_bonus,
                        'popularity_bonus': popularity_bonus,
                        'positive_ratio': game_row.get('positive_ratio', 0),
                        'all_features': features,
                        'game_data': game_row.to_dict()
                    }
                )
                recommendations.append(rec)
        
        # Ordenar por score (maior primeiro)
        recommendations.sort(key=lambda x: x.score, reverse=True)
        
        logger.info(f"Encontrados {len(recommendations)} jogos no dataset com as features")
        
        return recommendations[:n_recommendations]
    
    def _calculate_quality_bonus(self, game_row) -> float:
        """
        Calcula bônus baseado na qualidade do jogo
        """
        bonus = 0.0
        pos_ratio = game_row.get('positive_ratio', 0)
        
        if pos_ratio >= 90:
            bonus = 0.4
        elif pos_ratio >= 80:
            bonus = 0.3
        elif pos_ratio >= 70:
            bonus = 0.2
        
        return bonus
    
    def _calculate_popularity_bonus(self, game_row) -> float:
        """
        Calcula bônus baseado na popularidade
        """
        bonus = 0.0
        total_reviews = 0
        
        if 'positive_ratings' in game_row and 'negative_ratings' in game_row:
            total_reviews = game_row.get('positive_ratings', 0) + game_row.get('negative_ratings', 0)
        
        if total_reviews > 50000:
            bonus = 0.3
        elif total_reviews > 10000:
            bonus = 0.2
        elif total_reviews > 1000:
            bonus = 0.1
        
        return bonus
    
    def _calculate_profile_bonus(self, game_id: int, user_id: str) -> float:
        """
        Calcula bônus baseado no perfil do usuário usando ProfileAnalyzer
        
        Args:
            game_id: appid do jogo
            user_id: ID do usuário
            
        Returns:
            Bônus de perfil (0.0 a 0.4)
        """
        try:
            # Verificar se o usuário está no modelo do ProfileAnalyzer
            if (hasattr(self.profile_analyzer, 'user_encoder') and 
                user_id in self.profile_analyzer.user_encoder):
                
                # Obter recomendações colaborativas para o usuário
                collab_df = self.profile_analyzer.recommend_from_profile(
                    user_id=user_id,
                    games_df=self.games_df,
                    top_n=100,
                    exclude_played=True,
                    user_library=[]
                )
                
                # Verificar se o jogo está nas recomendações colaborativas
                if not collab_df.empty and 'appid' in collab_df.columns:
                    matching_row = collab_df[collab_df['appid'] == game_id]
                    if not matching_row.empty:
                        # Extrair score colaborativo (normalizado para 0-1)
                        collab_score = float(matching_row.iloc[0].get('score', 0))
                        # Converter para bônus (0.0 a 0.4)
                        return min(collab_score * 0.3, 0.4)
            
        except Exception as e:
            logger.warning(f"Erro ao calcular bônus de perfil: {e}")
        
        return 0.0
    
    def _get_popular_fallback(self, user_library: List[int], n_recommendations: int, user_id: str) -> List[Recommendation]:
        """
        Fallback: recomenda jogos populares e bem avaliados
        """
        recommendations = []
        
        try:
            # Filtrar jogos com boa avaliação e popularidade
            popular_df = self.games_df.copy()
            
            # Aplicar filtros básicos de qualidade
            if 'positive_ratio' in popular_df.columns and 'positive_ratings' in popular_df.columns:
                popular_df = popular_df[
                    (popular_df['positive_ratio'] >= 70) &
                    (popular_df['positive_ratings'] >= 500)
                ]
            
            # Filtrar jogos que não estão na biblioteca
            if user_library:
                popular_df = popular_df[~popular_df['appid'].isin(user_library)]
            
            # Ordenar por popularidade (avaliações positivas) e depois por avaliação
            if 'positive_ratings' in popular_df.columns:
                popular_df = popular_df.sort_values(['positive_ratings', 'positive_ratio'], 
                                                  ascending=[False, False])
            
            for _, row in popular_df.head(n_recommendations).iterrows():
                # Score baseado na popularidade
                base_score = 0.6  # Score base decente
                
                # Bônus por avaliação
                pos_ratio = row.get('positive_ratio', 0)
                if pos_ratio > 90:
                    base_score += 0.2
                elif pos_ratio > 80:
                    base_score += 0.1
                elif pos_ratio > 70:
                    base_score += 0.05
                
                # Bônus por número de reviews
                total_reviews = row.get('positive_ratings', 0) + row.get('negative_ratings', 0)
                if total_reviews > 10000:
                    base_score += 0.2
                elif total_reviews > 1000:
                    base_score += 0.1

                 # Bônus do perfil
                profile_bonus = self._calculate_profile_bonus(row['appid'], user_id) if user_id else 0.0
                
                rec = Recommendation(
                    game_id=row['appid'],
                    game_name=row.get('name', 'Desconhecido'),
                    score=min(base_score, 0.9),
                    rationale="Jogo popular bem avaliado pela comunidade Steam",
                    source="popular_fallback",
                    metadata={
                        'positive_ratio': pos_ratio,
                        'total_reviews': total_reviews,
                        'profile_bonus': profile_bonus,
                        'popular_fallback': True
                    }
                )
                recommendations.append(rec)
                
        except Exception as e:
            logger.error(f"Erro no fallback de popularidade: {e}")
        
        return recommendations[:n_recommendations]
    
    def _generate_library_rationale(self, recommendation: Recommendation, user_prompt: str) -> str:
        """
        Gera explicação aprimorada para jogos da biblioteca
        """
        playtime = recommendation.metadata.get('playtime_hours', 0)
        best_feature = recommendation.metadata.get('best_feature', '')
        match_score = recommendation.metadata.get('match_score', 0)
        profile_bonus = recommendation.metadata.get('profile_bonus', 0)
        
        # Informação sobre tempo de jogo
        if playtime == 0:
            time_info = "que você ainda não jogou"
            suggestion = "É uma ótima oportunidade para descobrir este jogo que já está na sua biblioteca!"
        else:
            time_info = f"que você jogou apenas {playtime:.1f} horas"
            suggestion = "Parece que você ainda não explorou tudo que este jogo tem a oferecer!"
        
        # Força do match
        if match_score > 0.7:
            match_strength = "combina perfeitamente"
            match_detail = f"com sua busca por {best_feature}"
        elif match_score > 0.4:
            match_strength = "se encaixa muito bem"
            match_detail = f"no que você está procurando em termos de {best_feature}"
        else:
            match_strength = "tem relação"
            match_detail = f"com aspectos de {best_feature}, como você mencionou"

        # Informação do perfil 
        profile_info = ""
        if profile_bonus > 0.2:
            profile_info = " Além disso, jogadores com gostos similares aos seus também gostam deste jogo."
        elif profile_bonus > 0.1:
            profile_info = " Este jogo também é apreciado por jogadores com perfil similar ao seu."
        
          # Informação sobre qualidade do jogo
        quality_info = ""
        game_data = recommendation.metadata.get('game_data', {})
        if 'positive_ratio' in game_data:
            pos_ratio = game_data['positive_ratio']
            if pos_ratio >= 90:
                quality_info = " Ele tem avaliações excelentes da comunidade."
            elif pos_ratio >= 80:
                quality_info = " É um jogo muito bem avaliado pelos jogadores."
        
        rationale = (
            f"**{recommendation.game_name}** é um jogo {time_info} que {match_strength} {match_detail}. "
            f"{suggestion}{quality_info}{profile_info}"
        )
        return rationale    
    
    def _generate_dataset_rationale(self, recommendation: Recommendation) -> str:
        """
        Gera explicação detalhada para jogos do dataset
        """
        best_feature = recommendation.metadata.get('best_feature', '')
        match_score = recommendation.metadata.get('match_score', 0)
        quality_bonus = recommendation.metadata.get('quality_bonus', 0)
        popularity_bonus = recommendation.metadata.get('popularity_bonus', 0)
        profile_bonus = recommendation.metadata.get('profile_bonus', 0)
        game_data = recommendation.metadata.get('game_data', {})
        
        # Parte 1: Match com a busca
        if match_score > 0.7:
            match_phrase = f"**{recommendation.game_name}** é uma excelente escolha que atende perfeitamente"
        elif match_score > 0.4:
            match_phrase = f"**{recommendation.game_name}** se encaixa muito bem"
        else:
            match_phrase = f"**{recommendation.game_name}** tem características que podem interessar"
        
        feature_phrase = f"à sua busca por {best_feature}"
        
        # Parte 2: Qualidade do jogo
        quality_phrase = ""
        if 'positive_ratio' in game_data:
            pos_ratio = game_data['positive_ratio']
            if pos_ratio >= 90:
                quality_phrase = f" Com impressionantes {pos_ratio}% de avaliações positivas,"
            elif pos_ratio >= 80:
                quality_phrase = f" Com {pos_ratio}% de avaliações positivas,"
            elif pos_ratio >= 70:
                quality_phrase = f" Bem avaliado pelos jogadores ({pos_ratio}% positivas),"
        
        quality_detail = ""
        if quality_bonus >= 0.2:
            quality_detail = " é considerado um jogo de alta qualidade pela comunidade."
        elif quality_bonus >= 0.1:
            quality_detail = " tem recebido boas críticas dos jogadores."
        
        # Parte 3: Popularidade
        popularity_phrase = ""
        if popularity_bonus >= 0.2:
            popularity_phrase = " Além disso, é muito popular na Steam"
        elif popularity_bonus >= 0.1:
            popularity_phrase = " É um jogo bastante popular"
        
        # Parte 4: Recomendação baseada no perfil
        profile_phrase = ""
        if profile_bonus > 0.2:
            profile_phrase = " Jogadores com gostos similares aos seus costumam gostar muito deste título."
        elif profile_bonus > 0.1:
            profile_phrase = " Baseado no seu perfil de jogador, este jogo pode ser uma boa escolha."
        
        # Parte 5: Informações adicionais
        extra_info = ""
        if 'genres' in game_data:
            genres = str(game_data['genres']).split(';')
            if len(genres) > 0 and genres[0]:
                main_genre = genres[0].strip()
                extra_info = f" O jogo é primariamente do gênero {main_genre}."
        
        if 'average_playtime' in game_data and game_data['average_playtime'] > 0:
            avg_hours = game_data['average_playtime'] / 60
            if avg_hours < 10:
                extra_info += " As sessões de jogo costumam ser mais curtas, ideal para jogar em intervalos."
            elif avg_hours > 50:
                extra_info += " Oferece muitas horas de conteúdo, perfeito para se imergir por longos períodos."
        
        # Construir a explicação final
        rationale_parts = [
            f"{match_phrase} {feature_phrase}.",
            f"{quality_phrase}{quality_detail}",
            f"{popularity_phrase}.",
            f"{profile_phrase}",
            f"{extra_info}"
        ]
        
        # Remover partes vazias e juntar
        rationale = " ".join([part for part in rationale_parts if part.strip()])
        
        return rationale
    
    def save_recommendation_history(self, user_id, prompt, recommendations, filepath):
        """
        Salva histórico de recomendações
        """
        try:
            history_entry = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'user_id': str(user_id),
                'prompt': str(prompt),
                'recommendations': []
            }
            
            for rec in recommendations:
                rec_dict = {
                    'appid': int(rec.game_id),
                    'game_name': str(rec.game_name),
                    'score': float(rec.score),
                    'rationale': str(rec.rationale),
                    'source': rec.source,
                    'metadata': rec.metadata
                }
                history_entry['recommendations'].append(rec_dict)
            
            # Carregar histórico existente
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            else:
                history = []
            
            # Adicionar e limitar histórico
            history.append(history_entry)
            if len(history) > 100:
                history = history[-100:]
            
            # Salvar
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Histórico salvo: {len(history)} entradas")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao salvar histórico: {e}")
            return False