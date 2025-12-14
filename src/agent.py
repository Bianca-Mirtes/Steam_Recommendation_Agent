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
class ImprovedRecommendation:
    """Classe para representar uma recomendação"""
    game_id: Any
    game_name: str
    score: float
    rationale: str
    source: str
    metadata: Dict[str, Any]

class ImprovedRecommendationAgent:
    def __init__(self, 
                 prompt_interpreter,
                 games_df,
                 steam_api):
        """
        Agente melhorado de recomendações
        
        Args:
            prompt_interpreter: Instância do PromptInterpreter
            games_df: DataFrame com informações dos jogos
            steam_api: Instância da SteamAPI
        """
        self.prompt_interpreter = prompt_interpreter
        self.games_df = games_df
        self.steam_api = steam_api
        
        # Configuração
        self.max_library_hours = 3  # Máximo de horas para considerar jogo "não jogado"
        
        logger.info(f"Agente melhorado inicializado: {len(games_df)} jogos disponíveis")
    
    def recommend_from_prompt(self, 
                            user_id: str,
                            user_prompt: str,
                            n_recommendations: int = 5) -> List[ImprovedRecommendation]:
        """
        Fluxo principal de recomendação:
        1. Extrai TODAS as features importantes do prompt
        2. Busca primeiro na biblioteca
        3. Se não encontrar, busca no dataset considerando múltiplas features
        """
        logger.info(f"Iniciando recomendação para usuário {user_id}")
        logger.info(f"Prompt: {user_prompt}")
        
        try:
            # 1. Obter biblioteca do usuário
            library_data = self.steam_api.get_owned_games(user_id)
            if not library_data:
                logger.error("Não foi possível obter biblioteca do usuário")
                return self._get_popular_fallback([], n_recommendations)
            
            user_library = library_data['games']
            user_library_appids = [game['appid'] for game in user_library]
            logger.info(f"Biblioteca do usuário: {len(user_library)} jogos")
            
            # 2. Extrair TODAS as features importantes do prompt (não só a principal)
            prompt_features = self.prompt_interpreter.interpret_to_query(user_prompt)
            all_features = self._get_all_important_features(prompt_features)
            
            if not all_features:
                logger.warning("Não foi possível extrair features do prompt")
                return self._get_popular_fallback(user_library_appids, n_recommendations)
            
            logger.info(f"Features importantes: {all_features}")
            
            # 3. Buscar na biblioteca do usuário
            library_recs = self._search_in_library_with_features(
                user_library=user_library,
                features=all_features,
                max_hours=self.max_library_hours,
                n_recommendations=n_recommendations
            )
            
            logger.info(f"Encontrados {len(library_recs)} jogos na biblioteca")
            
            # 4. Se encontrou o suficiente, retornar
            if len(library_recs) >= n_recommendations:
                sorted_recs = sorted(library_recs, key=lambda x: x.score, reverse=True)
                final_recs = sorted_recs[:n_recommendations]
                
                for rec in final_recs:
                    rec.rationale = self._generate_library_rationale(rec)
                
                logger.info(f"Retornando {len(final_recs)} recomendações da biblioteca")
                return final_recs
            
            # 5. Se não encontrou o suficiente, buscar no dataset
            remaining = n_recommendations - len(library_recs)
            logger.info(f"Buscando {remaining} jogos no dataset...")
            
            dataset_recs = self._search_in_dataset_with_features(
                features=all_features,
                user_library=user_library_appids,
                n_recommendations=remaining
            )
            
            # 6. Combinar resultados
            all_recs = library_recs + dataset_recs
            
            # 7. Se ainda não tem o suficiente, adicionar fallback popular
            if len(all_recs) < n_recommendations:
                additional = n_recommendations - len(all_recs)
                fallback_recs = self._get_popular_fallback(user_library_appids, additional)
                all_recs.extend(fallback_recs)
            
            # Garantir que não passamos do limite
            all_recs = all_recs[:n_recommendations]
            
            # Gerar explicações finais
            for rec in all_recs:
                if rec.rationale == "":
                    rec.rationale = self._generate_dataset_rationale(rec)
            
            logger.info(f"Recomendações finais: {len(all_recs)} (biblioteca: {len(library_recs)}, dataset: {len(dataset_recs)})")
            
            return all_recs
            
        except Exception as e:
            logger.error(f"Erro no fluxo de recomendação: {str(e)}")
            logger.exception(e)
            # Fallback em caso de erro
            return self._get_popular_fallback([], n_recommendations)
    
    def _get_all_important_features(self, prompt_features: Dict) -> List[str]:
        """
        Extrai todas as features importantes do prompt (não só a principal)
        
        Args:
            prompt_features: Output do PromptInterpreter
            
        Returns:
            Lista de features importantes (mínimo score de 0.5)
        """
        if 'features' not in prompt_features:
            logger.warning("PromptInterpreter não retornou 'features'")
            return []
        
        top_features = prompt_features['features']
        
        if not top_features:
            return []
        
        # Filtrar features com score alto (>= 0.5)
        important_features = []
        for feature, score in top_features.items():
            if score >= 0.5:
                # Normalizar feature (mapear sinônimos)
                normalized_feature = self._normalize_feature(feature)
                if normalized_feature:
                    important_features.append(normalized_feature)
        
        # Remover duplicados
        important_features = list(set(important_features))
        
        logger.info(f"Features importantes filtradas: {important_features}")
        
        return important_features
    
    def _normalize_feature(self, feature: str) -> str:
        """
        Normaliza uma feature mapeando sinônimos para termos padrão
        
        Args:
            feature: Feature extraída
            
        Returns:
            Feature normalizada ou None se não for relevante
        """
        feature_lower = feature.lower().strip()
        
        # Mapeamento de sinônimos para termos padrão
        feature_mapping = {
            # RPG
            'rpg': ['rpg', 'role-playing', 'role playing', 'papel'],
            
            # Mundo Aberto
            'mundo aberto': ['mundo aberto', 'open world', 'sandbox', 'livre', 'explorar', 'exploração'],
            
            # Ação
            'ação': ['ação', 'action', 'tiro', 'shooter', 'fps', 'combate', 'batalha'],
            
            # Aventura
            'aventura': ['aventura', 'adventure', 'descoberta', 'viagem'],
            
            # Terror
            'terror': ['terror', 'horror', 'assustador', 'medo', 'suspense'],
            
            # Multijogador
            'multijogador': ['multijogador', 'multiplayer', 'coop', 'cooperativo', 'co-op', 'online', 'amigos'],
            
            # Estratégia
            'estratégia': ['estratégia', 'strategy', 'tático', 'planejamento', 'cérebro'],
            
            # Simulação
            'simulação': ['simulação', 'simulation', 'simulador'],
            
            # Corrida
            'corrida': ['corrida', 'racing', 'carro', 'automobilismo', 'velocidade'],
            
            # Esportes
            'esporte': ['esporte', 'sports', 'futebol', 'basquete', 'fifa', 'nba'],
            
            # Puzzle
            'puzzle': ['puzzle', 'quebra-cabeça', 'lógica', 'enigma'],
            
            # Indie
            'indie': ['indie', 'independente'],
            
            # Casual
            'casual': ['casual', 'relaxante', 'leve', 'simples', 'fácil', 'rápido'],
            
            # Competitivo
            'competitivo': ['competitivo', 'pvp', 'ranked', 'versus'],
            
            # História
            'história': ['história', 'story', 'narrativa', 'enredo', 'personagem', 'trama'],
            
            # Sobrevivência
            'sobrevivência': ['sobrevivência', 'survival', 'crafting', 'sobreviver'],
            
            # Roguelike
            'roguelike': ['roguelike', 'procedural', 'permadeath', 'rogue', 'lite'],
            
            # Metroidvania
            'metroidvania': ['metroidvania', 'metroid', 'vania'],
        }
        
        # Verificar se a feature está no mapeamento
        for key, synonyms in feature_mapping.items():
            if feature_lower in synonyms or any(syn in feature_lower for syn in synonyms):
                return key
        
        # Se não encontrou mapeamento, retorna a própria feature
        return feature_lower
    
    def _search_in_library_with_features(self, 
                                        user_library: List[Dict],
                                        features: List[str],
                                        max_hours: float = 3,
                                        n_recommendations: int = 5) -> List[ImprovedRecommendation]:
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
                # Bônus por ser da biblioteca e pouco jogado
                library_bonus = 0.3 if game['playtime_hours'] == 0 else 0.1
                
                # Score final
                final_score = min(best_score + library_bonus, 1.0)
                
                rec = ImprovedRecommendation(
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
                                        n_recommendations: int) -> List[ImprovedRecommendation]:
        """
        Busca no dataset por jogos que tenham PELO MENOS UMA das features
        """
        recommendations = []
        
        if not features:
            return self._get_popular_fallback(user_library, n_recommendations)
        
        logger.info(f"Buscando jogos no dataset com features: {features}")
        
        # Filtrar jogos que não estão na biblioteca do usuário
        available_games = self.games_df[~self.games_df['appid'].isin(user_library)]
        
        if len(available_games) == 0:
            logger.warning("Nenhum jogo disponível no dataset após filtrar biblioteca")
            return self._get_popular_fallback(user_library, n_recommendations)
        
        # Para cada jogo disponível, verificar se tem alguma feature
        for _, game_row in available_games.iterrows():
            if len(recommendations) >= n_recommendations * 2:
                break
            
            best_feature, best_score = self._check_features_in_game_complete(game_row, features)
            
            if best_feature and best_score > 0.1:  # Threshold baixo para aceitar
                # Adicionar bônus por qualidade (do agente antigo)
                quality_bonus = self._calculate_quality_bonus(game_row)
                
                # Score final = match_score + quality_bonus (máximo 1.0)
                final_score = min(best_score + quality_bonus, 1.0)
                
                # Bônus extra por popularidade (se muito bem avaliado)
                popularity_bonus = self._calculate_popularity_bonus(game_row)
                final_score = min(final_score + (popularity_bonus * 0.5), 1.0)
                
                rec = ImprovedRecommendation(
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
                        'all_features': features
                    }
                )
                recommendations.append(rec)
        
        # Ordenar por score (maior primeiro)
        recommendations.sort(key=lambda x: x.score, reverse=True)
        
        logger.info(f"Encontrados {len(recommendations)} jogos no dataset com as features")
        
        return recommendations[:n_recommendations]
    
    def _calculate_quality_bonus(self, game_row) -> float:
        """
        Calcula bônus baseado na qualidade do jogo (do agente antigo)
        """
        bonus = 0.0
        pos_ratio = game_row.get('positive_ratio', 0)
        
        if pos_ratio >= 90:
            bonus = 0.3
        elif pos_ratio >= 80:
            bonus = 0.2
        elif pos_ratio >= 70:
            bonus = 0.1
        
        return bonus
    
    def _calculate_popularity_bonus(self, game_row) -> float:
        """
        Calcula bônus baseado na popularidade (do agente antigo)
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
    
    def _get_popular_fallback(self, user_library: List[int], n_recommendations: int) -> List[ImprovedRecommendation]:
        """
        Fallback: recomenda jogos populares e bem avaliados (do agente antigo)
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
                
                rec = ImprovedRecommendation(
                    game_id=row['appid'],
                    game_name=row.get('name', 'Desconhecido'),
                    score=min(base_score, 0.9),
                    rationale="Jogo popular bem avaliado pela comunidade Steam",
                    source="popular_fallback",
                    metadata={
                        'positive_ratio': pos_ratio,
                        'total_reviews': total_reviews,
                        'popular_fallback': True
                    }
                )
                recommendations.append(rec)
                
        except Exception as e:
            logger.error(f"Erro no fallback de popularidade: {e}")
        
        return recommendations[:n_recommendations]
    
    def _generate_library_rationale(self, recommendation: ImprovedRecommendation) -> str:
        """Gera explicação para jogos da biblioteca"""
        playtime = recommendation.metadata.get('playtime_hours', 0)
        best_feature = recommendation.metadata.get('best_feature', '')
        match_score = recommendation.metadata.get('match_score', 0)
        
        if playtime == 0:
            time_info = "que você ainda não jogou"
        else:
            time_info = f"que você jogou apenas {playtime:.1f} horas"
        
        if match_score > 0.6:
            match_strength = "combina muito bem"
        elif match_score > 0.3:
            match_strength = "combina bem"
        else:
            match_strength = "tem relação com"
        
        return f"{recommendation.game_name} é um jogo {time_info} e {match_strength} com sua busca por jogos de {best_feature}. Vale a pena dar uma chance!"
    
    def _generate_dataset_rationale(self, recommendation: ImprovedRecommendation) -> str:
        """Gera explicação para jogos do dataset"""
        source = recommendation.source
        best_feature = recommendation.metadata.get('best_feature', '')
        match_score = recommendation.metadata.get('match_score', 0)
        quality_bonus = recommendation.metadata.get('quality_bonus', 0)
        
        if source == "dataset":
            if match_score > 0.7:
                match_desc = "combina muito bem"
            elif match_score > 0.4:
                match_desc = "combina bem"
            else:
                match_desc = "tem relação com"
            
            rationale = f"{recommendation.game_name} {match_desc} com sua busca por jogos de {best_feature}."
            
            # Adicionar informação sobre qualidade
            if quality_bonus >= 0.2:
                rationale += " Tem avaliações excelentes da comunidade."
            elif quality_bonus >= 0.1:
                rationale += " É bem avaliado pelos jogadores."
        
        elif source == "popular_fallback":
            rationale = f"{recommendation.game_name} é um jogo popular bem avaliado que pode te interessar."
        
        else:
            rationale = f"{recommendation.game_name} é uma boa recomendação baseada no seu perfil."
        
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