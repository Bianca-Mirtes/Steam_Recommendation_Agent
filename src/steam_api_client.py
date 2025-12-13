import requests
import json
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import os

# === SUA API KEY AQUI ===
STEAM_API_KEY = os.environ.get('STEAM_API_KEY', '2B0DFBFA39ACA4D637DDA0AEB88873C1')
# ========================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SteamAPI:
    def __init__(self, api_key: str = None):
        """Inicializa a conexão com a API da Steam"""
        # Usa a API key fornecida OU a configurada acima
        self.api_key = api_key or STEAM_API_KEY
        
        if not self.api_key or self.api_key == "SUA_CHAVE_API_AQUI":
            raise ValueError("❌ API Key da Steam não configurada. Cole sua chave no código.")
        
        self.base_url = "https://api.steampowered.com"
        self.timeout = 15
        
        logger.info(f"API Steam inicializada (key: {self.api_key[:8]}...)")
    
    def extract_steam_id(self, user_input: str) -> Optional[str]:
        """
        Extrai o SteamID64 a partir de diferentes formatos:
        - SteamID64 direto (7656119xxxxxxxx)
        - SteamID3 ([U:1:xxxxxx])
        - SteamID32 (STEAM_0:x:xxxxxx)
        - Vanity URL (nome de usuário)
        - URL de perfil completo
        
        Args:
            user_input: Input do usuário (ID, URL, etc.)
            
        Returns:
            SteamID64 ou None se não encontrado
        """
        # Se já é um SteamID64 (17 dígitos começando com 7656119)
        if user_input.isdigit() and len(user_input) >= 17:
            return user_input
        
        # Se é uma URL, extrair o Vanity
        if 'steamcommunity.com' in user_input:
            # Extrair o último segmento da URL
            parts = user_input.rstrip('/').split('/')
            vanity = parts[-1]
            
            # Se for profiles/7656119...
            if vanity.isdigit():
                return vanity
            else:
                # Converter Vanity para SteamID64
                return self._vanity_to_steamid64(vanity)
        
        # Tentar converter outras formas
        return self._convert_to_steamid64(user_input)
    
    def _vanity_to_steamid64(self, vanity: str) -> Optional[str]:
        """Converte Vanity URL para SteamID64"""
        try:
            url = f"{self.base_url}/ISteamUser/ResolveVanityURL/v0001/"
            params = {
                'key': self.api_key,
                'vanityurl': vanity
            }
            
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            if data['response']['success'] == 1:
                return data['response']['steamid']
            else:
                logger.warning(f"Vanity URL '{vanity}' não encontrada")
                return None
                
        except Exception as e:
            logger.error(f"Erro ao converter vanity URL: {e}")
            return None
    
    def _convert_to_steamid64(self, steam_id: str) -> Optional[str]:
        """Converte outros formatos de SteamID para SteamID64"""
        try:
            # Para simplificação, vamos apenas tentar obter o perfil
            # Em uma implementação completa, você teria lógica para cada formato
            url = f"http://api.steampowered.com/ISteamUser/GetPlayerSummaries/v0002/"
            params = {
                'key': self.api_key,
                'steamids': steam_id
            }
            
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            if data['response']['players']:
                return data['response']['players'][0]['steamid']
            return None
            
        except Exception as e:
            logger.error(f"Erro ao converter SteamID: {e}")
            return None
    
    def get_player_summary(self, steam_id: str) -> Optional[Dict]:
        """Obtém informações básicas do jogador"""
        try:
            url = f"{self.base_url}/ISteamUser/GetPlayerSummaries/v0002/"
            params = {
                'key': self.api_key,
                'steamids': steam_id
            }
            
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            if data['response']['players']:
                return data['response']['players'][0]
            return None
            
        except Exception as e:
            logger.error(f"Erro ao obter perfil do jogador: {e}")
            return None
    
    def get_owned_games(self, steam_id: str) -> Optional[Dict]:
        """Obtém a lista de jogos do jogador"""
        try:
            url = f"{self.base_url}/IPlayerService/GetOwnedGames/v0001/"
            params = {
                'key': self.api_key,
                'steamid': steam_id,
                'include_appinfo': 1,
                'include_played_free_games': 1
            }
            
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            
            if 'response' in data and 'games' in data['response']:
                games = data['response']['games']
                
                # Converter minutos para horas
                total_playtime_minutes = sum(game.get('playtime_forever', 0) for game in games)
                total_playtime_hours = total_playtime_minutes / 60
                
                return {
                    'games': games,
                    'game_count': len(games),
                    'total_playtime_minutes': total_playtime_minutes,
                    'total_playtime_hours': round(total_playtime_hours, 2)
                }
            return {
                'games': [],
                'game_count': 0,
                'total_playtime_minutes': 0,
                'total_playtime_hours': 0
            }
            
        except Exception as e:
            logger.error(f"Erro ao obter jogos: {e}")
            return None

# Função auxiliar para análise de dados
def analyze_gaming_profile(games_data: Dict) -> Dict:
    """
    Analisa os dados dos jogos para criar um perfil
    
    Args:
        games_data: Dados dos jogos obtidos da API
        
    Returns:
        Perfil analisado com preferências
    """
    if not games_data or 'games' not in games_data:
        return {
            'playstyle': 'Casual',
            'avg_session_hours': 1.5,
            'preferred_genres': ['Action', 'Adventure'],
            'top_games': [],
            'total_value': 0
        }
    
    games = games_data['games']
    
    # Ordenar por tempo jogado
    sorted_games = sorted(games, key=lambda x: x['playtime_forever'], reverse=True)
    top_games = sorted_games[:10]
    
    # Mapeamento básico de gêneros (em produção, usaria uma base de dados)
    # Esta é uma simplificação - em produção, usaria a Steam Store API
    genre_mapping = {
        'Counter-Strike': ['FPS', 'Action', 'Multiplayer'],
        'Dota 2': ['MOBA', 'Strategy', 'Multiplayer'],
        'Team Fortress 2': ['FPS', 'Action', 'Multiplayer'],
        'Grand Theft Auto': ['Action', 'Open World', 'Adventure'],
        'The Witcher': ['RPG', 'Adventure', 'Story Rich'],
        'Stardew Valley': ['Simulation', 'Casual', 'Farming'],
        'Terraria': ['Adventure', 'Sandbox', 'Crafting'],
        'Left 4 Dead': ['FPS', 'Horror', 'Co-op'],
        'Portal': ['Puzzle', 'First-Person', 'Story'],
        'Half-Life': ['FPS', 'Action', 'Story']
    }
    
    # Analisar gêneros
    genre_count = {}
    for game in games[:20]:  # Analisar apenas os 20 principais
        name_lower = game['name'].lower()
        
        for keyword, genres in genre_mapping.items():
            if keyword.lower() in name_lower:
                for genre in genres:
                    genre_count[genre] = genre_count.get(genre, 0) + 1
    
    # Determinar gêneros favoritos
    if genre_count:
        preferred_genres = sorted(genre_count.items(), key=lambda x: x[1], reverse=True)[:5]
        preferred_genres = [genre for genre, _ in preferred_genres]
    else:
        preferred_genres = ['Action', 'Adventure']
    
    # Determinar estilo de jogo
    total_hours = games_data['total_playtime_hours']
    game_count = games_data['game_count']
    
    if total_hours > 1000:
        playstyle = 'Hardcore'
    elif total_hours > 300:
        playstyle = 'Moderado'
    else:
        playstyle = 'Casual'
    
    # Calcular valor estimado da biblioteca
    avg_price = 20  # USD - média estimada
    total_value = game_count * avg_price
    
    return {
        'playstyle': playstyle,
        'avg_session_hours': total_hours / max(game_count * 10, 1),  # Estimativa
        'preferred_genres': preferred_genres,
        'top_games': top_games,
        'total_value_usd': total_value,
        'game_count': game_count,
        'total_hours': total_hours
    }