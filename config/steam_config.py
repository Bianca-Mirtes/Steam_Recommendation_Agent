"""
Configuração da API Steam - Configure sua chave aqui uma vez
"""

# SUA STEAM WEB API KEY - Obtenha em: https://steamcommunity.com/dev/apikey
STEAM_API_KEY = "2B0DFBFA39ACA4D637DDA0AEB88873C1"

# Configurações da API
STEAM_API_CONFIG = {
    'api_key': STEAM_API_KEY,
    'timeout': 15,
    'max_retries': 3,
    'cache_duration': 3600  # 1 hora em segundos
}

# URLs da API
STEAM_API_URLS = {
    'player_summary': 'https://api.steampowered.com/ISteamUser/GetPlayerSummaries/v2/',
    'owned_games': 'https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/',
    'recent_games': 'https://api.steampowered.com/IPlayerService/GetRecentlyPlayedGames/v1/',
    'resolve_vanity': 'https://api.steampowered.com/ISteamUser/ResolveVanityURL/v1/',
    'friend_list': 'https://api.steampowered.com/ISteamUser/GetFriendList/v1/'
}