import re
import numpy as np
from transformers import pipeline
import spacy
from collections import defaultdict
import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PromptInterpreter:
    def __init__(self, use_gpu=True):
        """
        Inicializa o interpretador de prompts
        
        Args:
            use_gpu: Se True, usa GPU quando disponível
        """
        self.device = 0 if torch.cuda.is_available() and use_gpu else -1
        
        # Tente carregar spaCy para português
        try:
            self.nlp = spacy.load("pt_core_news_md")
            logger.info("spaCy carregado para português")
        except:
            logger.warning("spaCy para português não encontrado, usando regex apenas")
            self.nlp = None

        self.combined_features = None
        # ============== ADIÇÕES IMPORTANTES ==============
        # FEATURES ESSENCIAIS DE JOGOS (prioridade alta)
        self.feature_map = {
            # Gêneros principais
            'terror': ['terror', 'horror', 'assustador', 'susto', 'medo', 'pesadelo', 'macabro'],
            'action': ['ação', 'action', 'combate'],
            'rpg': ['rpg', 'role playing', 'papel', 'personagem', 'level', 'witcher', 'skyrim', 'final fantasy', 'dragon age'],
            'strategy': ['estratégia', 'strategy', 'tática', 'planejamento', 'cérebro', 'civilization', 'age of empires'],
            'adventure': ['aventura', 'adventure', 'exploração', 'descoberta', 'viagem', 'tomb raider', 'uncharted'],
            'simulação': ['simulação', 'simulation', 'simulador', 'realista', 'sim city', 'cities skylines'],
            'sports': ['esportes', 'sports', 'futebol', 'fifa', 'basquete', 'nba', 'pes'],
            'racing': ['corrida', 'carro', 'racing', 'velocidade', 'forza', 'gran turismo'],
            'puzzle': ['puzzle', 'quebra-cabeça', 'lógica', 'enigma', 'brain teaser'],
            'fighting': ['luta', 'fighting', 'street fighter', 'mortal kombat', 'tekken'],
            'platformer': ['plataforma', 'platformer', 'mario', 'sonic', '2d platform'],
            'rhythm': ['musicais', 'rhythm', 'ritmo', 'dance', 'dança'],
            'educational': ['educacional', 'educational', 'aprendizado', 'learning'],
            
            # Características de gameplay
            'multiplayer': ['multijogador', 'multiplayer', 'online', 'amigos', 'jogar junto', 'grupo', 'coop', 'cooperativo'],
            'cooperative': ['cooperativo', 'cooperative', 'coop', 'co-op', 'jogar junto', 'equipe', 'juntos'],
            'competitivo': ['competitivo', 'competitive', 'pvp', 'versus', 'ranked', 'elite', 'pro'],
            'mundo aberto': ['mundo aberto', 'open world', 'sandbox', 'livre', 'explorar', 'gta', 'red dead', 'skyrim'],
            'history': ['história', 'story', 'narrative', 'enredo', 'personagem', 'trama', 'good plot', 'cinemático'],
            'sobrevivência': ['sobrevivência', 'survival', 'crafting', 'sobreviver', 'construção', 'resource gathering'],
            'roguelike': ['roguelike', 'procedural', 'permadeath', 'rogue', 'lite', 'replayability'],
            'metroidvania': ['metroidvania', 'exploração', 'backtracking', 'mapa', 'upgrades'],
            
            # Estilos e atmosfera
            'casual': ['casual', 'simples', 'fácil', 'rápido', 'acessível', 'relaxante', 'pick up and play'],
            'hardcore': ['hardcore', 'difícil', 'challenging', 'desafiador', 'complexo', 'elite', 'pro'],
            'relaxante': ['relaxante', 'calmo', 'tranquilo', 'pacífico', 'zen', 'meditative', 'chill'],
            'challenging': ['desafiador', 'difícil', 'hardcore', 'desafio', 'challenging', 'complexo'],
            'imersivo': ['imersivo', 'immersive', 'profundo', 'absorbing', 'realista', 'detalhado'],
            'creative': ['criativo', 'creative', 'criação', 'design', 'building', 'construção', 'craft'],
            
            # Temas e ambientações
            'fantasy': ['fantasia', 'fantasy', 'medieval', 'dragões', 'magia', 'elfos', 'anões'],
            'sci-fi': ['ficção científica', 'sci-fi', 'futurista', 'espaço', 'alien', 'cyberpunk', 'distopia'],
            'historical': ['histórico', 'historical', 'segunda guerra', 'medieval', 'antigo', 'era vitoriana'],
            'post-apocalyptic': ['pós-apocalíptico', 'post-apocalyptic', 'apocalypse', 'zumbi', 'zombie', 'survival'],
            'cyberpunk': ['cyberpunk', 'high tech', 'low life', 'futurista', 'distopia tecnológica'],
            'steampunk': ['steampunk', 'vapor', 'retro-futurista', 'victorian technology'],
            'noir': ['noir', 'detective', 'mistery', 'suspense', 'crime'],
            
            'realistic': ['realista', 'realistic', 'visual', '3d', 'next-gen'],
            'pixel art': ['pixel art', 'pixel', 'retro', '8-bit', '16-bit', 'old school'],
            'cartoon': ['estilizado', 'stylized', 'cartoon', 'cel-shaded', 'artístico', 'unique art'],
            'first person': ['primeira pessoa', 'first person', 'fps', '1st person'],
            'third person': ['terceira pessoa', 'third person', 'tps', '3rd person', 'over the shoulder'],
            'isometric': ['isométrico', 'isometric', 'top down', 'diagonal view'],
            
            'indie': ['indie', 'independente', 'pequeno', 'independent', 'small studio'],
            'triple A': ['triplo A', 'AAA', 'grande orçamento', 'blockbuster', 'major studio'],
            'free': ['gratuito', 'free', 'grátis', 'free to play', 'f2p'],
            'early access': ['early access', 'acesso antecipado', 'beta', 'alpha', 'em desenvolvimento'],
            'sad': ['triste', 'sad', 'melancólico', 'emotional', 'drama'],
            'épico': ['épico', 'epic', 'grande escala', 'large scale', 'cinematic'],
            
            'crafting': ['crafting', 'criação', 'construção', 'building', 'fabricação'],
            'building': ['construção', 'building', 'base building', 'city building', 'simcity'],
            'management': ['gerenciamento', 'management', 'simulação de negócios', 'tycoon'],
            'discovery': ['exploração', 'exploration', 'descobrir', 'discovery', 'map uncovering'],
            'customization': ['personalização', 'customization', 'character creator', 'outfits', 'skins'],
        }
        
        # COMBINAÇÕES POPULARES (para boost automático)
        self.popular_combinations = {
            ('terror', 'multijogador'): ['coop horror', 'online horror', 'multiplayer horror'],
            ('terror', 'cooperativo'): ['coop horror', 'team horror', 'survival horror coop'],
            ('ação', 'multijogador'): ['coop shooter', 'online action', 'multiplayer fps'],
            ('rpg', 'multijogador'): ['mmo', 'online rpg', 'multiplayer rpg'],
            ('estratégia', 'multijogador'): ['multiplayer strategy', 'online strategy', 'competitive strategy'],
            ('mundo aberto', 'rpg'): ['open world rpg', 'sandbox rpg'],
            ('sobrevivência', 'crafting'): ['survival crafting', 'base building survival'],
        }
        
        # Mapeamento de sinônimos
        self.synonyms = {
            'fps': ['jogo de tiro', 'first person', 'fps', 'atirador'],
            'rpg': ['rpg', 'role playing', 'papel', 'personagem'],
            'mmo': ['mmo', 'massivo', 'multijogador online', 'online'],
            'moba': ['moba', 'batalha arena', 'dota', 'league'],
            'roguelike': ['roguelike', 'procedural', 'permadeath', 'rogue'],
            'sandbox': ['sandbox', 'caixa de areia', 'livre', 'aberto'],
            'survival': ['sobrevivência', 'survival', 'crafting', 'sobreviver'],
            'horror': ['terror', 'horror', 'medo', 'susto', 'assustador'],
            'relax': ['relaxante', 'calmo', 'tranquilo', 'zen', 'pacífico'],
            'challenge': ['desafiador', 'difícil', 'hardcore', 'desafio', 'complexo'],
            'multijogador': ['multiplayer', 'online', 'coop'],
            'cooperativo': ['coop', 'co-op', 'team'],
            'ação': ['action', 'tiro', 'shooter'],
            'indie': ['independente', 'small studio'],
            'gratuito': ['free', 'free to play']
        }
        
        # Padrões regex para extração
        self.patterns = {
            'time': r'(\d+)\s*(minutos?|min|horas?|h|hrs?)',
            'price': r'(\d+[.,]?\d*)\s*(reais?|R\$|dolares?|\$|usd)',
            'players': r'(\d+)\s*jogadores?',
            'rating': r'classificação\s*(\d+|\w+)',
            'genre': r'(?:gênero|tipo)\s+(?:de\s+)?(\w+)'
        }
    
    def extract_features(self, prompt_text):
        """
        Extrai features do prompt do usuário - VERSÃO FILTRADA
        """
        prompt_text = prompt_text.lower().strip()
        features = defaultdict(float)

        # 1. Extrair combinações
        combinations = self._extract_combinations(prompt_text)
        self.combined_features = combinations
        
        # EXTRAÇÃO POR PALAVRAS-CHAVE (heurística robusta)
        for category, keywords in self.feature_map.items():
            for keyword in keywords:
                if keyword in prompt_text:
                    # Boost para correspondências exatas
                    features[category] = max(features.get(category, 0), 0.7)
                    break

        # busca por sinonimos
        for key, syns in self.synonyms.items():
            for syn in syns:
                if syn in prompt_text:
                    features[key] = max(features.get(key, 0), 0.7)
                    break
        
        # Se mencionou "estilo [jogo]", adicionar características desse jogo 
        if 'estilo' in prompt_text or 'tipo' in prompt_text or 'como' in prompt_text:
            # Padrões para detectar jogos mencionados
            patterns = [
                r'estilo\s+([\w\s]+)',
                r'tipo\s+([\w\s]+)', 
                r'como\s+([\w\s]+)',
                r'parecido com\s+([\w\s]+)'
            ]
            
            mentioned_game = None
            for pattern in patterns:
                match = re.search(pattern, prompt_text)
                if match:
                    mentioned_game = match.group(1).lower()
                    break
            
            if mentioned_game:
                # Mapear jogos para features (você pode expandir isso)
                game_features_map = {
                    'devour': {'terror': 0.95, 'multijogador': 0.8, 'cooperativo': 0.8, 'ação': 0.7},
                    'call of duty': {'ação': 0.8, 'tiro': 0.95, 'fps': 0.95, 'multijogador': 0.85},
                    'cod': {'ação': 0.95, 'tiro': 0.9, 'fps': 0.9, 'multijogador': 0.85},
                    'halo': {'ação': 0.8, 'tiro': 0.9, 'fps': 0.9, 'multijogador': 0.8},
                    'the witcher': {'rpg': 0.95, 'aventura': 0.9, 'história rica': 0.9, 'mundo aberto': 0.8},
                    'skyrim': {'rpg': 0.9, 'aventura': 0.85, 'mundo aberto': 0.9, 'fantasia': 0.8},
                    'stardew valley': {'simulação': 0.9, 'casual': 0.8, 'relaxante': 0.7, 'mundo aberto': 0.8},
                    'minecraft': {'sandbox': 0.95, 'criativo': 0.9, 'multijogador': 0.8, 'aventura': 0.7},
                    'among us': {'multijogador': 0.95, 'casual': 0.8, 'social': 0.9, 'puzzle': 0.6},
                    'phasmophobia': {'terror': 0.95, 'multijogador': 0.9, 'cooperativo': 0.9, 'investigação': 0.8},
                    'left 4 dead': {'ação': 0.9, 'terror': 0.8, 'multijogador': 0.9, 'cooperativo': 0.9},
                    'counter strike': {'ação': 0.95, 'tiro': 0.9, 'fps': 0.9, 'competitivo': 0.9},
                    'cs': {'ação': 0.95, 'tiro': 0.9, 'fps': 0.9, 'competitivo': 0.9},
                    'fortnite': {'ação': 0.9, 'battle royale': 0.95, 'multijogador': 0.9, 'competitivo': 0.8},
                    'dota': {'estratégia': 0.9, 'moba': 0.95, 'competitivo': 0.95, 'multijogador': 0.9},
                    'league of legends': {'estratégia': 0.9, 'moba': 0.95, 'competitivo': 0.95, 'multijogador': 0.9},
                    'lol': {'estratégia': 0.9, 'moba': 0.95, 'competitivo': 0.95, 'multijogador': 0.9},
                    'valorant': {'ação': 0.9, 'fps': 0.9, 'competitivo': 0.9, 'multijogador': 0.85},
                    'overwatch': {'ação': 0.9, 'fps': 0.9, 'multijogador': 0.9, 'competitivo': 0.8},
                }
                
                # Encontrar o jogo mais parecido
                best_match = None
                best_score = 0
                
                for game_name, game_features in game_features_map.items():
                    # Verificar similaridade (pode ser simples matching)
                    if game_name in mentioned_game or mentioned_game in game_name:
                        similarity = len(game_name) / max(len(game_name), len(mentioned_game))
                        if similarity > best_score:
                            best_score = similarity
                            best_match = game_name
                
                if best_match and best_score > 0.6:
                    game_features = game_features_map[best_match]
                    for feature, score in game_features.items():
                        features[feature] = max(features.get(feature, 0), score)

        # fallback para spaCy se nada foi extraído
        if not features:
            doc = self.nlp(prompt_text)

            for feature, keywords in self.feature_map.items():
                for keyword in keywords:
                    sim = doc.similarity(self.nlp(keyword))
                    if sim > 0.75:
                        features[feature] = max(features.get(feature, 0), sim * 0.2)

        # 5. Normalização
        if features:
            # Separar valores numéricos
            numeric_features = {}
            for key, value in features.items():
                if isinstance(value, (int, float)):
                    numeric_features[key] = float(value)
            
            if numeric_features:
                max_score = max(numeric_features.values())
                if max_score > 0:
                    for key in numeric_features:
                        numeric_features[key] = numeric_features[key] / max_score
                
                # Atualizar features com valores normalizados
                features.update(numeric_features)
        
        # 6. Log apenas das top 5 features
        top_features = sorted(
            [(k, v) for k, v in features.items() if isinstance(v, (int, float))],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        logger.info(f"Top features extraídas: {dict(top_features)}")
        return dict(top_features)


    def _extract_combinations(self, text):
        combinations = {}
        
        # Para cada combinação popular, verificar se os termos aparecem no texto
        for (feat1, feat2), keywords in self.popular_combinations.items():
            # Verificar se ambos os features aparecem no texto (ou sinônimos)
            feat1_keywords = self.feature_map.get(feat1, [])
            feat2_keywords = self.feature_map.get(feat2, [])
            
            # Verificar se há pelo menos uma keyword de cada feature
            has_feat1 = any(keyword in text for keyword in feat1_keywords)
            has_feat2 = any(keyword in text for keyword in feat2_keywords)
            
            if has_feat1 and has_feat2:
                # Usamos a tupla (feat1, feat2) como chave
                combinations[(feat1, feat2)] = 0.9  # Score alto para combinações
        
        return combinations
    
    def interpret_to_query(self, prompt_text, top_n=5):
        """
        Converte prompt em query de busca - CORRIGIDO
        """
        features = self.extract_features(prompt_text)
        
        # Filtrar apenas features numéricas e ordenar
        numeric_features = [(k, v) for k, v in features.items() 
                        if isinstance(v, (int, float))]
        
        # Ordenar por score
        sorted_features = sorted(numeric_features, key=lambda x: x[1], reverse=True)

        # Priorizar features essenciais
        essential_features = []
        other_features = []
        
        for feature, score in sorted_features:
            if feature in self.feature_map:
                essential_features.append((feature, score))
            else:
                other_features.append((feature, score))
        
        # Combinar: essenciais primeiro, depois outras
        prioritized_features = essential_features[:top_n]
        if len(prioritized_features) < top_n:
            prioritized_features.extend(other_features[:top_n - len(prioritized_features)])
        
        # Criar query: priorizar as duas primeiras features, se houver
        query_parts = []
        for feature, score in prioritized_features:
            if score > 0.5:
                query_parts.append(feature)
        
        # Query final: limitar a 4 termos, mas priorizando as combinações
        query_text = " ".join(query_parts[:5])

        primary_categories = []
        for feature, score in prioritized_features[:5]:
            if score > 0.5:
                primary_categories.append(feature)
    
        return {
            'text_query': query_text,
            'features': features,
            'primary_categories': primary_categories,
            'session_length': features.get('session_length', 'média'),
        }