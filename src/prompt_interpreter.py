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
        
        # Carregar modelos
        logger.info("Carregando modelos de NLP...")
        
        # Zero-shot classification para categorias - usando modelo multilíngue
        try:
            self.zero_shot_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=self.device
            )
        except Exception as e:
            logger.warning(f"Erro ao carregar zero-shot classifier: {e}")
            self.zero_shot_classifier = None
        
        # Tente carregar spaCy para português
        try:
            self.nlp = spacy.load("pt_core_news_sm")
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
            'ação': ['ação', 'action', 'tiro', 'fps', 'shooter', 'combate', 'batalha', 'call of duty', 'battlefield', 'halo'],
            'rpg': ['rpg', 'role playing', 'papel', 'personagem', 'level', 'witcher', 'skyrim', 'final fantasy', 'dragon age'],
            'estratégia': ['estratégia', 'strategy', 'tática', 'planejamento', 'cérebro', 'civilization', 'age of empires', 'starcraft'],
            'aventura': ['aventura', 'adventure', 'exploração', 'descoberta', 'viagem', 'tomb raider', 'uncharted'],
            'simulação': ['simulação', 'simulation', 'simulador', 'realista', 'sim city', 'cities skylines'],
            'esportes': ['esporte', 'sports', 'futebol', 'fifa', 'basquete', 'nba', 'corrida', 'racing'],
            'corrida': ['corrida', 'carro', 'automobilismo', 'velocidade', 'forza', 'gran turismo'],
            'puzzle': ['puzzle', 'quebra-cabeça', 'lógica', 'enigma', 'brain teaser'],
            'luta': ['luta', 'fighting', 'street fighter', 'mortal kombat', 'tekken'],
            'plataforma': ['plataforma', 'platformer', 'mario', 'sonic', '2d platform'],
            'musical': ['musical', 'rhythm', 'ritmo', 'dance', 'dança'],
            'educacional': ['educacional', 'educational', 'aprendizado', 'learning'],
            
            # Características de gameplay
            'multijogador': ['multijogador', 'multiplayer', 'online', 'amigos', 'jogar junto', 'grupo', 'coop', 'cooperativo'],
            'cooperativo': ['cooperativo', 'cooperative', 'coop', 'co-op', 'jogar junto', 'equipe', 'juntos'],
            'competitivo': ['competitivo', 'competitive', 'pvp', 'versus', 'ranked', 'elite', 'pro'],
            'singleplayer': ['singleplayer', 'single player', 'solo', 'campanha', 'campaign', 'história solo'],
            'mundo aberto': ['mundo aberto', 'open world', 'sandbox', 'livre', 'explorar', 'gta', 'red dead', 'skyrim'],
            'história rica': ['história', 'story', 'narrativa', 'enredo', 'personagem', 'trama', 'plot', 'cinemático'],
            'sobrevivência': ['sobrevivência', 'survival', 'crafting', 'sobreviver', 'construção', 'resource gathering'],
            'roguelike': ['roguelike', 'procedural', 'permadeath', 'rogue', 'lite', 'replayability'],
            'metroidvania': ['metroidvania', 'exploração', 'backtracking', 'mapa', 'upgrades'],
            
            # Estilos e atmosfera
            'casual': ['casual', 'simples', 'fácil', 'rápido', 'acessível', 'relaxante', 'pick up and play'],
            'hardcore': ['hardcore', 'difícil', 'challenging', 'desafiador', 'complexo', 'elite', 'pro'],
            'relaxante': ['relaxante', 'calmo', 'tranquilo', 'pacífico', 'zen', 'meditative', 'chill'],
            'desafiador': ['desafiador', 'difícil', 'hardcore', 'desafio', 'challenging', 'complexo'],
            'rápido': ['rápido', 'fast', 'quick', 'ação rápida', 'fast paced', 'arcade', 'twitch'],
            'lento': ['lento', 'slow', 'paciente', 'estratégico', 'turn based', 'contemplativo'],
            'imersivo': ['imersivo', 'immersive', 'profundo', 'absorbing', 'realista', 'detalhado'],
            'criativo': ['criativo', 'creative', 'criação', 'design', 'building', 'construção', 'craft'],
            
            # Temas e ambientações
            'fantasia': ['fantasia', 'fantasy', 'medieval', 'dragões', 'magia', 'elfos', 'anões'],
            'ficção científica': ['ficção científica', 'sci-fi', 'futurista', 'espaço', 'alien', 'cyberpunk', 'distopia'],
            'histórico': ['histórico', 'historical', 'segunda guerra', 'medieval', 'antigo', 'era vitoriana'],
            'pós-apocalíptico': ['pós-apocalíptico', 'post-apocalyptic', 'apocalypse', 'zumbi', 'zombie', 'survival'],
            'cyberpunk': ['cyberpunk', 'high tech', 'low life', 'futurista', 'distopia tecnológica'],
            'steampunk': ['steampunk', 'vapor', 'retro-futurista', 'victorian technology'],
            'noir': ['noir', 'detective', 'mistery', 'suspense', 'crime'],
            
            # Aspectos técnicos e visuais
            'gráficos realistas': ['gráficos', 'graphics', 'realista', 'realistic', 'visual', '3d', 'next-gen'],
            'pixel art': ['pixel art', 'pixel', 'retro', '8-bit', '16-bit', 'old school'],
            'estilizado': ['estilizado', 'stylized', 'cartoon', 'cel-shaded', 'artístico', 'unique art'],
            'primeira pessoa': ['primeira pessoa', 'first person', 'fps', '1st person'],
            'terceira pessoa': ['terceira pessoa', 'third person', 'tps', '3rd person', 'over the shoulder'],
            'isométrico': ['isométrico', 'isometric', 'top down', 'diagonal view'],
            
            # Modelo de negócio e comunidade
            'indie': ['indie', 'independente', 'pequeno', 'independent', 'small studio'],
            'triplo A': ['triplo A', 'AAA', 'grande orçamento', 'blockbuster', 'major studio'],
            'gratuito': ['gratuito', 'free', 'grátis', 'free to play', 'f2p'],
            'pago': ['pago', 'paid', 'premium', 'buy to play', 'b2p'],
            'early access': ['early access', 'acesso antecipado', 'beta', 'alpha', 'em desenvolvimento'],
            'mods': ['mods', 'moddable', 'customizável', 'community content'],
            
            # Emoções e experiências
            'nostálgico': ['nostálgico', 'nostalgia', 'retro', 'clássico', 'old school', 'childhood'],
            'emocionante': ['emocionante', 'exciting', 'thrilling', 'adrenaline', 'intense'],
            'assustador': ['assustador', 'scary', 'fear', 'terror', 'horror'],
            'divertido': ['divertido', 'fun', 'funny', 'comédia', 'comedy', 'humor'],
            'triste': ['triste', 'sad', 'melancólico', 'emotional', 'drama'],
            'épico': ['épico', 'epic', 'grande escala', 'large scale', 'cinematic'],
            
            # Recursos específicos
            'crafting': ['crafting', 'criação', 'construção', 'building', 'fabricação'],
            'construção': ['construção', 'building', 'base building', 'city building', 'simcity'],
            'gerenciamento': ['gerenciamento', 'management', 'simulação de negócios', 'tycoon'],
            'exploração': ['exploração', 'exploration', 'descobrir', 'discovery', 'map uncovering'],
            'coleta': ['coleta', 'collecting', 'achievements', 'trophies', 'completionist'],
            'personalização': ['personalização', 'customization', 'character creator', 'outfits', 'skins'],
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
        
        # Categorias de jogos pré-definidas (em português)
        self.game_categories = [
            "ação", "aventura", "rpg", "estratégia", "simulação",
            "esportes", "corrida", "luta", "tiro", "puzzle",
            "plataforma", "musical", "educacional", "casual",
            "multijogador", "singleplayer", "cooperativo", "competitivo",
            "mundo aberto", "história rica", "gráficos realistas",
            "pixel art", "terror", "comédia", "drama", "fantasia",
            "ficção científica", "histórico", "realista", "arcade",
            "relaxante", "desafiador", "rápido", "lento", "violento",
            "família", "indie", "triplo A", "gratuito", "pago",
            "exploração", "sobrevivência", "crafting", "construção",
            "gerenciamento", "roguelike", "sandbox", "metroidvania"
        ]
        
        # Mapeamento de sinônimos
        self.synonyms = {
            'fps': ['tiro', 'first person', 'fps', 'atirador'],
            'rpg': ['rpg', 'role playing', 'papel', 'personagem'],
            'mmo': ['mmo', 'massivo', 'multijogador online', 'online'],
            'moba': ['moba', 'batalha arena', 'dota', 'league'],
            'roguelike': ['roguelike', 'procedural', 'permadeath', 'rogue'],
            'sandbox': ['sandbox', 'caixa de areia', 'livre', 'aberto'],
            'survival': ['sobrevivência', 'survival', 'crafting', 'sobreviver'],
            'horror': ['terror', 'horror', 'medo', 'susto', 'assustador'],
            'relax': ['relaxante', 'calmo', 'tranquilo', 'zen', 'pacífico'],
            'challenge': ['desafiador', 'difícil', 'hardcore', 'desafio', 'complexo'],
            'terror': ['horror', 'assustador', 'macabro'],
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
                    features[category] = max(features.get(category, 0), 0.8)
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
                    'call of duty': {'ação': 0.95, 'tiro': 0.9, 'fps': 0.9, 'multijogador': 0.85},
                    'cod': {'ação': 0.95, 'tiro': 0.9, 'fps': 0.9, 'multijogador': 0.85},
                    'halo': {'ação': 0.9, 'tiro': 0.85, 'fps': 0.85, 'multijogador': 0.8},
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
        
        # 3. Features derivadas
        #self._add_derived_features(features, prompt_text)
        
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
            reverse=False
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

    def _extract_emotions(self, text):
        """
        Extrai emoções do texto usando heurísticas simples
        """
        emotions = defaultdict(float)
        
        # Palavras-chave para emoções
        emotion_keywords = {
            'relaxing': ['relaxar', 'desestressar', 'calmo', 'tranquilo', 'zen', 'meditar', 'descansar'],
            'exciting': ['empolgado', 'emoção', 'adrenalina', 'intenso', 'rápido', 'eletrizante', 'emocionante'],
            'social': ['amigos', 'jogar junto', 'multijogador', 'coop', 'grupo', 'equipe', 'parceria'],
            'story': ['história', 'narrativa', 'enredo', 'conto', 'personagem', 'trama', 'plot'],
            'challenge': ['desafio', 'difícil', 'hardcore', 'perfeito', 'domínio', 'habilidade', 'desafiante'],
            'nostalgic': ['nostalgia', 'antigo', 'clássico', 'retro', 'infância', 'lembrança', 'saudosista'],
            'creative': ['criar', 'construir', 'design', 'inventar', 'customizar', 'personalizar', 'criativo'],
            'immersive': ['imersivo', 'profundo', 'absorvente', 'cativante', 'envolvendo', 'realista']
        }
        
        for emotion, keywords in emotion_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    emotions[emotion] += 0.3
        
        # Limitar valores
        for emotion in emotions:
            emotions[emotion] = min(emotions[emotion], 1.0)
        
        return dict(emotions)
    
    def _add_derived_features(self, features, text):
        """
        Adiciona features derivadas do contexto - GARANTIR VALORES NUMÉRICOS
        """
        # Determinar intensidade do jogo - SEMPRE float
        intense_words = ['rápido', 'veloz', 'intenso', 'adrenalina', 'competitivo', 
                        'ação', 'tiro', 'luta', 'batalha', 'frenético']
        calm_words = ['lento', 'calmo', 'tranquilo', 'paciente', 'estratégico',
                    'relaxante', 'pacífico', 'suave', 'contemplativo']
        
        intense_count = sum(1 for word in intense_words if word in text)
        calm_count = sum(1 for word in calm_words if word in text)
        
        if intense_count > calm_count:
            features['intensity'] = float(0.7 + (intense_count * 0.05))
        elif calm_count > intense_count:
            features['intensity'] = float(0.3 - (calm_count * 0.05))
        else:
            features['intensity'] = 0.5
        
        # Determinar complexidade - SEMPRE float
        complex_words = ['complexo', 'profundo', 'detalhado', 'estratégia', 'aprender',
                        'tático', 'cérebro', 'pensar', 'raciocínio', 'lógica']
        simple_words = ['simples', 'fácil', 'acessível', 'casual', 'rápido',
                    'intuitivo', 'direto', 'básico', 'iniciante']
        
        complex_score = sum(1 for word in complex_words if word in text)
        simple_score = sum(1 for word in simple_words if word in text)
        
        if complex_score > simple_score:
            features['complexity'] = float(0.7 + (complex_score * 0.05))
        elif simple_score > complex_score:
            features['complexity'] = float(0.3 - (simple_score * 0.05))
        else:
            features['complexity'] = 0.5
        
        # Determinar se é para sessões curtas ou longas
        if 'playtime_minutes' in features:
            minutes = features['playtime_minutes']
            if isinstance(minutes, (int, float)):
                if minutes <= 30:
                    features['session_length'] = 'curta'
                    features['session_length_score'] = 0.3
                elif minutes <= 120:
                    features['session_length'] = 'média'
                    features['session_length_score'] = 0.6
                else:
                    features['session_length'] = 'longa'
                    features['session_length_score'] = 0.9
        else:
            # Inferir do contexto
            if any(word in text for word in ['rápido', 'curto', 'intervalo', 'pausa']):
                features['session_length'] = 'curta'
                features['session_length_score'] = 0.3
            elif any(word in text for word in ['longo', 'férias', 'fim de semana', 'noite toda']):
                features['session_length'] = 'longa'
                features['session_length_score'] = 0.9
            else:
                features['session_length'] = 'média'
                features['session_length_score'] = 0.6
    
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
        query_text = " ".join(query_parts[:4])

        primary_categories = []
        for feature, score in prioritized_features[:3]:
            if score > 0.5:
                primary_categories.append(feature)
    
        
        return {
            'text_query': query_text,
            'features': features,
            'primary_categories': primary_categories,
            'session_length': features.get('session_length', 'média'),
        }
    
    def get_recommendation_rationale(self, prompt_text, game_name, match_features):
        """
        Gera explicação de por que um jogo foi recomendado
        
        Args:
            prompt_text: Prompt do usuário
            game_name: Nome do jogo recomendado
            match_features: Features que fizeram match (dicionário)
            
        Returns:
            rationale: Texto explicativo
        """
        features = self.extract_features(prompt_text)
        
        # Encontrar as top 3 correspondências entre features extraídas e match_features
        matches = []
        match_scores = []
        
        # Primeiro, verificar correspondências diretas
        for feature, score in features.items():
            if isinstance(score, (int, float)) and score > 0.5:
                # Verificar se feature está em match_features
                if feature in match_features:
                    matches.append(feature)
                    match_scores.append(score)
                # Verificar sinônimos
                else:
                    for category, synonyms in self.synonyms.items():
                        if feature in synonyms and category in match_features:
                            matches.append(category)
                            match_scores.append(score)
                            break
        
        # Se não encontrou correspondências, usar as principais features do jogo
        if not matches:
            if isinstance(match_features, dict):
                # Pegar as 3 features com maior valor
                top_match_features = sorted(
                    match_features.items(),
                    key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0,
                    reverse=True
                )[:3]
                matches = [feature for feature, _ in top_match_features]
        
        # Gerar explicação baseada nas correspondências
        if len(matches) >= 3:
            rationale = (f"Recomendo **{game_name}** porque você pediu algo que envolve "
                        f"**{matches[0]}**, **{matches[1]}** e **{matches[2]}**.")
        elif len(matches) == 2:
            rationale = (f"Recomendo **{game_name}** pois combina com seu interesse em "
                        f"**{matches[0]}** e **{matches[1]}**.")
        elif len(matches) == 1:
            rationale = (f"**{game_name}** é perfeito para seu interesse em "
                        f"**{matches[0]}**.")
        else:
            # Fallback genérico
            rationale = (f"**{game_name}** é um jogo popular que combina bem com o que você está procurando.")
        
        # Adicionar contexto de tempo se relevante
        if 'playtime_minutes' in features:
            time = features['playtime_minutes']
            if time <= 30:
                rationale += " É ótimo para sessões curtas de até 30 minutos."
            elif time <= 90:
                rationale += " Ideal para sessões de 1-2 horas."
            else:
                rationale += " Perfeito para longas sessões de jogo."
        elif 'session_length' in features:
            session_type = features['session_length']
            if session_type == 'curta':
                rationale += " Perfeito para sessões curtas e rápidas."
            elif session_type == 'longa':
                rationale += " Ideal para sessões longas e imersivas."
        
        # Adicionar contexto de intensidade/complexidade
        if 'intensity' in features:
            intensity = features['intensity']
            if intensity > 0.7:
                rationale += " Oferece ação intensa e adrenalina."
            elif intensity < 0.3:
                rationale += " Tem um ritmo calmo e relaxante."
        
        if 'complexity' in features:
            complexity = features['complexity']
            if complexity > 0.7:
                rationale += " Possui profundidade estratégica."
            elif complexity < 0.3:
                rationale += " É acessível e fácil de aprender."
        
        return rationale