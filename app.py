import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import json
import pickle

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent / "src"))

# Configuração da página
st.set_page_config(
    page_title="🎮 Steam Recommendation Agent",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .game-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 5px solid #1E88E5;
    }
    .recommendation-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<h1 class="main-header">🎮 Steam Game Recommendation Agent</h1>', 
            unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn.akamai.steamstatic.com/store/home/store_home_share.jpg", 
             use_column_width=True)
    
    st.markdown("### ⚙️ Configurações")
    
    # Estratégia de recomendação
    strategy = st.selectbox(
        "Estratégia de Recomendação",
        ["Híbrida", "Colaborativa", "Baseada em Conteúdo", "Contextual"],
        help="Híbrida combina todas as abordagens"
    )
    
    # Número de recomendações
    n_recommendations = st.slider(
        "Número de Recomendações",
        min_value=3,
        max_value=10,
        value=5
    )
    
    # Configurações avançadas
    with st.expander("⚙️ Configurações Avançadas"):
        diversity_weight = st.slider(
            "Peso de Diversidade",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            help="Valores mais altos geram recomendações mais diversas"
        )
        
        min_confidence = st.slider(
            "Confiança Mínima",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            help="Filtra recomendações com baixa confiança"
        )
    
    st.markdown("---")
    
    # Informações do sistema
    st.markdown("### 📊 Status do Sistema")
    
    # Carregar metadados
    try:
        with open("data/processed/metadata.json", "r") as f:
            metadata = json.load(f)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🎮 Jogos", f"{metadata.get('num_games', 0):,}")
        with col2:
            st.metric("👥 Usuários", f"{metadata.get('num_users', 0):,}")
    except:
        st.info("Execute o pipeline de dados primeiro")
    
    st.markdown("---")
    st.markdown("Desenvolvido com ❤️ usando:")
    st.markdown("- 🤖 Transformers")
    st.markdown("- 🔍 FAISS")
    st.markdown("- 📊 Streamlit")
    st.markdown("- 🎮 Dados do Steam")

# Funções auxiliares
@st.cache_resource
def load_models():
    """Carrega todos os modelos necessários"""
    try:
        # Carregar dados processados
        games_df = pd.read_pickle("data/processed/games_processed.pkl")
        
        # Carregar embeddings
        from src.game_embedder import GameEmbedder
        embedder = GameEmbedder()
        # Carregar embeddings e índice (usando o novo formato)
        embedder.load_all("data/embeddings")
        
        # Carregar índice FAISS
        import faiss
        faiss_index = faiss.read_index("data/embeddings/game_index.faiss")
        
        # Carregar modelo de perfil
        from src.profile_analyzer import ProfileAnalyzer
        analyzer = ProfileAnalyzer()
        with open("models/profile_model.pkl", "rb") as f:
            analyzer = pickle.load(f)
        
        # Carregar interpretador de prompt
        from src.prompt_interpreter import PromptInterpreter
        interpreter = PromptInterpreter()
        
        # Carregar agente
        from src.agent_orchestrator import RecommendationAgent, RecommendationStrategy
        from enum import Enum
        
        # Mapear estratégia
        strategy_map = {
            "Híbrida": RecommendationStrategy.HYBRID,
            "Colaborativa": RecommendationStrategy.COLLABORATIVE,
            "Baseada em Conteúdo": RecommendationStrategy.CONTENT_BASED,
            "Contextual": RecommendationStrategy.CONTEXTUAL
        }
        
        agent = RecommendationAgent(
            profile_analyzer=analyzer,
            game_embedder=embedder,
            prompt_interpreter=interpreter,
            games_df=games_df,
            strategy=strategy_map[strategy]
        )
        
        return {
            'games_df': games_df,
            'embedder': embedder,
            'analyzer': analyzer,
            'interpreter': interpreter,
            'agent': agent,
            'faiss_index': faiss_index
        }
    except Exception as e:
        st.error(f"Erro ao carregar modelos: {e}")
        return None

# Usar session_state para manter um contador único
if 'widget_counter' not in st.session_state:
    st.session_state.widget_counter = 0

def create_user_profile_form():
    st.markdown('<h3 class="sub-header">🔗 Conectar à Sua Conta Steam</h3>', 
                unsafe_allow_html=True)
    
    # Container principal
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Usar instance_id para tornar a chave única
            steam_input = st.text_input(
                "Digite seu SteamID, Vanity URL ou URL do perfil:",
                value=st.session_state.get('steam_input', ''),
                placeholder="Ex: 76561197960287930 ou https://steamcommunity.com/id/seunome",
                help="Você pode encontrar seu SteamID em https://steamid.io/",
                key="steam_input_field"  # Chave única
            )
            
            # Atualizar session_state
            if steam_input != st.session_state.get('steam_input', ''):
                st.session_state.steam_input = steam_input
            
            # Exemplos clicáveis
            st.caption("💡 Exemplos: `76561197960287930` ou `https://steamcommunity.com/id/gabeloganneweller`")
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            connect_button = st.button(
                "🎮 Conectar à Minha Steam",
                type="primary",
                use_container_width=True,
                disabled=not steam_input,
                key="connect_button"  # Chave única
            )
    
    # Se clicou para conectar
    if connect_button and steam_input:
        st.info(f"🔍 Tentando conectar com: {steam_input}")
        result = process_steam_connection(steam_input)
        
        # DEBUG
        if result:
            st.success("✅ Perfil obtido com sucesso!")
        else:
            st.error("❌ Falha ao obter perfil")
            
        return result

    # Mostrar instruções se ainda não tentou conectar
    if not st.session_state.get('form_submitted', False):
        show_instructions()

    return None

def show_instructions():
    """Mostra instruções de uso"""
    st.info("""
        ### 🎮 Como usar:
        1. **Cole seu SteamID ou URL do perfil** acima
        2. **Clique em "Conectar à Minha Steam"**
        3. **Descreva o que quer jogar** na caixa de texto abaixo
        4. **Receba recomendações personalizadas** baseadas nos seus jogos!
        
        ⚠️ *Seu perfil precisa ser público para análise completa.*
        """)

def process_steam_connection(steam_input):
    """Processa a conexão com a Steam"""
    # Importar API
    try:
        from src.steam_api_client import SteamAPI, analyze_gaming_profile
        
        # Status visual
        status_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        # Etapa 1: Inicializar API
        status_placeholder.markdown("**🔐 Conectando ao servidor Steam...**")
        progress_bar.progress(10)
        
        steam_api = SteamAPI()  # Usa API key configurada automaticamente
        
        # Etapa 2: Processar input do usuário
        status_placeholder.markdown("**🔍 Identificando seu perfil...**")
        progress_bar.progress(30)
        
        steam_id = steam_api.extract_steam_id(steam_input)
        
        if not steam_id:
            st.error("❌ Não foi possível identificar sua conta Steam. Verifique o formato.")
            return get_fallback_profile()

        st.session_state.connected = True
        st.session_state.steam_id = steam_id
        
        # Etapa 3: Buscar dados do perfil
        status_placeholder.markdown("**📡 Buscando suas informações...**")
        progress_bar.progress(50)
        
        player_data = steam_api.get_player_summary(steam_id)
        
        if not player_data:
            st.error("❌ Perfil não encontrado ou privado.")
            return get_fallback_profile(steam_id)
        
        # Etapa 4: Buscar biblioteca de jogos
        status_placeholder.markdown("**🎮 Analisando sua biblioteca...**")
        progress_bar.progress(70)
        
        games_data = steam_api.get_owned_games(steam_id)
        
        if not games_data or games_data['game_count'] == 0:
            st.warning("⚠️ Sua biblioteca está vazia ou privada. Usando modo de demonstração.")
            return get_fallback_profile(steam_id, player_data)
        
        # Etapa 5: Analisar perfil
        status_placeholder.markdown("**📊 Criando seu perfil de jogador...**")
        progress_bar.progress(90)
        
        profile_analysis = analyze_gaming_profile(games_data)
        
        # Etapa 6: Finalizar
        progress_bar.progress(100)
        status_placeholder.empty()
        progress_bar.empty()
        
        # Mostrar boas-vindas personalizada
        st.success(f"✨ **Bem-vindo(a), {player_data['personaname']}!**")
        
        # Preparar dados para o modelo de recomendação
        # Agora usando appid (números) em vez de nomes
        library_appids = [game['appid'] for game in games_data['games'][:150]]  # Limitar para performance
        
        playtimes_dict = {}
        for game in games_data['games'][:100]:  # Top 100 jogos
            if game['playtime_forever'] > 0:
                playtimes_dict[game['appid']] = game['playtime_forever'] // 60
        
        # Retornar perfil completo
        user_profile = {
            'user_id': steam_id,
            'persona_name': player_data['personaname'],
            'profile_url': player_data['profileurl'],
            'avatar': player_data.get('avatar', ''),
            'playstyle': profile_analysis['playstyle'],
            'avg_playtime': games_data['total_playtime_hours'] / max(games_data['game_count'], 1),
            'favorite_genre': profile_analysis['preferred_genres'],
            'user_library': library_appids,  # Agora é lista de appids
            'playtimes': playtimes_dict,     # Agora mapeia appid -> horas
            'total_hours': games_data['total_playtime_hours'],
            'game_count': games_data['game_count'],
            'profile_data': player_data  # Dados brutos para referência
        }

        # Mostrar resumo do perfil
        show_profile_summary(user_profile, games_data, profile_analysis)

        # Limpar estado de submissão
        st.session_state.form_submitted = False

        return user_profile

    except ValueError as e:
        # Erro de API key não configurada
        st.error(f"❌ {str(e)}")
        st.session_state.form_submitted = False
        st.info("""
        **Para o desenvolvedor:** Configure sua Steam Web API Key:
        1. Acesse https://steamcommunity.com/dev/apikey
        2. Crie uma chave
        3. Adicione no arquivo `config/steam_config.py`
        """)
        return get_fallback_profile()
    
    except Exception as e:
        st.error(f"❌ Erro na conexão: {str(e)}")
        st.session_state.form_submitted = False
        st.info("Verifique sua conexão ou tente novamente em alguns instantes.")
        return get_fallback_profile()

def show_profile_summary(user_profile, games_data, profile_analysis):
    # Seção de resumo do perfil
    with st.expander(f"👤 Seu Perfil Steam - {user_profile['persona_name']}", expanded=True):
        # Métricas
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🎮 Jogos", games_data['game_count'])
        
        with col2:
            total_hours = games_data['total_playtime_hours']
            st.metric("⏱️ Horas Jogadas", f"{total_hours:,}")
        
        with col3:
            st.metric("🎯 Estilo", profile_analysis['playstyle'])
        
        # Gêneros favoritos
        if profile_analysis['preferred_genres']:
            st.markdown("**🌟 Seus Gêneros Preferidos:**")
            tags = " ".join([f"`{genre}`" for genre in profile_analysis['preferred_genres'][:5]])
            st.markdown(tags)
        
        # Top jogos
        st.markdown("**🏆 Seus Jogos Mais Jogados:**")
        
        # Preparar top jogos
        top_games = []
        for game in games_data['games'][:10]:  # Top 10
            if game['playtime_forever'] > 0:
                hours = game['playtime_forever'] // 60
                top_games.append({
                    'name': game['name'],
                    'hours': hours
                })
        
        # Ordenar por horas
        top_games.sort(key=lambda x: x['hours'], reverse=True)
        
        # Mostrar como gráfico de barras
        if top_games:
            import plotly.express as px
            top_df = pd.DataFrame(top_games[:5])
            
            fig = px.bar(
                top_df,
                x='hours',
                y='name',
                orientation='h',
                title='Top 5 Jogos por Tempo',
                color='hours',
                color_continuous_scale='Blues'
            )
            
            fig.update_layout(
                height=300,
                showlegend=False,
                yaxis_title="",
                xaxis_title="Horas Jogadas"
            )
            
            st.plotly_chart(fig, use_container_width=True)

def get_fallback_profile(steam_id=None, player_data=None):
    """Retorna um perfil de demonstração quando a API falha"""
    # Usando appids reais em vez de nomes
    fallback_appids = [
        730,    # Counter-Strike 2
        570,    # Dota 2
        271590, # GTA V
        292030, # The Witcher 3
        413150, # Stardew Valley
        105600, # Terraria
        550,    # Left 4 Dead 2
        620,    # Portal 2
        4000,   # Garry's Mod
        10,     # Counter-Strike
    ]
    
    return {
        'user_id': steam_id or 'user_demo',
        'persona_name': player_data['personaname'] if player_data else 'Demo User',
        'playstyle': 'Moderado',
        'avg_playtime': 15,
        'favorite_genre': ['Action', 'Adventure'],
        'user_library': fallback_appids,  # Lista de appids
        'playtimes': {
            730: 350,    # Counter-Strike 2
            570: 220,    # Dota 2
            271590: 85,  # GTA V
            292030: 120, # The Witcher 3
            413150: 65,  # Stardew Valley
        },
        'total_hours': 1500,
        'game_count': 50
    }

def display_recommendation_metrics(recommendations):
    """Exibe métricas das recomendações"""
    if not recommendations:
        return
    
    # Garantir que scores são numéricos
    scores = []
    for rec in recommendations:
        score = rec.score
        if isinstance(score, str):
            try:
                score = float(score)
            except (ValueError, TypeError):
                score = 0.0
        elif not isinstance(score, (int, float, np.number)):
            score = 0.0
        scores.append(score)

    sources = [rec.source for rec in recommendations]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_score = np.mean(scores)
        st.markdown(f'<div class="metric-card">'
                   f'<h3>📊</h3>'
                   f'<h4>{avg_score:.2f}</h4>'
                   f'<p>Score Médio</p>'
                   f'</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'<div class="metric-card">'
                   f'<h3>🎯</h3>'
                   f'<h4>{len(recommendations)}</h4>'
                   f'<p>Recomendações</p>'
                   f'</div>', unsafe_allow_html=True)
    
    with col3:
        unique_sources = len(set(sources))
        st.markdown(f'<div class="metric-card">'
                   f'<h3>🔄</h3>'
                   f'<h4>{unique_sources}</h4>'
                   f'<p>Fontes</p>'
                   f'</div>', unsafe_allow_html=True)
    
    with col4:
        diversity = 1.0 if len(set(sources)) > 1 else 0.0
        st.markdown(f'<div class="metric-card">'
                   f'<h3>🌈</h3>'
                   f'<h4>{diversity:.1f}</h4>'
                   f'<p>Diversidade</p>'
                   f'</div>', unsafe_allow_html=True)

def visualize_recommendations(recommendations):
    """Cria visualizações para as recomendações"""
    if not recommendations:
        return
    
    # Criar DataFrame para visualização
    df = pd.DataFrame([{
        'name': rec.game_name,
        'score': rec.score,
        'source': rec.source,
        'game_id': rec.game_id
    } for rec in recommendations])
    
    # Gráfico de barras
    fig_bar = px.bar(
        df,
        x='name',
        y='score',
        color='source',
        title='📈 Scores das Recomendações',
        labels={'name': 'Jogo', 'score': 'Score', 'source': 'Fonte'},
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig_bar.update_layout(
        xaxis_tickangle=-45,
        height=400,
        showlegend=True
    )
    
    # Gráfico de radar (apenas se houver features)
    try:
        features_list = []
        for rec in recommendations:
            if hasattr(rec, 'match_features'):
                features_list.append({
                    'name': rec.game_name,
                    **rec.match_features
                })
        
        if features_list:
            features_df = pd.DataFrame(features_list)
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) >= 3:
                fig_radar = go.Figure()
                
                for i, row in features_df.iterrows():
                    fig_radar.add_trace(go.Scatterpolar(
                        r=[row[col] for col in numeric_cols[:5]],
                        theta=numeric_cols[:5],
                        fill='toself',
                        name=row['name']
                    ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    showlegend=True,
                    title='📊 Análise de Features',
                    height=500
                )
                
                st.plotly_chart(fig_radar, use_container_width=True)
    except:
        pass
    
    st.plotly_chart(fig_bar, use_container_width=True)

def main():
    """Função principal da aplicação"""  
    # Inicializar estados importantes primeiro
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.user_profile = None
        st.session_state.steam_input = ""
        st.session_state.form_submitted = False
    
    # Carregar modelos (apenas uma vez)
    if 'models_loaded' not in st.session_state:
        with st.spinner("Carregando sistema de recomendação..."):
            models = load_models()
            if models:
                st.session_state.models = models
                st.session_state.models_loaded = True
            else:
                st.error("Sistema não inicializado. Execute `python main.py` primeiro.")
                return
    
    # VERIFICAÇÃO DE API KEY (apenas uma vez)
    if 'api_configured' not in st.session_state:
        try:
            from src.steam_api_client import SteamAPI
            # Teste silencioso da API key
            test_api = SteamAPI()
            st.session_state.api_configured = True
        except ValueError:
            st.session_state.api_configured = False
            st.warning("""
            ⚠️ **API Steam não configurada** 
            
            Para usar a integração completa com a Steam:
            1. Crie o arquivo `config/steam_config.py`
            2. Adicione: `STEAM_API_KEY = "SUA_CHAVE_AQUI"`
            3. Reinicie o app
            
            *Usando modo de demonstração por enquanto.*
            """)

    models = st.session_state.models
    
    # Se já tem perfil, mostrar conteúdo principal
    if st.session_state.user_profile:
        user_profile = st.session_state.user_profile
    
        # Mostrar que está conectado
        st.success(f"✅ Conectado como **{user_profile['persona_name']}**")
        
        # Criar tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 Recomendações", 
            "📊 Análise", 
            "🧠 Como Funciona",
            "📈 Dashboard"
        ])
        
        with tab1:
            st.markdown('<h2 class="sub-header">🎮 Obter Recomendações</h2>', 
                    unsafe_allow_html=True)
                    # Em vez disso, mostrar informações do usuário já conectado
            with st.expander("👤 Seu Perfil Steam", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🎮 Jogos", user_profile.get('game_count', 0))
                with col2:
                    if 'total_hours' in user_profile:
                        st.metric("⏱️ Horas", f"{user_profile['total_hours']:,}")
                with col3:
                    st.metric("🎯 Estilo", user_profile.get('playstyle', 'Moderado'))
                
                # Botão para reconectar se necessário
                if st.button("🔁 Reconectar à Steam", key="reconnect_button"):
                    st.session_state.pop('user_profile', None)
                    st.session_state.pop('user_profile_loaded', None)
                    st.rerun()
            
            # Prompt do usuário
            st.markdown("### 💭 O que você quer jogar hoje?")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                prompt = st.text_area(
                    "Descreva o que está procurando:",
                    value="Quero um jogo relaxante para jogar por 1 hora, singleplayer, com boa história",
                    height=100,
                    help="Seja específico! Ex: 'jogo competitivo rápido para jogar com amigos'"
                )
            
            with col2:
                st.markdown("<br><br>", unsafe_allow_html=True)
                generate_button = st.button(
                    "🎮 Gerar Recomendações",
                    type="primary",
                    use_container_width=True
                )
            
            # Exemplos de prompts
            with st.expander("💡 Exemplos de prompts"):
                examples = [
                    "Jogo de estratégia complexo para jogar nas férias",
                    "Algo rápido e casual para jogar no intervalo do trabalho",
                    "RPG com mundo aberto e muita exploração",
                    "Jogo cooperativo para jogar com amigos online",
                    "Algo desafiador que me faça pensar"
                ]
                
                for example in examples:
                    if st.button(example, key=f"example_{example}"):
                        st.session_state.prompt_example = example
                        st.rerun()
            
            if 'prompt_example' in st.session_state:
                prompt = st.session_state.prompt_example
                del st.session_state.prompt_example
            
            # Gerar recomendações
            if generate_button and prompt and user_profile['user_library']:
                with st.spinner("🤖 Analisando seu perfil e gerando recomendações..."):
                    try:
                        recommendations = models['agent'].recommend(
                            user_profile=user_profile,
                            user_prompt=prompt,
                            user_library=user_profile['user_library'],
                            n_recommendations=n_recommendations
                        )
                        # Exibir métricas
                        display_recommendation_metrics(recommendations)
                        
                        st.markdown("---")
                        st.markdown(f'<h3 class="sub-header">🎪 Top {len(recommendations)} Recomendações</h3>', 
                                unsafe_allow_html=True)
                        
                        # Exibir cada recomendação
                        for i, rec in enumerate(recommendations, 1):
                            with st.expander(f"#{i} - {rec.game_name} (Score: {rec.score:.2f})", 
                                            expanded=(i == 1)):
                                col1, col2 = st.columns([3, 1])
                                
                                with col1:
                                    st.markdown(f"**🎯 Por que recomendamos:**")
                                    st.markdown(f"> {rec.rationale}")
                                    
                                    # Features de match
                                    if rec.match_features:
                                        st.markdown("**🔍 Match Features:**")
                                        features_html = ""
                                        for feature, value in rec.match_features.items():
                                            if isinstance(value, (int, float)):
                                                features_html += f"- `{feature}`: {value:.2f}<br>"
                                        st.markdown(features_html, unsafe_allow_html=True)
                                
                                with col2:
                                    st.markdown("**📊 Detalhes:**")
                                    st.markdown(f"- **Fonte:** {rec.source}")
                                    st.markdown(f"- **ID:** {rec.game_id}")
                                    
                                    # Botão de mais informações
                                    if st.button("📖 Mais info", key=f"more_info_{i}"):
                                        st.session_state[f"show_details_{i}"] = True
                                
                                # Detalhes expandidos
                                if st.session_state.get(f"show_details_{i}", False):
                                    st.markdown("**📈 Análise Detalhada:**")
                                    
                                    # Criar gráfico de score
                                    if 'score_components' in rec.metadata:
                                        scores = rec.metadata['score_components']
                                        fig = go.Figure(data=[
                                            go.Bar(
                                                x=[f"Componente {j+1}" for j in range(len(scores))],
                                                y=scores,
                                                marker_color='lightblue'
                                            )
                                        ])
                                        
                                        fig.update_layout(
                                            title="Decomposição do Score",
                                            height=300
                                        )
                                        
                                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Visualizações
                        st.markdown("---")
                        st.markdown('<h3 class="sub-header">📊 Visualizações</h3>', 
                                unsafe_allow_html=True)
                        visualize_recommendations(recommendations)
                        
                        # Salvar histórico
                        try:
                            models['agent'].save_recommendation_history(
                                user_profile['user_id'],
                                prompt,
                                recommendations,
                                "data/recommendation_history.json"
                            )
                        except:
                            pass
                            
                    except Exception as e:
                        st.error(f"Erro ao gerar recomendações: {str(e)}")
                        st.info("""
                        **Possíveis soluções:**
                        1. Verifique se o pipeline de dados foi executado
                        2. Adicione mais jogos à sua biblioteca
                        3. Tente um prompt diferente
                        """)
        
        with tab2:
            st.markdown('<h2 class="sub-header">📊 Análise do Sistema</h2>', 
                    unsafe_allow_html=True)
            
            if 'games_df' in models:
                games_df = models['games_df']
                
                # Estatísticas básicas
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total de Jogos", len(games_df))
                
                with col2:
                    if 'genres' in games_df.columns:
                        all_genres = []
                        for genres in games_df['genres']:
                            if isinstance(genres, list):
                                all_genres.extend(genres)
                        st.metric("Gêneros Únicos", len(set(all_genres)))
                
                with col3:
                    if 'tags' in games_df.columns:
                        all_tags = []
                        for tags in games_df['tags']:
                            if isinstance(tags, list):
                                all_tags.extend(tags)
                        st.metric("Tags Únicas", len(set(all_tags)))
                
                # Distribuição de gêneros
                st.markdown("### 🎭 Distribuição de Gêneros")
                
                if 'genres' in games_df.columns:
                    genre_counts = {}
                    for genres in games_df['genres']:
                        if isinstance(genres, list):
                            for genre in genres:
                                genre_counts[genre] = genre_counts.get(genre, 0) + 1
                    
                    if genre_counts:
                        genre_df = pd.DataFrame(
                            genre_counts.items(), 
                            columns=['Gênero', 'Count']
                        ).sort_values('Count', ascending=False).head(15)
                        
                        fig = px.bar(
                            genre_df,
                            x='Gênero',
                            y='Count',
                            color='Count',
                            title='Gêneros Mais Comuns',
                            color_continuous_scale='Blues'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown('<h2 class="sub-header">🧠 Como o Sistema Funciona</h2>', 
                    unsafe_allow_html=True)
            
            # Explicação do sistema
            st.markdown("""
            ### 🏗️ Arquitetura do Sistema
            
            O Steam Recommendation Agent combina **três abordagens** de machine learning:
            
            """)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                **🤝 Filtragem Colaborativa**
                - Analisa usuários similares a você
                - Recomenda jogos que usuários parecidos gostam
                - Baseado em padrões de comportamento
                """)
            
            with col2:
                st.markdown("""
                **📝 Baseada em Conteúdo**
                - Analisa descrições, tags e gêneros
                - Usa embeddings semânticos (Sentence-BERT)
                - Encontra jogos com conteúdo similar
                """)
            
            with col3:
                st.markdown("""
                **🎯 Contextual**
                - Interpreta seu prompt natural
                - Extrai intenções e preferências
                - Considera contexto específico
                """)
            
            st.markdown("""
            ### 🔄 Processo de Recomendação
            
            1. **Análise do Perfil**: Seu histórico e preferências são analisados
            2. **Interpretação do Prompt**: Seu pedido é convertido em features
            3. **Busca Multifonte**: Cada abordagem gera candidatos
            4. **Fusão Híbrida**: Os resultados são combinados inteligentemente
            5. **Ranking Final**: Jogos são ordenados por relevância
            6. **Explicação**: Cada recomendação vem com justificativa
            """)
        
        with tab4:
            st.markdown('<h2 class="sub-header">📈 Dashboard de Performance</h2>', 
                    unsafe_allow_html=True)
            # Carregar histórico se existir
            try:
                with open("data/recommendation_history.json", "r") as f:
                    history = json.load(f)
                
                if history:
                    # Converter para DataFrame - CORREÇÃO: Garantir que user_id existe
                    history_data = []
                    for entry in history[-20:]:  # Últimas 20 entradas
                        for rec in entry['recommendations']:
                            history_data.append({
                                'timestamp': entry['timestamp'],
                                'user_id': entry.get('user_id', 'unknown'),  # Usar .get() com valor padrão
                                'prompt': entry['prompt'],
                                'game': rec['game_name'],
                                'score': rec['score']
                            })
                    
                    history_df = pd.DataFrame(history_data)
                    
                    # Métricas gerais - CORREÇÃO: Verificar se colunas existem
                    st.markdown("### 📊 Estatísticas do Histórico")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total de Recomendações", len(history_df))
                    with col2:
                        if 'user_id' in history_df.columns:
                            st.metric("Usuários Únicos", history_df['user_id'].nunique())
                        else:
                            st.metric("Usuários Únicos", 0)
                    with col3:
                        if 'score' in history_df.columns:
                            st.metric("Score Médio", f"{history_df['score'].mean():.2f}")
                        else:
                            st.metric("Score Médio", "N/A")
                    
                    # Gráfico de evolução - CORREÇÃO: Verificar se timestamp existe
                    st.markdown("### 📈 Evolução das Recomendações")
                    
                    if 'timestamp' in history_df.columns:
                        try:
                            history_df['date'] = pd.to_datetime(history_df['timestamp']).dt.date
                            daily_avg = history_df.groupby('date')['score'].mean().reset_index()
                            
                            fig = px.line(
                                daily_avg,
                                x='date',
                                y='score',
                                title='Score Médio Diário',
                                markers=True
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Não foi possível criar gráfico de evolução: {str(e)}")
                    
                    # Jogos mais recomendados
                    st.markdown("### 🏆 Jogos Mais Recomendados")
                    
                    if 'game' in history_df.columns:
                        top_games = history_df['game'].value_counts().head(10).reset_index()
                        top_games.columns = ['Jogo', 'Recomendações']
                        
                        fig = px.bar(
                            top_games,
                            x='Jogo',
                            y='Recomendações',
                            color='Recomendações',
                            color_continuous_scale='Viridis'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Dados insuficientes para análise de jogos mais recomendados.")
                else:
                    st.info("Nenhum histórico de recomendações disponível ainda.")
            except FileNotFoundError:
                st.info("Gere algumas recomendações primeiro para ver o dashboard!")
            except Exception as e:
                st.error(f"Erro ao carregar dashboard: {str(e)}")
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666;">
            <p>🎮 Steam Recommendation Agent v1.0</p>
            <p>Desenvolvido com Streamlit, Transformers e FAISS</p>
            <p>⚠️ Este é um projeto demonstrativo. Dados do Steam usados para fins educacionais.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Se não tem perfil, mostrar formulário de conexão
        user_profile = create_user_profile_form()
        
        # Se conseguiu criar perfil, salvar na sessão e recarregar
        if user_profile:
            st.session_state.user_profile = user_profile
            st.rerun()

if __name__ == "__main__":
    main()