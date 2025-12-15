import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import json
import pickle
import faiss

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
    .library-recommendation {
        background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
    }
    .dataset-recommendation {
        background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%);
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
    .hours-badge {
        background-color: #FF9800;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 10px;
        font-size: 0.8rem;
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
    
    # Número de recomendações
    n_recommendations = st.slider(
        "Número de Recomendações",
        min_value=3,
        max_value=7,
        value=5
    )
    
    # Configurações da biblioteca
    with st.expander("📚 Configurações da Biblioteca"):
        max_library_hours = st.slider(
            "Máximo de horas para considerar 'não jogado'",
            min_value=0.5,
            max_value=10.0,
            value=3.0,
            step=0.5,
            help="Jogos com menos horas que esta serão considerados da biblioteca"
        )
        
        prioritize_library = st.checkbox(
            "Priorizar jogos da biblioteca",
            value=True,
            help="Buscar primeiro na sua biblioteca antes de recomendar novos jogos"
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
    st.markdown("Desenvolvido usando:")
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
        interaction_df = pd.read_pickle("data/processed/interactions_processed.pkl")
        
        # Carregar embeddings
        from src.game_embedder import GameEmbedder
        embedder = GameEmbedder()
        embedder.load_all("data/embeddings")
        
        # Carregar modelo de perfil
        from src.profile_analyzer_deep import DeepProfileAnalyzer
        analyzer = DeepProfileAnalyzer()
        analyzer.load_model("models/deep_profile_model.pkl")
        
        # Carregar interpretador de prompt
        from src.prompt_interpreter import PromptInterpreter
        interpreter = PromptInterpreter()
        
        # Carregar API da Steam
        from src.steam_api_client import SteamAPI
        steam_api = SteamAPI()
        
        # Carregar AGENTE
        from src.agent import RecommendationAgent
        
        agent = RecommendationAgent(
            profile_analyzer=analyzer,
            prompt_interpreter=interpreter,
            embedder=embedder,
            games_df=games_df,
            interactions_df=interaction_df,
            steam_api=steam_api
        )
        
        return {
            'games_df': games_df,
            'embedder': embedder,
            'analyzer': analyzer,
            'embedder': embedder,
            'interpreter': interpreter,
            'steam_api': steam_api,
            'agent': agent
        }
    except Exception as e:
        st.error(f"Erro ao carregar modelos: {e}")
        

def create_user_profile_form():
    st.markdown('<h3 class="sub-header">🔗 Conectar à Sua Conta Steam</h3>', 
                unsafe_allow_html=True)
    
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            steam_input = st.text_input(
                "Digite seu SteamID, Vanity URL ou URL do perfil:",
                value=st.session_state.get('steam_input', ''),
                placeholder="Ex: 76561197960287930 ou https://steamcommunity.com/id/seunome",
                help="Você pode encontrar seu SteamID em https://steamid.io/",
                key="steam_input_field"
            )
            
            if steam_input != st.session_state.get('steam_input', ''):
                st.session_state.steam_input = steam_input
            
            st.caption("💡 Exemplos: `76561197960287930` ou `https://steamcommunity.com/id/gabeloganneweller`")
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            connect_button = st.button(
                "🎮 Conectar à Minha Steam",
                type="primary",
                use_container_width=True,
                disabled=not steam_input,
                key="connect_button"
            )
    
    if connect_button and steam_input:
        st.info(f"🔍 Tentando conectar com: {steam_input}")
        result = process_steam_connection(steam_input)
        
        if result:
            st.success("✅ Perfil obtido com sucesso!")
        else:
            st.error("❌ Falha ao obter perfil")
            
        return result

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
    try:
        from src.steam_api_client import SteamAPI, analyze_gaming_profile
        
        status_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        # Etapa 1: Inicializar API
        status_placeholder.markdown("**🔐 Conectando ao servidor Steam...**")
        progress_bar.progress(10)
        
        steam_api = SteamAPI()
        
        # Etapa 2: Processar input do usuário
        status_placeholder.markdown("**🔍 Identificando seu perfil...**")
        progress_bar.progress(30)
        
        steam_id = steam_api.extract_steam_id(steam_input)
        
        if not steam_id:
            st.error("❌ Não foi possível identificar sua conta Steam. Verifique o formato.")

        st.session_state.connected = True
        st.session_state.steam_id = steam_id
        
        # Etapa 3: Buscar dados do perfil
        status_placeholder.markdown("**📡 Buscando suas informações...**")
        progress_bar.progress(50)
        
        player_data = steam_api.get_player_summary(steam_id)
        
        if not player_data:
            st.error("❌ Perfil não encontrado ou privado.")
        
        # Etapa 4: Buscar biblioteca de jogos
        status_placeholder.markdown("**🎮 Analisando sua biblioteca...**")
        progress_bar.progress(70)
        
        games_data = steam_api.get_owned_games(steam_id)
        
        if not games_data or games_data['game_count'] == 0:
            st.warning("⚠️ Sua biblioteca está vazia ou privada.")
        
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
        
        # Preparar dados da biblioteca com informações completas
        library_with_details = []
        playtimes_dict = {}
        
        for game in games_data['games']:
            playtime_hours = game.get('playtime_forever', 0) / 60
            library_with_details.append({
                'appid': game['appid'],
                'name': game.get('name', 'Desconhecido'),
                'playtime_minutes': game.get('playtime_forever', 0),
                'playtime_hours': playtime_hours
            })
            
            if playtime_hours > 0:
                playtimes_dict[game['appid']] = playtime_hours
        
        # Contar jogos com menos de 3 horas
        games_under_3h = sum(1 for game in library_with_details 
                           if game['playtime_hours'] < 3)
        
        # Preparar perfil completo
        user_profile = {
            'user_id': steam_id,
            'persona_name': player_data['personaname'],
            'profile_url': player_data['profileurl'],
            'avatar': player_data.get('avatar', ''),
            'playstyle': profile_analysis['playstyle'],
            'avg_playtime': games_data['total_playtime_hours'] / max(games_data['game_count'], 1),
            'favorite_genre': profile_analysis['preferred_genres'],
            'user_library': [game['appid'] for game in games_data['games']],
            'library_details': library_with_details,  # Informações detalhadas
            'playtimes': playtimes_dict,
            'total_hours': games_data['total_playtime_hours'],
            'game_count': games_data['game_count'],
            'games_under_3h': games_under_3h,
            'profile_data': player_data
        }

        # Mostrar resumo do perfil
        show_profile_summary(user_profile, games_data, profile_analysis)

        # Limpar estado de submissão
        st.session_state.form_submitted = False

        return user_profile

    except ValueError as e:
        st.error(f"❌ {str(e)}")
        st.session_state.form_submitted = False
        st.info("""
        **Para o desenvolvedor:** Configure sua Steam Web API Key:
        1. Acesse https://steamcommunity.com/dev/apikey
        2. Crie uma chave
        3. Adicione no arquivo `config/steam_config.py`
        """)
    
    except Exception as e:
        st.error(f"❌ Erro na conexão: {str(e)}")
        st.session_state.form_submitted = False
        st.info("Verifique sua conexão ou tente novamente em alguns instantes.")

def show_profile_summary(user_profile, games_data, profile_analysis):
    """Mostra resumo do perfil do usuário"""
    with st.expander(f"👤 Seu Perfil Steam - {user_profile['persona_name']}", expanded=True):
        # Métricas
        col1, col2, col3, col4 = st.columns(4)
        
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


def display_recommendation_metrics(recommendations):
    """Exibe métricas das recomendações"""
    if not recommendations:
        return
    
    # Garantir que scores são numéricos
    scores = []
    library_count = 0
    dataset_count = 0
    
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
        
        if rec.source == "user_library":
            library_count += 1
        else:
            dataset_count += 1
    
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
                   f'<h3>📚</h3>'
                   f'<h4>{library_count}</h4>'
                   f'<p>Da sua biblioteca</p>'
                   f'</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'<div class="metric-card">'
                   f'<h3>🛒</h3>'
                   f'<h4>{dataset_count}</h4>'
                   f'<p>Novos jogos</p>'
                   f'</div>', unsafe_allow_html=True)
    
    with col4:
        total_recs = len(recommendations)
        st.markdown(f'<div class="metric-card">'
                   f'<h3>🎯</h3>'
                   f'<h4>{total_recs}</h4>'
                   f'<p>Total</p>'
                   f'</div>', unsafe_allow_html=True)

def visualize_recommendations(recommendations):
    """Cria visualizações para as recomendações"""
    if not recommendations:
        return
    
    # Criar DataFrame para visualização
    data = []
    for rec in recommendations:
        playtime = None
        if rec.source == "user_library" and 'playtime_hours' in rec.metadata:
            playtime = rec.metadata['playtime_hours']
        
        data.append({
            'name': rec.game_name,
            'score': rec.score,
            'source': rec.source,
            'playtime_hours': playtime,
            'game_id': rec.game_id
        })
    
    df = pd.DataFrame(data)
    
    # Gráfico de barras por fonte
    source_counts = df['source'].value_counts().reset_index()
    source_counts.columns = ['Fonte', 'Quantidade']
    
    fig_source = px.pie(
        source_counts,
        values='Quantidade',
        names='Fonte',
        title='📊 Distribuição por Fonte',
        color='Fonte',
        color_discrete_map={
            'user_library': '#4CAF50',
            'dataset_fallback': '#2196F3',
            'popular_fallback': '#FF9800'
        }
    )
    
    fig_source.update_layout(height=400)
    
    # Gráfico de barras de scores
    fig_bar = px.bar(
        df,
        x='name',
        y='score',
        color='source',
        title='📈 Scores das Recomendações',
        labels={'name': 'Jogo', 'score': 'Score', 'source': 'Fonte'},
        color_discrete_map={
            'user_library': '#4CAF50',
            'dataset_fallback': '#2196F3',
            'popular_fallback': '#FF9800'
        }
    )
    
    fig_bar.update_layout(
        xaxis_tickangle=-45,
        height=400,
        showlegend=True
    )
    
    # Exibir gráficos
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_source, use_container_width=True)
    with col2:
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
                
                # Verificar se estamos usando agente antigo
                if models.get('using_old_agent', False):
                    st.warning("⚠️ Usando agente antigo. Execute o pipeline de dados completo para habilitar a busca na biblioteca.")
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
            
            # Mostrar informações do usuário
            with st.expander("👤 Seu Perfil Steam", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🎮 Jogos", user_profile.get('game_count', 0))
                with col2:
                    if 'total_hours' in user_profile:
                        st.metric("⏱️ Horas", f"{user_profile['total_hours']:,}")
                with col3:
                    st.metric("🎯 Estilo", user_profile.get('playstyle', 'Moderado'))

                # Gêneros favoritos
                if user_profile['favorite_genre']:
                    st.markdown("**🌟 Seus Gêneros Preferidos:**")
                    tags = " ".join([f"`{genre}`" for genre in user_profile['favorite_genre'][:5]])
                    st.markdown(tags)
                
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
                    value="Quero um jogo relaxante singleplayer e com boa história",
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
            
            # Gerar recomendações
            if generate_button and prompt and user_profile['user_library']:
                with st.spinner("🤖 Analisando seu perfil e gerando recomendações..."):
                    try:
                        # Verificar se estamos usando o novo agente
                        if hasattr(models['agent'], 'recommend_from_prompt'):
                            recommendations = models['agent'].recommend_from_prompt(
                                user_id=user_profile['user_id'],
                                user_prompt=prompt,
                                n_recommendations=n_recommendations
                            )

                        # Exibir métricas
                        display_recommendation_metrics(recommendations)
                        
                        st.markdown("---")
                        st.markdown(f'<h3 class="sub-header">🎪 Top {len(recommendations)} Recomendações</h3>', 
                                unsafe_allow_html=True)
                        
                        # Exibir cada recomendação
                        for i, rec in enumerate(recommendations, 1):
                            # Determinar cor baseada na fonte
                            if rec.source == "user_library":
                                card_class = "library-recommendation"
                                source_badge = "📚 Da sua biblioteca"
                            elif rec.source == "dataset_fallback":
                                card_class = "dataset-recommendation"
                                source_badge = "🛒 Novo jogo"
                            else:
                                card_class = "recommendation-card"
                                source_badge = f"🎯 {rec.source}"
                            
                            # Criar card expansível
                            with st.expander(f"#{i} - {rec.game_name} (Score: {rec.score:.2f})", 
                                            expanded=(i == 1)):
                                
                                # Mostrar badge de fonte
                                col1, col2 = st.columns([3, 1])
                                
                                with col1:
                                    st.markdown(f"**{source_badge}**")
                                    
                                    # Mostrar horas jogadas se for da biblioteca
                                    if rec.source == "user_library" and 'playtime_hours' in rec.metadata:
                                        hours = rec.metadata['playtime_hours']
                                        if hours == 0:
                                            st.markdown("⏳ **Você ainda não jogou este jogo!**")
                                        else:
                                            st.markdown(f"⏱️ **Você jogou apenas {hours:.1f} horas**")
                                    
                                    st.markdown(f"**🎯 Por que recomendamos:**")
                                    st.markdown(f"> {rec.rationale}")
                                    
                                    # Features de match
                                    if rec.metadata:
                                        st.markdown("**🔍 Match Features:**")
                                        features_html = ""
                                        for feature, value in rec.metadata.items():
                                            if isinstance(value, (int, float)):
                                                features_html += f"- `{feature}`: {value:.2f}<br>"
                                        st.markdown(features_html, unsafe_allow_html=True)
                                
                                with col2:
                                    st.markdown("**📊 Detalhes:**")
                                    st.markdown(f"- **Fonte:** {rec.source}")
                                    st.markdown(f"- **ID:** {rec.game_id}")
                                    st.markdown(f"- **Score:** {rec.score:.2f}")
                                    
                                    # Botão de mais informações
                                    if st.button("📖 Mais info", key=f"more_info_{i}"):
                                        st.session_state[f"show_details_{i}"] = True
                                
                                # Detalhes expandidos
                                if st.session_state.get(f"show_details_{i}", False):
                                    st.markdown("**📈 Análise Detalhada:**")
                                    
                                    # Criar gráfico de score
                                    if 'metadata' in rec.__dict__:
                                        features = rec.metadata
                                        # Filtrar features numéricas
                                        numeric_features = {k: v for k, v in features.items() 
                                                          if isinstance(v, (int, float))}
                                        
                                        if numeric_features:
                                            fig = go.Figure(data=[
                                                go.Bar(
                                                    x=list(numeric_features.keys()),
                                                    y=list(numeric_features.values()),
                                                    marker_color='lightblue'
                                                )
                                            ])
                                            
                                            fig.update_layout(
                                                title="Decomposição do Score",
                                                height=300,
                                                xaxis_tickangle=-45
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
                **📚 Busca na Biblioteca**
                - Analisa os jogos que você já possui
                - Foca em jogos com poucas horas jogadas para as recomendações
                - Recomenda títulos que você pode ter esquecido
                """)
            
            with col2:
                st.markdown("""
                **🤝 Filtragem Colaborativa**
                - Analisa usuários similares a você
                - Recomenda jogos que usuários parecidos gostam
                - Baseado em padrões de comportamento
                """)
            
            with col3:
                st.markdown("""
                **📝 Baseada em Conteúdo**
                - Analisa descrições, tags e gêneros
                - Usa embeddings semânticos (Sentence-BERT)
                - Encontra jogos com conteúdo similar
                """)
            
            st.markdown("""
            ### 🔄 Processo de Recomendação
            
            1. **Análise do Perfil**: Seu histórico e preferências são analisados
            2. **Busca na Biblioteca**: Procura jogos que você já tem mas jogou pouco
            3. **Interpretação do Prompt**: Seu pedido é convertido em features
            4. **Busca Multifonte**: Cada abordagem gera candidatos
            5. **Fusão Híbrida**: Os resultados são combinados inteligentemente
            6. **Ranking Final**: Jogos são ordenados por relevância
            7. **Explicação**: Cada recomendação vem com justificativa
            """)
        
        with tab4:
            st.markdown('<h2 class="sub-header">📈 Dashboard de Performance</h2>', 
                    unsafe_allow_html=True)
            # Carregar histórico se existir
            try:
                with open("data/recommendation_history.json", "r") as f:
                    history = json.load(f)
                
                if history:
                    # Converter para DataFrame
                    history_data = []
                    for entry in history[-20:]:
                        for rec in entry['recommendations']:
                            history_data.append({
                                'timestamp': entry['timestamp'],
                                'user_id': entry.get('user_id', 'unknown'),
                                'prompt': entry['prompt'],
                                'game': rec['game_name'],
                                'score': rec['score'],
                                'source': rec.get('source', 'unknown')
                            })
                    
                    history_df = pd.DataFrame(history_data)
                    
                    # Métricas gerais
                    st.markdown("### 📊 Estatísticas do Histórico")
                    
                    col1, col2, col3, col4 = st.columns(4)
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
                    with col4:
                        if 'source' in history_df.columns:
                            library_pct = (history_df['source'] == 'user_library').mean() * 100
                            st.metric("Da Biblioteca", f"{library_pct:.1f}%")
                        else:
                            st.metric("Da Biblioteca", "N/A")
                    
                    # Gráfico de evolução
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
            <p>🎮 Steam Recommendation Agent v2.0</p>
            <p>Novo: Busca inteligente na sua biblioteca!</p>
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