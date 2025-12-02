import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
import random
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Media Storm Analytics", layout="wide")
st.title("🌪️ Media Storm (影视飓风) Data Command Center")

# --- SIDEBAR NAVIGATION ---
st.sidebar.header("Dashboard Navigation")
page = st.sidebar.radio("Go to:", ["1. Fan Analysis", "2. Video Metrics", "3. Team & Brand", "4. Client Matching",
                                   "5. AI Creativity Lab"])


# --- DATA LOADING AND MERGING (Replaced Mock Data with Real CSVs) ---
@st.cache_data
def load_and_process_data():
    data_files = {
        'metadata': "processed_video_metadata.csv",
        'emotional_curve': "emotional_curve_data.csv",
        'personas': "processed_fan_personas.csv"
    }

    loaded_data = {}

    # 1. Load Real Data from Crawler Outputs
    for key, filename in data_files.items():
        if os.path.exists(filename):
            try:
                loaded_data[key] = pd.read_csv(filename)
                st.sidebar.success(f"✅ Loaded {key} data.")
            except pd.errors.EmptyDataError:
                st.sidebar.error(f"⚠️ {filename} is empty.")
                loaded_data[key] = pd.DataFrame()
        else:
            st.sidebar.warning(f"❌ {filename} not found. Using partial mock data.")
            loaded_data[key] = pd.DataFrame()

    df_meta = loaded_data.get('metadata', pd.DataFrame())
    df_curve = loaded_data.get('emotional_curve', pd.DataFrame())
    df_personas = loaded_data.get('personas', pd.DataFrame())

    # --- Data Structure Adjustments ---

    # 1. Fan Platform Comparison (df_fans): Load from manual_fan_data.csv
    fan_data_file = "manual_fan_data.csv"
    if os.path.exists(fan_data_file):
        try:
            # 直接加载您创建的 CSV 文件
            fan_data = pd.read_csv(fan_data_file)
            # 确保 Followers 是整数
            fan_data['Followers'] = fan_data['Followers'].astype(int)
            # 添加一个虚拟的 Engagement_Rate 列，避免后续代码出错（如果需要，请根据实际情况调整）
            fan_data['Engagement_Rate'] = 0.05
            st.sidebar.success("✅ Loaded manual fan data from CSV.")
        except Exception as e:
            st.sidebar.error(f"⚠️ Failed to load {fan_data_file}. Falling back to mock data. Error: {e}")
            # 如果加载失败，回退到模拟数据
            fan_data = pd.DataFrame({
                'Platform': ['Bilibili', 'YouTube', 'Douyin', 'Weibo'],
                'Followers': [14468000, 765000, 9599000, 1764000],
                'Engagement_Rate': [0.10, 0.08, 0.05, 0.02]
            })
    else:
        st.sidebar.warning(f"❌ {fan_data_file} not found. Using the hardcoded data you provided.")
        # 使用您提供的硬编码数据
        fan_data = pd.DataFrame({
            'Platform': ['Bilibili', 'YouTube', 'Douyin', 'Weibo'],
            'Followers': [14468000, 765000, 9599000, 1764000],
            'Engagement_Rate': [0.10, 0.08, 0.05, 0.02] # 仍需一个默认值以兼容后续代码
        })


    # 2. Emotional Curve Data (df_video): Use Real Data
    df_video = df_curve.rename(columns={'Minute': 'Minute', 'Average_Sentiment': 'Sentiment_Score'})

    # 3. AI Creativity Data (df_create): Keep Mocked (Requires internal historical data)
    # FIX: Renamed 'create_data' to 'df_create' to resolve NameError
    df_create = pd.DataFrame({
        'Is_Tech': np.random.randint(0, 2, 100),
        'Has_Face': np.random.randint(0, 2, 100),
        'Budget': np.random.randint(1, 6, 100),
        'Views': np.random.randint(100000, 5000000, 100)
    })

    return fan_data, df_video, df_create, df_personas, df_meta  # Return all necessary DFs


df_fans, df_video, df_create, df_personas, df_meta = load_and_process_data()

# --- DASHBOARD LOGIC ---

# 1. FAN ANALYSIS [Ref: Source 3-6]
if page == "1. Fan Analysis":
    st.header("👥 Fan Profiling & Platform Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("平台粉丝数据对比")
        # 确保 Bili Followers 是显示最新数据
        bili_followers_display = f"{df_fans[df_fans['Platform'] == 'Bilibili']['Followers'].iloc[0]:,}"
        st.metric(label="Bilibili 最新粉丝数", value=bili_followers_display)

        fig_bar = px.bar(df_fans, x='Platform', y='Followers', color='Platform', title="粉丝量对比 (收集最新数据)")
        st.plotly_chart(fig_bar, use_container_width=True)
        # 更改说明文字
        st.caption("数据来源收集的最新全平台粉丝量数据。")

    with col2:
        st.subheader("活跃粉丝画像聚类 (K-Means)")

        if df_personas.empty:
            st.warning("🚨 粉丝画像数据缺失或样本量不足。请检查 `processed_fan_personas.csv`。")
        else:
            # Calculate Cluster Summary
            cluster_summary = df_personas.groupby('Persona_Cluster')[
                ['Level', 'Sign_Length', 'Sex_Encoded']].mean().reset_index()
            cluster_summary['Sample_Size'] = df_personas['Persona_Cluster'].value_counts().sort_index()

            # Scatter Plot for Visualization
            fig_scatter = px.scatter(
                df_personas,
                x='Level',
                y='Sign_Length',
                color=df_personas['Persona_Cluster'].astype(str),
                size='Sign_Length',
                hover_data=['user_id', 'Sex'],
                title=f"粉丝画像聚类图 ({len(df_personas)}个样本)"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

            # Display Cluster Mean Table
            st.markdown("##### 各画像群体特征均值")
            st.dataframe(
                cluster_summary.rename(columns={
                    'Level': '平均等级',
                    'Sign_Length': '平均签名长度',
                    'Sex_Encoded': '平均性别编码 (0/1/2)',
                    'Sample_Size': '样本数'
                }).set_index('Persona_Cluster')
            )
            st.caption("聚类依据：公开等级、签名长度和性别。样本数受 B站反爬限制。")


# 2. VIDEO METRICS [Ref: Source 7-12]
elif page == "2. Video Metrics":
    st.header("🎬 Video Deep Dive & 情绪曲线分析")

    if df_video.empty:
        st.error("❌ 情绪曲线数据缺失。请检查 `emotional_curve_data.csv`。")
    else:
        # Allow selection of video if multiple BVIDs were scraped
        bvid_list = df_video['bvid'].unique().tolist()
        selected_bvid = st.selectbox("选择要分析情绪曲线的视频 (BVID):", bvid_list)

        df_single_curve = df_video[df_video['bvid'] == selected_bvid]

        st.markdown("### 观众情绪曲线 (Derived from Comments)")
        st.write("可视化观众根据弹幕/评论情绪的波动。")

        # Line chart for emotion
        fig_line = px.line(df_single_curve, x='Minute', y='Sentiment_Score', markers=True,
                           title=f"视频 {selected_bvid} 评论情绪趋势 (按时间段)",
                           labels={'Minute': '时间（分钟）', 'Sentiment_Score': '平均情绪得分 (-1: 负面, 1: 正面)'})

        fig_line.add_hrect(y0=0, y1=1.0, line_width=0, fillcolor="green", opacity=0.1, annotation_text="Positive Area")
        fig_line.add_hrect(y0=-1.0, y1=0, line_width=0, fillcolor="red", opacity=0.1, annotation_text="Negative Area")
        fig_line.add_hline(y=0, line_dash="dash", line_color="gray")

        st.plotly_chart(fig_line, use_container_width=True)
        st.caption("情绪得分基于 SnowNLP 计算，数值代表相对积极/消极程度。")

        st.markdown("### 视频核心指标排名")
        if not df_meta.empty:
            df_meta['Interactions'] = df_meta['likes'] + df_meta['coins'] + df_meta['favorites'] + df_meta['shares']
            df_meta['Score'] = (df_meta['Interactions'] / df_meta['views']) * 1000 if df_meta['views'].sum() > 0 else 0

            st.dataframe(
                df_meta[['title', 'views', 'Interactions', 'Score', 'comment_count']]
                .sort_values('Score', ascending=False)
                .head(10)
                .rename(columns={'title': '标题', 'views': '播放量', 'Interactions': '互动总数', 'Score': '互动得分',
                                 'comment_count': '评论数'})
                .set_index('标题')
            )
        else:
            st.warning("⚠️ 视频元数据缺失，无法显示指标排名。")


# 3. TEAM & BRAND [Ref: Source 13-19]
elif page == "3. Team & Brand":
    st.header("🏢 Team Efficiency & Matrix Management")

    # Metric Cards (Retained Mock for demonstration)
    c1, c2, c3 = st.columns(3)
    c1.metric(label="团队效率得分", value="8.4/10", delta="1.2")
    c2.metric(label="品牌垂直度", value="高", delta="科技摄影领域")
    c3.metric(label="季度产出", value=f"{len(df_meta)} 个视频" if not df_meta.empty else "24 视频 (模拟)", delta="-2")

    st.caption("Source [14, 16]: 量化团队贡献和品牌影响力 (需要内部数据支持)。")


# 4. CLIENT MATCHING [Ref: Source 20-22]
elif page == "4. Client Matching":
    st.header("🤝 Commercial & Client Analytics")

    # Simulating a Client Match System (Retained Mock for demonstration)
    client_name = st.text_input("输入甲方/品牌名称:", "Example Camera Brand")
    client_tags = st.multiselect("甲方需求标签:", ["科技评测", "情感故事", "Vlog", "高预算"],
                                 default=["科技评测"])

    if st.button("计算匹配度得分"):
        score = np.random.randint(60, 99)
        st.progress(score)
        st.success(f"与 {client_name} 的匹配得分: {score}%")
        st.info("基于历史合作客户受众与现有粉丝群体的重叠度 [Source 21]。")

# 5. AI CREATIVITY LAB [Ref: Source 23-28]
elif page == "5. AI Creativity Lab":
    st.header("🧠 AI 流量预测 & 创意库")

    st.markdown("""
    **目标:** 输入创意参数，在拍摄前预测视频表现。
    *[Source 24]: 基于机器学习的流量预测模型 (模拟)*
    """)

    # Machine Learning Model (Simple Random Forest) - Retained Mock
    X = df_create[['Is_Tech', 'Has_Face', 'Budget']]
    y = df_create['Views']

    try:
        model = RandomForestRegressor(random_state=42)
        model.fit(X, y)
    except Exception as e:
        st.error(f"无法训练模拟的 AI 模型: {e}")
        model = None

    # User Input for Prediction
    col1, col2, col3 = st.columns(3)
    with col1:
        is_tech = st.selectbox("是否是科技视频?", [0, 1])
    with col2:
        has_face = st.selectbox("封面是否有人脸?", [0, 1])
    with col3:
        budget = st.slider("预算等级 (1-5)", 1, 5, 3)

    if st.button("🔮 预测流量"):
        if model:
            # Model expects a 2D array input
            prediction = model.predict([[is_tech, has_face, budget]])
            st.metric(label="预测播放量", value=f"{int(prediction[0]):,}")

            if prediction[0] > 3000000:
                st.balloons()
                st.success("推荐: **必须发布!** 高潜力。")
            else:
                st.warning("推荐: 优化封面或标题。")
        else:
            st.error("AI 模型未成功加载。")

    st.divider()
    st.subheader("创意库检索")
    st.text_input("搜索创意数据库 (例如: '慢动作火焰'):")
    st.caption("Source [27]: 检索系统，用于将创意与甲方需求进行匹配。")