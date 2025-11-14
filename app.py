import streamlit as st
import pandas as pd
import time

# 关键：从你上一步创建的 model.py 文件中导入那个主类
# 确保 model.py 和 app.py 在同一个文件夹里
try:
    from model import MovieRecommendationSystem
except ImportError:
    st.error("错误：找不到 'model.py'。请确保你已经把 x1.ipynb 里的代码复制并保存为 'model.py'。")
    st.stop()


# ---------------------------------------------------------------------
# 1. 配置你的 .dat 数据集文件夹路径
# ---------------------------------------------------------------------
# !!! 必须修改这里 !!!
# 把这里的路径改成你存放 users.dat, movies.dat, ratings.dat 的文件夹路径
# 提示：使用正斜杠 /，即使在Windows上也一样
DATA_PATH = "/Users/gloria/Desktop/7008 project/movie_dataset/"  # <-- ！！修改这里！！


# ---------------------------------------------------------------------
# 2. 职业ID和名称的映射
# (我从 zsh/lwy 的 preprocessing.ipynb 文件中帮你复制过来了)
# ---------------------------------------------------------------------
OCCUPATION_MAP = {
    0: "other", 1: "academic/educator", 2: "artist", 3: "clerk",
    4: "college/grad student", 5: "customer service", 6: "doctor/health care",
    7: "executive/managerial", 8: "farmer", 9: "homemaker", 10: "K-12 student",
    11: "lawyer", 12: "programmer", 13: "retired", 14: "sales/marketing",
    15: "scientist", 16: "self-employed", 17: "technician/engineer",
    18: "tradesman/craftsman", 19: "unemployed", 20: "writer"
}

# ---------------------------------------------------------------------
# 3. 年龄分组
# (这是 x1.ipynb 训练时使用的年龄)
# ---------------------------------------------------------------------
AGE_MAP = {
    1: "Under 18", 18: "18-24", 25: "25-34",
    35: "35-44", 45: "45-49", 50: "50-55", 56: "56+"
}


# ---------------------------------------------------------------------
# 4. 加载和缓存模型
# (这是最关键的函数，它只会运行一次)
# ---------------------------------------------------------------------
@st.cache_resource  # 使用 Streamlit 缓存，避免每次刷新都重新训练
def load_recommendation_system(data_path):
    """
    加载并运行完整的机器学习流水线。
    这会非常慢（可能需要几分钟），但只会运行一次。
    """
    with st.spinner("正在初始化推荐系统，这可能需要几分钟..."):
        try:
            # 1. 初始化系统
            system = MovieRecommendationSystem(data_path)
            
            # 2. 运行完整的训练流水线
            # (这会加载数据、构建画像、训练模型等)
            system.run_complete_pipeline()
            
            return system
        except Exception as e:
            st.error(f"加载模型失败: {e}")
            st.error(f"请检查 DATA_PATH 变量是否设置正确，并且 '{data_path}' 路径下有 .dat 文件。")
            return None

# 启动模型加载
system = load_recommendation_system(DATA_PATH)

if system is None:
    st.stop()

# ---------------------------------------------------------------------
# 5. 构建 Streamlit 网站界面
# ---------------------------------------------------------------------
st.title("🎬 电影推荐系统（新用户冷启动）")
st.write("这是为新用户设计的推荐系统。请选择新用户的基本画像信息：")

# --- 创建输入组件 ---
col1, col2, col3 = st.columns(3)

with col1:
    selected_age_label = st.selectbox(
        "选择年龄段:",
        list(AGE_MAP.values())
    )
    # 反向查找年龄ID
    selected_age_id = [k for k, v in AGE_MAP.items() if v == selected_age_label][0]


with col2:
    selected_gender = st.selectbox(
        "选择性别:",
        ["M", "F"]
    )

with col3:
    selected_occ_label = st.selectbox(
        "选择职业:",
        list(OCCUPATION_MAP.values())
    )
    # 反向查找职业ID
    selected_occ_id = [k for k, v in OCCUPATION_MAP.items() if v == selected_occ_label][0]


# --- 推荐按钮和逻辑 ---
if st.button("🚀 开始推荐", type="primary"):
    
    # 1. 准备新用户信息字典
    new_user_info = {
        'gender': selected_gender,
        'age': selected_age_id,
        'occupation': selected_occ_id,
    }

    st.subheader(f"为 {selected_age_label}, {selected_gender}, {selected_occ_label} 画像推荐：")

    # 2. 调用模型的核心推荐功能
    with st.spinner("正在计算推荐结果..."):
        try:
            # recommend_for_new_user 返回 [(movie_idx, pred_rating), ...]
            recommendations = system.recommend_for_new_user(new_user_info)
            
            if not recommendations:
                st.warning("未找到匹配的推荐。")
            
            # 3. 显示结果
            for i, (movie_idx, pred_rating) in enumerate(recommendations, 1):
                # 从 system.movies_df 中查找电影信息
                # .iloc[movie_idx] 是因为模型返回的是0-based索引
                movie_info = system.movies_df.iloc[movie_idx]
                title = movie_info['title']
                genres = movie_info['genres']
                
                st.markdown(f"**{i}. 《{title}》**")
                st.write(f"&nbsp;&nbsp;&nbsp;&nbsp; **类型**: {genres} | **预测评分**: {pred_rating:.2f}")

        except Exception as e:
            st.error(f"推荐时出错: {e}")