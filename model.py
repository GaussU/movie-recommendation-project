import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.font_manager as fm
import platform
import warnings
warnings.filterwarnings('ignore')

# 首先定义字体设置函数
def setup_chinese_font():
    """设置中文字体支持"""
    # 根据操作系统设置字体
    system = platform.system()
    
    if system == 'Windows':
        # Windows系统
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'SimSun']
    elif system == 'Darwin':
        # macOS系统
        chinese_fonts = ['Heiti SC', 'STHeiti', 'AppleGothic', 'PingFang SC']
    else:
        # Linux系统
        chinese_fonts = ['WenQuanYi Micro Hei', 'DejaVu Sans', 'Noto Sans CJK SC']
    
    # 查找可用的中文字体
    available_fonts = []
    for font_name in chinese_fonts:
        try:
            font_path = fm.findfont(fm.FontProperties(family=font_name))
            if font_path:
                available_fonts.append(font_name)
                print(f"找到字体: {font_name} -> {font_path}")
        except:
            continue
    
    if available_fonts:
        # 设置字体
        plt.rcParams['font.sans-serif'] = available_fonts + ['DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
        print(f"已设置中文字体: {available_fonts[0]}")
        return True
    else:
        print("警告: 未找到中文字体，将使用英文显示")
        return False

# 现在定义所有类
class UserProfileBuilder:
    """用户画像构建器"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)
        self.kmeans = None
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        # 设置中文字体
        self.chinese_supported = setup_chinese_font()
    
    def build_user_features(self, users_df, ratings_df, movies_df):
        """构建用户特征矩阵"""
        print("=== 开始构建用户特征 ===")
        
        # 基本特征
        print("步骤1: 处理用户基本特征...")
        users_df = users_df.copy()
        users_df['gender_encoded'] = users_df['gender'].map({'M': 0, 'F': 1})
        
        # 用户行为特征
        print("步骤2: 构建用户行为特征...")
        user_behavior = self._build_user_behavior_features(ratings_df, movies_df)
        
        # 合并特征
        print("步骤3: 合并所有特征...")
        user_features = pd.merge(users_df, user_behavior, on='user_id', how='left')
        
        # 选择数值型特征 - 修正列名问题
        feature_columns = ['gender_encoded', 'age', 'occupation', 'rating_count', 
                          'avg_rating', 'rating_std', 'total_rating_hours']
        
        # 添加类型偏好特征
        genre_columns = [col for col in user_features.columns if col.startswith('genre_')]
        feature_columns.extend(genre_columns)
        
        # 检查并处理缺失的列
        available_columns = [col for col in feature_columns if col in user_features.columns]
        missing_columns = set(feature_columns) - set(available_columns)
        
        if missing_columns:
            print(f"警告: 以下列不存在，将被跳过: {missing_columns}")
        
        # 填充缺失值
        user_features[available_columns] = user_features[available_columns].fillna(0)
        
        print(f"用户特征维度: {user_features[available_columns].shape}")
        
        return user_features[available_columns], user_features
    
    def _build_user_behavior_features(self, ratings_df, movies_df):
        """构建用户行为特征"""
        # 基本评分统计
        user_stats = ratings_df.groupby('user_id').agg({
            'rating': ['count', 'mean', 'std'],
            'timestamp': ['min', 'max']
        }).reset_index()
        
        user_stats.columns = ['user_id', 'rating_count', 'avg_rating', 'rating_std', 
                             'first_rating_time', 'last_rating_time']
        
        # 计算评分时长（小时）
        user_stats['total_rating_hours'] = (
            user_stats['last_rating_time'] - user_stats['first_rating_time']
        ) / 3600
        
        # 用户类型偏好 - 添加错误处理
        try:
            ratings_with_genres = pd.merge(ratings_df, movies_df[['movie_id', 'genres']], 
                                         on='movie_id', how='left')
            
            # 展开类型
            ratings_with_genres['genres_list'] = ratings_with_genres['genres'].str.split('|')
            exploded_ratings = ratings_with_genres.explode('genres_list')
            
            # 计算用户对每种类型的平均评分
            genre_preferences = exploded_ratings.groupby(['user_id', 'genres_list'])['rating'].mean().unstack(fill_value=0)
            genre_preferences.columns = [f'genre_{col}' for col in genre_preferences.columns]
            
            # 合并所有行为特征
            user_behavior = pd.merge(user_stats, genre_preferences, on='user_id', how='left')
        except Exception as e:
            print(f"类型偏好特征构建失败，使用基础特征: {e}")
            user_behavior = user_stats
        
        return user_behavior
    
    def reduce_dimensionality(self, features):
        """特征降维"""
        print("步骤4: 进行特征降维...")
        
        # 确保特征是数值型
        features = features.astype(float)
        
        # 标准化特征
        features_scaled = self.scaler.fit_transform(features)
        
        # PCA降维
        features_reduced = self.pca.fit_transform(features_scaled)
        
        print(f"降维后特征维度: {features_reduced.shape}")
        print(f"保留的方差比例: {np.sum(self.pca.explained_variance_ratio_):.4f}")
        
        return features_reduced
    
    def build_user_profiles(self, features_reduced, n_clusters=5):
        """构建用户画像聚类"""
        print("步骤5: 构建用户画像聚类...")
        
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        user_clusters = self.kmeans.fit_predict(features_reduced)
        
        self.is_fitted = True
        return user_clusters
    
    def visualize_user_profiles(self, features_reduced, user_clusters, user_features):
        """可视化用户画像"""
        print("步骤6: 生成用户画像可视化...")
        
        # 根据字体支持情况选择标签语言
        if self.chinese_supported:
            titles = {
                'cluster_visual': '用户聚类可视化 (前两个主成分)',
                'cluster_dist': '用户聚类分布',
                'age_dist': '各聚类平均年龄',
                'rating_dist': '各聚类平均评分',
                'xlabel_pc1': '主成分 1',
                'ylabel_pc2': '主成分 2',
                'xlabel_cluster': '聚类标签',
                'ylabel_count': '用户数量',
                'ylabel_age': '平均年龄',
                'ylabel_rating': '平均评分',
                'no_rating_data': '评分数据不可用'
            }
        else:
            titles = {
                'cluster_visual': 'User Clustering Visualization (First Two Principal Components)',
                'cluster_dist': 'User Cluster Distribution',
                'age_dist': 'Average Age by Cluster',
                'rating_dist': 'Average Rating by Cluster',
                'xlabel_pc1': 'Principal Component 1',
                'ylabel_pc2': 'Principal Component 2',
                'xlabel_cluster': 'Cluster Label',
                'ylabel_count': 'Number of Users',
                'ylabel_age': 'Average Age',
                'ylabel_rating': 'Average Rating',
                'no_rating_data': 'Rating Data Not Available'
            }
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 聚类结果可视化
        if features_reduced.shape[1] >= 2:
            scatter = axes[0,0].scatter(features_reduced[:, 0], features_reduced[:, 1], 
                                      c=user_clusters, cmap='viridis', alpha=0.6)
            axes[0,0].set_title(titles['cluster_visual'])
            axes[0,0].set_xlabel(titles['xlabel_pc1'])
            axes[0,0].set_ylabel(titles['ylabel_pc2'])
            plt.colorbar(scatter, ax=axes[0,0])
        
        # 2. 聚类分布
        cluster_counts = pd.Series(user_clusters).value_counts().sort_index()
        axes[0,1].bar(cluster_counts.index, cluster_counts.values)
        axes[0,1].set_title(titles['cluster_dist'])
        axes[0,1].set_xlabel(titles['xlabel_cluster'])
        axes[0,1].set_ylabel(titles['ylabel_count'])
        
        # 3. 年龄分布
        user_features['cluster'] = user_clusters
        age_cluster_stats = user_features.groupby('cluster')['age'].mean()
        axes[1,0].bar(age_cluster_stats.index, age_cluster_stats.values)
        axes[1,0].set_title(titles['age_dist'])
        axes[1,0].set_xlabel(titles['xlabel_cluster'])
        axes[1,0].set_ylabel(titles['ylabel_age'])
        
        # 4. 评分行为
        if 'avg_rating' in user_features.columns:
            rating_cluster_stats = user_features.groupby('cluster')['avg_rating'].mean()
            axes[1,1].bar(rating_cluster_stats.index, rating_cluster_stats.values)
            axes[1,1].set_title(titles['rating_dist'])
            axes[1,1].set_xlabel(titles['xlabel_cluster'])
            axes[1,1].set_ylabel(titles['ylabel_rating'])
        else:
            axes[1,1].text(0.5, 0.5, titles['no_rating_data'], 
                          ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1,1].set_title(titles['rating_dist'])
        
        plt.tight_layout()
        plt.savefig('user_profiles_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig

# 其他类保持不变，但确保它们也正确使用了字体设置
class MovieProfileBuilder:
    """电影画像构建器"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)
        self.kmeans = None
        
    def build_movie_features(self, movies_df, ratings_df):
        """构建电影特征矩阵"""
        print("=== 开始构建电影特征 ===")
        
        # 基本特征
        print("步骤1: 处理电影基本特征...")
        movies_df = movies_df.copy()
        
        # 类型特征
        print("步骤2: 构建电影类型特征...")
        genre_features = self._build_genre_features(movies_df)
        
        # 评分行为特征
        print("步骤3: 构建电影评分特征...")
        rating_features = self._build_rating_features(ratings_df)
        
        # 合并特征
        print("步骤4: 合并所有特征...")
        movie_features = pd.merge(movies_df[['movie_id', 'release_year']], 
                                genre_features, on='movie_id', how='left')
        movie_features = pd.merge(movie_features, rating_features, on='movie_id', how='left')
        
        # 选择数值型特征
        feature_columns = ['release_year'] + \
                         [col for col in movie_features.columns if col.startswith('genre_')] + \
                         ['avg_rating', 'rating_count', 'rating_std']
        
        # 检查并处理缺失的列
        available_columns = [col for col in feature_columns if col in movie_features.columns]
        missing_columns = set(feature_columns) - set(available_columns)
        
        if missing_columns:
            print(f"警告: 以下列不存在，将被跳过: {missing_columns}")
        
        # 填充缺失值
        movie_features[available_columns] = movie_features[available_columns].fillna(0)
        
        print(f"电影特征维度: {movie_features[available_columns].shape}")
        return movie_features[available_columns], movie_features
    
    def _build_genre_features(self, movies_df):
        """构建电影类型特征"""
        try:
            # 展开类型
            movies_df = movies_df.copy()
            movies_df['genres_list'] = movies_df['genres'].str.split('|')  # 注意这里是 | 分隔符
            
            # 获取所有类型
            all_genres = set()
            for genres in movies_df['genres_list'].dropna():
                all_genres.update(genres)
            
            # 创建类型特征矩阵
            genre_features = []
            for _, row in movies_df.iterrows():
                movie_genres = row['genres_list'] if isinstance(row['genres_list'], list) else []
                genre_row = {f'genre_{genre}': 1 if genre in movie_genres else 0 for genre in all_genres}
                genre_row['movie_id'] = row['movie_id']
                genre_features.append(genre_row)
            
            return pd.DataFrame(genre_features)
        except Exception as e:
            print(f"类型特征构建失败: {e}")
            return pd.DataFrame({'movie_id': movies_df['movie_id']})
    
    def _build_rating_features(self, ratings_df):
        """构建电影评分特征"""
        try:
            movie_stats = ratings_df.groupby('movie_id').agg({
                'rating': ['count', 'mean', 'std']
            }).reset_index()
            
            movie_stats.columns = ['movie_id', 'rating_count', 'avg_rating', 'rating_std']
            return movie_stats
        except Exception as e:
            print(f"评分特征构建失败: {e}")
            return pd.DataFrame({'movie_id': ratings_df['movie_id'].unique()})
    
    def reduce_dimensionality(self, features):
        """特征降维"""
        print("步骤5: 进行特征降维...")
        
        # 确保特征是数值型
        features = features.astype(float)
        
        # 标准化特征
        features_scaled = self.scaler.fit_transform(features)
        
        # PCA降维
        features_reduced = self.pca.fit_transform(features_scaled)
        
        print(f"降维后特征维度: {features_reduced.shape}")
        print(f"保留的方差比例: {np.sum(self.pca.explained_variance_ratio_):.4f}")
        
        return features_reduced
    
    def build_movie_profiles(self, features_reduced, n_clusters=8):
        """构建电影画像聚类"""
        print("步骤6: 构建电影画像聚类...")
        
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        movie_clusters = self.kmeans.fit_predict(features_reduced)
        
        return movie_clusters

class MappingModel:
    """用户-电影映射模型"""
    
    def __init__(self, n_factors=20, learning_rate=0.005, regularization=0.02, n_epochs=10):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.n_epochs = n_epochs
        self.user_factors = None
        self.movie_factors = None
        self.global_bias = None
        self.user_biases = None
        self.movie_biases = None
        self.train_losses = []
        
    def fit(self, ratings_matrix, user_features, movie_features):
        """训练映射模型"""
        print("=== 开始训练映射模型 ===")
        
        n_users, n_movies = ratings_matrix.shape
        
        # 初始化参数
        non_zero_ratings = ratings_matrix[ratings_matrix > 0]
        self.global_bias = np.mean(non_zero_ratings) if len(non_zero_ratings) > 0 else 3.0
        self.user_biases = np.zeros(n_users)
        self.movie_biases = np.zeros(n_movies)
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.movie_factors = np.random.normal(0, 0.1, (n_movies, self.n_factors))
        
        print("步骤1: 模型参数初始化完成")
        
        # 训练循环
        for epoch in range(self.n_epochs):
            print(f"步骤2: 训练第 {epoch+1}/{self.n_epochs} 轮次...")
            epoch_loss = 0
            n_ratings = 0
            
            for i in range(n_users):
                for j in range(n_movies):
                    if ratings_matrix[i, j] > 0:
                        # 预测评分
                        prediction = self._predict_single(i, j)
                        
                        # 计算误差
                        error = ratings_matrix[i, j] - prediction
                        
                        # 更新参数
                        self.user_biases[i] += self.learning_rate * (error - self.regularization * self.user_biases[i])
                        self.movie_biases[j] += self.learning_rate * (error - self.regularization * self.movie_biases[j])
                        
                        # 更新因子
                        for k in range(self.n_factors):
                            user_grad = error * self.movie_factors[j, k] - self.regularization * self.user_factors[i, k]
                            movie_grad = error * self.user_factors[i, k] - self.regularization * self.movie_factors[j, k]
                            
                            self.user_factors[i, k] += self.learning_rate * user_grad
                            self.movie_factors[j, k] += self.learning_rate * movie_grad
                        
                        epoch_loss += error ** 2
                        n_ratings += 1
            
            avg_loss = epoch_loss / n_ratings if n_ratings > 0 else 0
            self.train_losses.append(avg_loss)
            print(f"轮次 {epoch+1}, 平均损失: {avg_loss:.4f}")
        
        print("模型训练完成!")
        return self
    
    def _predict_single(self, user_idx, movie_idx):
        """预测单个评分"""
        try:
            prediction = (self.global_bias + 
                         self.user_biases[user_idx] + 
                         self.movie_biases[movie_idx] + 
                         np.dot(self.user_factors[user_idx], self.movie_factors[movie_idx]))
            
            # 确保预测值在合理范围内
            return np.clip(prediction, 0.5, 5.0)
        except Exception as e:
            print(f"预测失败 (用户{user_idx}, 电影{movie_idx}): {e}")
            return self.global_bias  # 返回全局平均值作为备选
    
    def predict(self, user_idx, movie_idx):
        """预测评分"""
        return float(self._predict_single(user_idx, movie_idx))  # 确保返回浮点数
    
    def recommend_for_user(self, user_idx, top_n=10):
        """为用户推荐电影"""
        n_movies = self.movie_factors.shape[0]
        predictions = []
        
        for movie_idx in range(n_movies):
            try:
                pred_rating = self.predict(user_idx, movie_idx)
                predictions.append((movie_idx, pred_rating))
            except Exception as e:
                print(f"推荐失败 (用户{user_idx}, 电影{movie_idx}): {e}")
                continue
        
        # 按预测评分排序
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:top_n]

class SimplifiedNewUserHandler:
    """简化版新用户处理器"""
    
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.knn = NearestNeighbors(n_neighbors=n_neighbors)
        self.user_basic_features = None
        
    def fit(self, users_df):
        """训练KNN模型"""
        print("=== 训练新用户处理模型 ===")
        
        # 只使用基本特征
        user_basic_features = users_df[['gender', 'age', 'occupation']].copy()
        user_basic_features['gender_encoded'] = user_basic_features['gender'].map({'M': 0, 'F': 1})
        
        # 选择数值型特征
        feature_columns = ['gender_encoded', 'age', 'occupation']
        self.user_basic_features = user_basic_features[feature_columns].values
        
        self.knn.fit(self.user_basic_features)
        print(f"KNN模型训练完成! 使用 {len(feature_columns)} 个基本特征")
        return self
    
    def handle_new_user(self, new_user_info):
        """处理新用户"""
        print("=== 处理新用户 ===")
        
        # 准备新用户特征
        new_user_features = self._prepare_new_user_features(new_user_info)
        
        # 找到相似用户
        distances, indices = self.knn.kneighbors([new_user_features])
        similar_users = indices[0]
        
        print(f"找到 {len(similar_users)} 个相似用户")
        
        return similar_users
    
    def _prepare_new_user_features(self, user_info):
        """准备新用户特征"""
        # 创建基本特征向量
        gender_encoded = 0 if user_info.get('gender', 'M') == 'M' else 1
        age = user_info.get('age', 25)
        occupation = user_info.get('occupation', 0)
        
        return np.array([gender_encoded, age, occupation])

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self):
        self.metrics = {}
        self.predictions = None
        self.actuals = None
        # 设置中文字体
        self.chinese_supported = setup_chinese_font()
    
    def evaluate(self, model, test_ratings):
        """评估模型性能"""
        print("=== 开始模型评估 ===")
        
        predictions = []
        actuals = []
        
        n_users, n_movies = test_ratings.shape
        
        print("步骤1: 生成预测结果...")
        count = 0
        for i in range(n_users):
            for j in range(n_movies):
                if test_ratings[i, j] > 0:
                    try:
                        pred = model.predict(i, j)
                        actual = test_ratings[i, j]
                        
                        predictions.append(float(pred))  # 确保是浮点数
                        actuals.append(float(actual))    # 确保是浮点数
                        count += 1
                    except Exception as e:
                        print(f"预测失败 (用户{i}, 电影{j}): {e}")
                        continue
        
        # 转换为numpy数组
        self.predictions = np.array(predictions, dtype=float)
        self.actuals = np.array(actuals, dtype=float)
        
        print(f"成功预测 {count} 个评分")
        
        if len(self.actuals) == 0:
            print("错误: 没有有效的预测结果")
            self.metrics = {'rmse': 0, 'mae': 0, 'explained_variance': 0}
            return self.metrics
        
        print("步骤2: 计算评估指标...")
        # 计算评估指标
        self.metrics['rmse'] = np.sqrt(mean_squared_error(self.actuals, self.predictions))
        self.metrics['mae'] = mean_absolute_error(self.actuals, self.predictions)
        
        # 计算解释方差 - 修复错误
        try:
            if len(self.actuals) > 1 and np.var(self.actuals) > 0:
                self.metrics['explained_variance'] = 1 - np.var(self.actuals - self.predictions) / np.var(self.actuals)
            else:
                self.metrics['explained_variance'] = 0
        except Exception as e:
            print(f"计算解释方差时出错: {e}")
            self.metrics['explained_variance'] = 0
        
        return self.metrics
    
    def visualize_evaluation(self):
        """可视化评估结果"""
        print("步骤3: 生成评估可视化...")
        
        if self.predictions is None or len(self.predictions) == 0:
            print("没有预测数据可用于可视化")
            return None
        
        # 根据字体支持情况选择标签语言
        if self.chinese_supported:
            titles = {
                'scatter': '预测vs实际评分',
                'residual': '残差分析',
                'metrics': '模型评估指标',
                'distribution': '评分分布对比',
                'xlabel_actual': '实际评分',
                'ylabel_pred': '预测评分',
                'xlabel_pred': '预测评分',
                'ylabel_residual': '残差',
                'xlabel_rating': '评分',
                'ylabel_freq': '频次',
                'legend_actual': '实际评分',
                'legend_pred': '预测评分'
            }
        else:
            titles = {
                'scatter': 'Predicted vs Actual Ratings',
                'residual': 'Residual Analysis',
                'metrics': 'Model Evaluation Metrics',
                'distribution': 'Rating Distribution Comparison',
                'xlabel_actual': 'Actual Rating',
                'ylabel_pred': 'Predicted Rating',
                'xlabel_pred': 'Predicted Rating',
                'ylabel_residual': 'Residual',
                'xlabel_rating': 'Rating',
                'ylabel_freq': 'Frequency',
                'legend_actual': 'Actual Rating',
                'legend_pred': 'Predicted Rating'
            }
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 预测vs实际散点图
        axes[0,0].scatter(self.actuals, self.predictions, alpha=0.5)
        min_val = min(self.actuals.min(), self.predictions.min())
        max_val = max(self.actuals.max(), self.predictions.max())
        axes[0,0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[0,0].set_xlabel(titles['xlabel_actual'])
        axes[0,0].set_ylabel(titles['ylabel_pred'])
        axes[0,0].set_title(titles['scatter'])
        
        # 2. 残差图
        residuals = self.actuals - self.predictions
        axes[0,1].scatter(self.predictions, residuals, alpha=0.5)
        axes[0,1].axhline(y=0, color='r', linestyle='--')
        axes[0,1].set_xlabel(titles['xlabel_pred'])
        axes[0,1].set_ylabel(titles['ylabel_residual'])
        axes[0,1].set_title(titles['residual'])
        
        # 3. 评估指标
        metrics_df = pd.DataFrame(list(self.metrics.items()), columns=['Metric', 'Value'])
        axes[1,0].bar(metrics_df['Metric'], metrics_df['Value'])
        axes[1,0].set_title(titles['metrics'])
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. 评分分布对比
        axes[1,1].hist(self.actuals, alpha=0.7, label=titles['legend_actual'], bins=20)
        axes[1,1].hist(self.predictions, alpha=0.7, label=titles['legend_pred'], bins=20)
        axes[1,1].set_xlabel(titles['xlabel_rating'])
        axes[1,1].set_ylabel(titles['ylabel_freq'])
        axes[1,1].set_title(titles['distribution'])
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.savefig('model_evaluation_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig

class MovieRecommendationSystem:
    """电影推荐系统主类"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        # 先设置全局字体
        setup_chinese_font()
        
        self.user_builder = UserProfileBuilder()
        self.movie_builder = MovieProfileBuilder()
        self.mapping_model = None
        self.new_user_handler = None
        self.evaluator = ModelEvaluator()
        
        # 数据存储
        self.users_df = None
        self.movies_df = None
        self.ratings_df = None
        self.user_features = None
        self.movie_features = None
        self.user_profiles = None
        self.movie_profiles = None
        self.ratings_matrix = None
    
    def load_data(self):
        """加载数据"""
        print("=== 加载数据 ===")
        
        try:
            # 加载用户数据
            print("步骤1: 加载用户数据...")
            self.users_df = pd.read_table(
                f"{self.data_path}/users.dat", sep="::",
                header=None, names=["user_id", "gender", "age", "occupation", "zip_code"],
                engine='python', encoding='latin1'
            )
            
            # 加载电影数据
            print("步骤2: 加载电影数据...")
            self.movies_df = pd.read_table(
                f"{self.data_path}/movies.dat", sep="::",
                header=None, names=["movie_id", "title", "genres"],
                engine='python', encoding='latin1'
            )
            
            # 加载评分数据
            print("步骤3: 加载评分数据...")
            self.ratings_df = pd.read_table(
                f"{self.data_path}/ratings.dat", sep="::",
                header=None, names=["user_id", "movie_id", "rating", "timestamp"],
                engine='python', encoding='latin1'
            )
            
            print("数据加载完成!")
            print(f"用户数: {len(self.users_df)}")
            print(f"电影数: {len(self.movies_df)}")
            print(f"评分记录数: {len(self.ratings_df)}")
            
        except Exception as e:
            print(f"数据加载失败: {e}")
            # 使用示例数据作为备选
            self._create_sample_data()
    
    def _create_sample_data(self):
        """创建示例数据（备用）"""
        print("使用示例数据...")
        
        # 创建示例用户数据
        self.users_df = pd.DataFrame({
            'user_id': range(1, 101),
            'gender': np.random.choice(['M', 'F'], 100),
            'age': np.random.choice([1, 18, 25, 35, 45, 50, 56], 100),
            'occupation': np.random.randint(0, 21, 100),
            'zip_code': ['00000'] * 100
        })
        
        # 创建示例电影数据
        genres_list = ['Action', 'Comedy', 'Drama', 'Thriller', 'Romance', 'Sci-Fi']
        self.movies_df = pd.DataFrame({
            'movie_id': range(1, 51),
            'title': [f'Movie {i}' for i in range(1, 51)],
            'genres': ['|'.join(np.random.choice(genres_list, 2, replace=False)) for _ in range(50)]
        })
        
        # 创建示例评分数据
        self.ratings_df = pd.DataFrame({
            'user_id': np.random.randint(1, 101, 1000),
            'movie_id': np.random.randint(1, 51, 1000),
            'rating': np.random.randint(1, 6, 1000),
            'timestamp': np.random.randint(0, 1000000, 1000)
        })
    
    def preprocess_data(self):
        """数据预处理"""
        print("=== 数据预处理 ===")
        
        # 处理电影发布年份
        print("步骤1: 处理电影发布年份...")
        self.movies_df['release_year'] = self.movies_df['title'].str.extract(r'\((\d{4})\)')
        self.movies_df['release_year'] = pd.to_numeric(self.movies_df['release_year'], errors='coerce')
        self.movies_df['release_year'] = self.movies_df['release_year'].fillna(1990)
        
        # 清理电影标题
        self.movies_df['title'] = self.movies_df['title'].str.replace(r'\(\d{4}\)', '', regex=True).str.strip()
        
        # 处理评分数据
        print("步骤2: 处理评分数据...")
        self.ratings_df['rating'] = pd.to_numeric(self.ratings_df['rating'], errors='coerce')
        self.ratings_df = self.ratings_df.dropna(subset=['rating'])
        
        print("数据预处理完成!")
    
    def build_profiles(self):
        """构建用户和电影画像"""
        print("=== 构建用户和电影画像 ===")
        
        # 构建用户画像
        print("步骤1: 构建用户画像...")
        user_features_matrix, self.user_features = self.user_builder.build_user_features(
            self.users_df, self.ratings_df, self.movies_df
        )
        
        # 检查是否有足够的特征进行降维
        if user_features_matrix.shape[1] > 1:
            user_features_reduced = self.user_builder.reduce_dimensionality(user_features_matrix)
            self.user_profiles = self.user_builder.build_user_profiles(user_features_reduced)
            
            # 可视化用户画像
            self.user_builder.visualize_user_profiles(
                user_features_reduced, self.user_profiles, self.user_features
            )
        else:
            print("用户特征不足，跳过降维和聚类")
            user_features_reduced = user_features_matrix.values
            self.user_profiles = np.zeros(len(user_features_matrix))
        
        # 构建电影画像
        print("步骤2: 构建电影画像...")
        movie_features_matrix, self.movie_features = self.movie_builder.build_movie_features(
            self.movies_df, self.ratings_df
        )
        
        # 检查是否有足够的特征进行降维
        if movie_features_matrix.shape[1] > 1:
            movie_features_reduced = self.movie_builder.reduce_dimensionality(movie_features_matrix)
            self.movie_profiles = self.movie_builder.build_movie_profiles(movie_features_reduced)
        else:
            print("电影特征不足，跳过降维和聚类")
            movie_features_reduced = movie_features_matrix.values
            self.movie_profiles = np.zeros(len(movie_features_matrix))
        
        print("用户和电影画像构建完成!")
        
        return user_features_reduced, movie_features_reduced
    
    def create_ratings_matrix(self):
        """创建评分矩阵"""
        print("=== 创建评分矩阵 ===")
        
        # 创建用户-电影评分矩阵
        n_users = len(self.users_df)
        n_movies = len(self.movies_df)
        
        self.ratings_matrix = np.zeros((n_users, n_movies))
        
        for _, rating in self.ratings_df.iterrows():
            user_idx = rating['user_id'] - 1  # 转换为0-based索引
            movie_idx = rating['movie_id'] - 1
            
            if user_idx < n_users and movie_idx < n_movies:
                self.ratings_matrix[user_idx, movie_idx] = rating['rating']
        
        print(f"评分矩阵形状: {self.ratings_matrix.shape}")
        return self.ratings_matrix
    
    def train_test_split(self, test_size=0.2):
        """划分训练集和测试集"""
        print("=== 划分训练集和测试集 ===")
        
        # 创建评分数据的DataFrame版本用于划分
        ratings_for_split = self.ratings_df.copy()
        
        # 确保有足够的数据进行划分
        if len(ratings_for_split) < 10:
            print("警告: 数据量太少，使用全部数据作为训练集")
            train_ratings = ratings_for_split
            test_ratings = pd.DataFrame(columns=ratings_for_split.columns)
        else:
            # 划分训练集和测试集
            train_ratings, test_ratings = train_test_split(
                ratings_for_split, test_size=test_size, random_state=42
            )
        
        # 创建训练和测试评分矩阵
        n_users = len(self.users_df)
        n_movies = len(self.movies_df)
        
        train_matrix = np.zeros((n_users, n_movies))
        test_matrix = np.zeros((n_users, n_movies))
        
        for _, rating in train_ratings.iterrows():
            user_idx = rating['user_id'] - 1
            movie_idx = rating['movie_id'] - 1
            if user_idx < n_users and movie_idx < n_movies:
                train_matrix[user_idx, movie_idx] = rating['rating']
        
        for _, rating in test_ratings.iterrows():
            user_idx = rating['user_id'] - 1
            movie_idx = rating['movie_id'] - 1
            if user_idx < n_users and movie_idx < n_movies:
                test_matrix[user_idx, movie_idx] = rating['rating']
        
        print(f"训练集大小: {len(train_ratings)}")
        print(f"测试集大小: {len(test_ratings)}")
        
        return train_matrix, test_matrix
    
    def train_mapping_model(self, train_ratings):
        """训练映射模型"""
        print("=== 训练用户-电影映射模型 ===")
        
        self.mapping_model = MappingModel(n_factors=20, n_epochs=10)
        self.mapping_model.fit(train_ratings, None, None)  # 简化版本，不使用额外特征
        
        print("映射模型训练完成!")
    
    def evaluate_model(self, test_ratings):
        """评估模型"""
        print("=== 评估模型性能 ===")
        
        metrics = self.evaluator.evaluate(self.mapping_model, test_ratings)
        
        # 显示评估结果
        print("\n=== 模型评估结果 ===")
        for metric, value in metrics.items():
            print(f"{metric.upper()}: {value:.4f}")
        
        # 可视化评估结果
        self.evaluator.visualize_evaluation()
        
        return metrics
    
    def recommend_for_new_user(self, new_user_info):
        """为新用户推荐电影"""
        print("=== 为新用户生成推荐 ===")
        
        # 训练新用户处理器（如果尚未训练）
        if self.new_user_handler is None:
            self.new_user_handler = SimplifiedNewUserHandler(n_neighbors=5)
            self.new_user_handler.fit(self.users_df)
        
        # 使用KNN找到相似用户
        similar_users = self.new_user_handler.handle_new_user(new_user_info)
        
        # 为相似用户生成推荐
        print("\n=== 新用户电影推荐 ===")
        if len(similar_users) > 0:
            # 使用第一个相似用户的画像
            similar_user_idx = similar_users[0]
            recommendations = self.mapping_model.recommend_for_user(similar_user_idx, top_n=10)
            
            for i, (movie_idx, pred_rating) in enumerate(recommendations, 1):
                if movie_idx < len(self.movies_df):
                    movie_info = self.movies_df.iloc[movie_idx]
                    print(f"{i}. {movie_info['title']} (预测评分: {pred_rating:.2f})")
                else:
                    print(f"{i}. 电影ID {movie_idx} (预测评分: {pred_rating:.2f})")
            
            return recommendations
        else:
            print("未找到相似用户，无法生成推荐")
            return []
    
    def _prepare_new_user_features(self, user_info):
        """准备新用户特征"""
        # 创建基本特征向量
        gender_encoded = 0 if user_info.get('gender', 'M') == 'M' else 1
        age = user_info.get('age', 25)
        occupation = user_info.get('occupation', 0)
        
        # 返回基本特征 - 现在NewUserHandler会自动处理维度不匹配问题
        return np.array([gender_encoded, age, occupation])
    
    def run_complete_pipeline(self):
        """运行完整的推荐系统流水线"""
        print("开始运行电影推荐系统...")
        
        try:
            # 1. 加载数据
            self.load_data()
            
            # 2. 数据预处理
            self.preprocess_data()
            
            # 3. 构建画像
            user_features_reduced, movie_features_reduced = self.build_profiles()
            
            # 4. 创建评分矩阵
            self.create_ratings_matrix()
            
            # 5. 划分训练测试集
            train_ratings, test_ratings = self.train_test_split()
            
            # 6. 训练模型
            self.train_mapping_model(train_ratings)
            
            # 7. 评估模型
            metrics = self.evaluate_model(test_ratings)
            
            print("\n=== 系统运行完成 ===")
            return metrics
            
        except Exception as e:
            print(f"系统运行失败: {e}")
            import traceback
            traceback.print_exc()
            return None

# 运行系统
# if __name__ == "__main__":
#     # 初始化系统 - 修正数据路径
#     system = MovieRecommendationSystem("C:/Users/我为玉露/Desktop/study/7008 project/数据集/")
    
#     # 运行完整流水线
#     results = system.run_complete_pipeline()
    
#     # 示例：为新用户推荐
#     if results is not None:
#         print("\n" + "="*50)
#         print("演示新用户推荐功能...")
#         new_user_example = {
#             'gender': 'M',
#             'age': 25,
#             'occupation': 12,  # 程序员
#             'zip_code': '100000'
#         }
#         system.recommend_for_new_user(new_user_example)