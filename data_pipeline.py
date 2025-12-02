import requests
import json
import time
import pandas as pd
import numpy as np
import random
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from snownlp import SnowNLP
from typing import List, Dict, Optional

# =======================================================================
#                  A. USER-DEFINED VARIABLES (已配置您的Cookie)
# =======================================================================

YOUR_BILI_USER_ID = "946974"  # 目标用户 ID

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36',
    'Referer': f'https://space.bilibili.com/{YOUR_BILI_USER_ID}/',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Origin': 'https://www.bilibili.com',
    # 🔴 关键反爬措施：已包含您的完整 Cookie 字符串
    'Cookie': 'buvid3=AF87EB3B-C6FB-0C36-BEDC-9173830BE99C74853infoc; b_nut=1764666074; b_lsid=D34101AE5_19ADE4B6C33; bsource=search_google; _uuid=7D68CE2B-8272-D1023-55E2-279AAABFC52D76219infoc; home_feed_column=4; browser_resolution=1320-835; buvid4=F25EAD55-770A-8298-BF3F-0FB06763E2E775369-025120217-o4gpl8d2PA+5CpOKUkvk9Q%3D%3D; buvid_fp=68f9729fe4a895c9d1c6342b989a4951; SESSDATA=e5367015%2C1780218100%2C6db91%2Ac2CjAsWIJyR-LE8xB1yITnWRwyxghCgpsU0VlRTvrcRubfi1AWXkMXXm-jCs5Jzl2na-YSVnFSRi1EbzFVN3NQZTMxMk9hV3JadlFENDVsTHNRUWU2RWxwV01Pa1dNcGRJU3MzcFBRcUlGZUJXNmd4enpnTGZ3YkRXZ2pTTWpVXzR0dkE3b2l0Rm5RIIEC; bili_jct=0b92cee94fc7acd03102fd25682cd34c; DedeUserID=454147076; DedeUserID__ckMd5=2e4c0e13bc1cc6ce; theme-tip-show=SHOWED; bili_ticket=eyJhbGciOiJIUzI1NiIsImtpZCI6InMwMyIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3NjQ5MjUzMTgsImlhdCI6MTc2NDY2NjA1OCwicGx0IjotMX0.z93310JCHruE10gv2RmjXc_z5AXCduZwwcKKWf6X35M; bili_ticket_expires=1764925258; theme-avatar-tip-show=SHOWED; CURRENT_FNVAL=2000; sid=emrz6lop',
}

API_URLS = {
    # 🔴 移除冗余的 'video_stat' 接口，只保留获取完整数据的 'video_view' 接口
    'video_view': 'https://api.bilibili.com/x/web-interface/view?bvid={bvid}',
    # 评论数据 API
    'comment_data': 'https://api.bilibili.com/x/v2/reply/main?oid={bvid}&type=1&pn={page}&ps=20',
    # 获取用户公开信息 API
    'user_info': 'https://api.bilibili.com/x/space/acc/info?mid={user_id}',
}


# =======================================================================
#                  B. ROBUST SCRAPING CLASS
# =======================================================================

class BiliDataPipeline:
    def __init__(self, mid: str):
        self.mid = mid
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.api_urls = API_URLS

        self.request_count = 0
        self.max_requests_per_minute = 10
        self.start_time = time.time()
        print(f"BiliDataPipeline initialized for MID: {mid}")

    def safe_request(self, url: str, params: Optional[Dict] = None, delay: float = 2.0, timeout: int = 15):
        """安全的请求方法，包含频率控制和随机延迟"""
        self.request_count += 1
        current_time = time.time()
        if (current_time - self.start_time < 60 and self.request_count >= self.max_requests_per_minute):
            wait_time = 60 - (current_time - self.start_time) + 1
            print(f"⏰ 频率控制: 等待 {wait_time:.1f} 秒")
            time.sleep(wait_time)
            self.request_count = 0
            self.start_time = time.time()

        total_delay = random.uniform(1, 3) + delay
        time.sleep(total_delay)

        try:
            response = self.session.get(url, params=params, timeout=timeout)
            return response
        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching {url}: {e}")
            return None

    def safe_json_parse(self, response) -> Optional[Dict]:
        """安全解析JSON，避免解码错误并检查状态码"""
        if response is None:
            return None

        try:
            if response.status_code == 412:
                print("❌ HTTP 412: Precondition Failed. WBI signature likely required/expired.")
                return None
            if response.status_code != 200:
                print(f"❌ HTTP状态码异常: {response.status_code}")
                if response.status_code in [403, 429]:
                    print("🚫 触发反爬/频率限制，等待60秒")
                    time.sleep(60)
                return None

            content_type = response.headers.get("content-type", "")
            if "application/json" not in content_type:
                print(f"⚠️ 响应不是JSON格式: {content_type}")
                return None
            if not response.text.strip():
                print("⚠️ 响应内容为空")
                return None

            data = response.json()
            if data.get('code') != 0:
                print(f"❌ API returned error code {data.get('code')}: {data.get('message', 'Unknown API Error')}")
                return None

            return data
        except json.JSONDecodeError as e:
            print(f"❌ JSON解析失败: {e}")
            return None
        except Exception as e:
            print(f"❌ 解析响应时发生异常: {e}")
            return None

    # ----------------------------------------------------
    # UPDATED: Relying solely on 'video_view' for all metrics
    # ----------------------------------------------------

    def get_video_stats(self, bvid: str) -> Optional[Dict]:
        """Fetch all metadata and statistics using the robust video_view API."""
        video_info = {'bvid': bvid}

        # 1. Fetch View/Metadata
        view_url = self.api_urls['video_view'].format(bvid=bvid)
        # 此请求最关键，使用较高的基础延迟
        view_response = self.safe_request(view_url, delay=3.0)
        view_data = self.safe_json_parse(view_response)

        if view_data and 'data' in view_data:
            data = view_data['data']
            stat = data['stat']  # 统计数据全部包含在 stat 字段中

            video_info.update({
                'title': data.get('title'),
                'duration_sec': data.get('duration'),
                'pubdate': data.get('pubdate'),
                # 提取完整的统计数据
                'views': stat.get('view', 0),
                'likes': stat.get('like', 0),
                'coins': stat.get('coin', 0),
                'favorites': stat.get('favorite', 0),
                'shares': stat.get('share', 0),
                'comment_count': stat.get('reply', 0),
                'danmaku_count': stat.get('danmaku', 0),
            })

            print(f"✅ 视频元数据和统计数据获取成功 (基于 video_view API).")
            return video_info
        else:
            print(f"❌ 无法获取视频 {bvid} 的元数据。可能视频不存在或触发反爬。")
            return None

    def scrape_comments_for_video(self, bvid: str, max_pages: int = 5) -> pd.DataFrame:
        """Scrapes main comments for a single video."""
        # ... (评论数据获取逻辑保持不变)
        all_comments = []
        page = 1

        print(f"--- Starting Comment Scrape for BVID: {bvid} (Max {max_pages} pages) ---")

        while page <= max_pages:
            url = self.api_urls['comment_data'].format(bvid=bvid, page=page)
            response = self.safe_request(url, delay=2.0)
            data = self.safe_json_parse(response)

            if data and data['data'].get('replies'):
                for reply in data['data']['replies']:
                    comment_info = {
                        'bvid': bvid,
                        'comment_text': reply['content']['message'],
                        'timestamp': reply['ctime'],
                        'user_id': reply['member']['mid'],
                    }
                    all_comments.append(comment_info)
                page += 1
            else:
                break

        print(f"Scraped {len(all_comments)} comments for {bvid}.")
        return pd.DataFrame(all_comments)

    def scrape_active_user_data(self, df_all_comments: pd.DataFrame, max_users: int = 50) -> pd.DataFrame:
        """通过评论用户ID间接爬取用户公开数据，用于粉丝画像。"""
        # ... (用户数据获取逻辑保持不变，但请注意延迟设置)
        if df_all_comments.empty:
            print("⚠️ 评论数据为空，无法爬取活跃用户数据。")
            return pd.DataFrame()

        unique_mids = df_all_comments['user_id'].unique()
        mids_to_scrape = unique_mids[:max_users]

        print(f"--- Starting Active User Data Scrape ({len(mids_to_scrape)} users) ---")

        user_data_list = []

        for i, mid in enumerate(mids_to_scrape):
            url = self.api_urls['user_info'].format(user_id=mid)
            # 访问用户主页 API 风险较高，建议保持或增加延迟
            response = self.safe_request(url, delay=5.0)
            data = self.safe_json_parse(response)

            if data and 'data' in data:
                user = data['data']
                user_info = {
                    'user_id': mid,
                    'user_name': user.get('name'),
                    'Level': user.get('level', 0),
                    'Sex': user.get('sex', '未知'),
                    'Sign_Length': len(user.get('sign', '')),
                }
                user_data_list.append(user_info)
                if (i + 1) % 10 == 0:
                    print(f"✅ 已成功爬取 {i + 1}/{len(mids_to_scrape)} 个用户数据。")
            else:
                print(f"❌ 无法获取用户 ID {mid} 的公开信息。")

        return pd.DataFrame(user_data_list)


# =======================================================================
#                      C. DATA WASHING/QUANTIFICATION FUNCTIONS
# =======================================================================

def quantify_fan_personas(df_fans: pd.DataFrame) -> pd.DataFrame:
    """使用爬取到的公开用户特征进行粉丝画像聚类。"""

    if df_fans.empty or not all(col in df_fans.columns for col in ['Level', 'Sign_Length']):
        print("⚠️ 真实用户特征不足或爬取失败，使用模拟数据进行 K-Means 演示。")
        # 模拟数据 (回退)
        df_fans = pd.DataFrame({
            'user_id': np.arange(50),
            'Level': np.random.randint(1, 6, 50),
            'Sign_Length': np.random.rand(50) * 100,
            'Sex_Encoded': np.random.randint(0, 3, 50)
        })
        features = ['Level', 'Sign_Length', 'Sex_Encoded']
        df_fans['Sex_Encoded'] = df_fans['Sex_Encoded'].replace({0: '男', 1: '女', 2: '未知'}).astype(
            'category').cat.codes
    else:
        print(f"✅ 使用 {len(df_fans)} 条爬取到的活跃用户数据进行画像聚类。")
        # 预处理真实数据
        df_fans['Sex_Encoded'] = df_fans['Sex'].astype('category').cat.codes
        features = ['Level', 'Sign_Length', 'Sex_Encoded']

    X = df_fans[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df_fans['Persona_Cluster'] = kmeans.fit_predict(X_scaled)

    return df_fans


def quantify_emotional_curve(df_comments: pd.DataFrame) -> pd.DataFrame:
    """Calculates sentiment score for comments (for emotional curve)。"""
    if df_comments.empty:
        return pd.DataFrame({'Minute': [], 'Average_Sentiment': []})

    def get_sentiment(text):
        """Calculates sentiment using SnowNLP (0=Negative, 1=Positive)."""
        try:
            return SnowNLP(text).sentiments
        except:
            return 0.5

    df_comments['Sentiment_Score'] = df_comments['comment_text'].apply(get_sentiment)
    df_comments['Normalized_Score'] = (df_comments['Sentiment_Score'] - 0.5) * 2
    df_comments['Minute'] = (df_comments['timestamp'] / 60).astype(int)

    emotional_curve = df_comments.groupby(['bvid', 'Minute'])['Normalized_Score'].mean().reset_index()
    emotional_curve.rename(columns={'Normalized_Score': 'Average_Sentiment'}, inplace=True)

    return emotional_curve


# =======================================================================
#                           D. MAIN EXECUTION
# =======================================================================

if __name__ == '__main__':
    pipeline = BiliDataPipeline(YOUR_BILI_USER_ID)

    VIDEO_LIST_FILE = "video_list.txt"
    all_video_metrics = []
    df_all_comments = pd.DataFrame()

    try:
        # 1. 读取 BVID 列表文件
        with open(VIDEO_LIST_FILE, 'r', encoding='utf-8') as f:
            bv_ids = [line.strip() for line in f if line.strip() and not line.startswith('#')]

        if not bv_ids:
            print(f"❌ 文件 '{VIDEO_LIST_FILE}' 为空或不存在有效的 BVID。")
        else:
            print(f"📋 成功读取 {len(bv_ids)} 个视频 BVID 进行爬取。")

            # 2. 循环爬取每个视频的元数据和评论
            for i, bvid in enumerate(bv_ids, 1):
                print(f"\n{'=' * 50}")
                print(f"🎬 正在处理视频 {i}/{len(bv_ids)}: {bvid}")

                # 2.1 爬取视频元数据和核心指标 (现在只依赖 video_view API)
                metrics = pipeline.get_video_stats(bvid)
                if metrics:
                    all_video_metrics.append(metrics)

                    # 2.2 爬取评论数据 (用于情绪曲线和用户ID提取)
                    df_comments = pipeline.scrape_comments_for_video(bvid, max_pages=10)
                    if not df_comments.empty:
                        df_all_comments = pd.concat([df_all_comments, df_comments], ignore_index=True)

                if i < len(bv_ids):
                    # 视频间强制休息 10-15 秒
                    wait_time = random.uniform(10, 15)
                    print(f"⏳ 视频间休息 {wait_time:.1f} 秒...")
                    time.sleep(wait_time)

            # 3. 数据清洗和量化分析
            df_videos = pd.DataFrame(all_video_metrics)

            # 3.2 爬取活跃用户数据 (间接画像数据)
            df_fans_real_data = pipeline.scrape_active_user_data(df_all_comments, max_users=50)

            # 3.3 粉丝画像（使用爬取到的数据进行聚类）
            df_fans_quantified = quantify_fan_personas(df_fans_real_data)

            # 3.4 情绪曲线分析
            df_emotional_curve = quantify_emotional_curve(df_all_comments)

            # 4. 存储结果
            df_videos.to_csv("processed_video_metadata.csv", index=False, encoding='utf-8-sig')
            df_all_comments.to_csv("raw_comments_with_sentiment.csv", index=False, encoding='utf-8-sig')
            df_fans_quantified.to_csv("processed_fan_personas.csv", index=False, encoding='utf-8-sig')
            df_emotional_curve.to_csv("emotional_curve_data.csv", index=False, encoding='utf-8-sig')

            print("\n✅ 数据管道完成。CSV 文件已生成。")

    except FileNotFoundError:
        print(f"❌ 错误: 文件 '{VIDEO_LIST_FILE}' 未找到。请创建此文件并填入 BVID 列表。")
    except Exception as e:
        print(f"❌ 运行时发生致命错误: {e}")