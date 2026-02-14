import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import glob
from sklearn.feature_extraction.text import TfidfVectorizer

# 설정
DATA_DIR = 'data'
st.set_page_config(page_title="Naver Shopping Insight Dashboard", layout="wide")

# --- 데이터 로드 함수 ---
@st.cache_data
def load_all_data():
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    
    # 일반 키워드 데이터 (오메가3, 비타민D, 트렌치코트 등)
    # dtype: trend, shopping, blog
    main_data = {}
    
    # 특수 분석 데이터
    special_data = {
        'seasonal_trend': None,      # 절기별 패션 트랜드
        'trench_attributes': None,   # 트렌치코트 세부 속성
        'trench_gender': None,       # 트렌치코트 성별
        'trench_age': None           # 트렌치코트 연령
    }

    for f in files:
        name = os.path.basename(f)
        df = pd.read_csv(f)
        
        # 특수 파일 처리
        if '절기별_패션트랜드' in name:
            special_data['seasonal_trend'] = df
            continue
        if '트렌치코트_세부속성' in name:
            special_data['trench_attributes'] = df
            continue
        if '트렌치코트_성별' in name:
            special_data['trench_gender'] = df
            continue
        if '트렌치코트_연령별' in name:
            special_data['trench_age'] = df
            continue

        # 일반 키워드 파일 처리
        parts = name.split('_')
        keyword = parts[0]
        dtype = ""
        if '쇼핑트랜드' in name:
            dtype = 'trend'
        elif '네이버쇼핑' in name:
            dtype = 'shopping'
        elif '블로그게시물' in name:
            dtype = 'blog'
        
        if dtype:
            if keyword not in main_data:
                main_data[keyword] = {}
            main_data[keyword][dtype] = df
            
    return main_data, special_data

def extract_keywords_tfidf(texts, top_n=20):
    if not texts:
        return pd.DataFrame()
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    sums = tfidf_matrix.sum(axis=0)
    data = []
    for col, idx in enumerate(feature_names):
        data.append((idx, sums[0, col]))
    ranking = sorted(data, key=lambda x: x[1], reverse=True)
    return pd.DataFrame(ranking[:top_n], columns=['Keyword', 'Score'])

# --- 앱 메인 로직 ---
main_data, special_data = load_all_data()
all_keywords = list(main_data.keys())

st.title("🚀 Naver API 쇼핑 트렌드 통합 대시보드 (V2)")
st.markdown("수집된 데이터를 바탕으로 키워드 트렌드, 쇼핑 현황, 인구통계 및 절기별 피크를 분석합니다.")

# 사이드바
st.sidebar.header("🔍 분석 설정")
selected_keywords = st.sidebar.multiselect("분석할 키워드 선택", all_keywords, default=all_keywords)

if not selected_keywords:
    st.warning("분석할 키워드를 선택해주세요.")
else:
    tabs = st.tabs(["통합 트랜드", "쇼핑 인사이트", "블로그 분석", "인구통계 분석", "심층 분석", "런칭 전략 (Deck)"])

    # --- Tab 1: 통합 트랜드 ---
    with tabs[0]:
        st.header("키워드별 검색지수 추이")
        
        trend_df_list = []
        for kw in selected_keywords:
            if 'trend' in main_data[kw]:
                df = main_data[kw]['trend'].copy()
                df['keyword'] = kw
                df['period'] = pd.to_datetime(df['period'])
                # 이동평균 계산 (7일)
                df['ma7'] = df['ratio'].rolling(window=7).mean()
                trend_df_list.append(df)
        
        if trend_df_list:
            full_trend_df = pd.concat(trend_df_list)
            
            # [그래프 1] 시계열 추이 (Plotly Line)
            fig_trend = px.line(full_trend_df, x='period', y='ratio', color='keyword', 
                                title="일별 검색지수 추이 (상대값)", labels={'ratio': '검색지수', 'period': '날짜'})
            st.plotly_chart(fig_trend, use_container_width=True)
            
            # [그래프 2] 이동평균선 (Plotly Line)
            fig_ma = px.line(full_trend_df, x='period', y='ma7', color='keyword',
                             title="7일 이동평균 검색 추세", labels={'ma7': '이동평균 (7일)', 'period': '날짜'})
            st.plotly_chart(fig_ma, use_container_width=True)
            
            # [표 1] 요약 통계
            st.subheader("검색지수 요약 통계")
            trend_summary = full_trend_df.groupby('keyword')['ratio'].agg(['mean', 'max', 'min', 'std']).reset_index()
            st.table(trend_summary)
            
            # 절기별 패션 데이터가 있다면 추가 표시
            if special_data['seasonal_trend'] is not None:
                st.subheader("입춘 전후 패션 카테고리 트랜드")
                sea_df = special_data['seasonal_trend'].copy()
                sea_df['period'] = pd.to_datetime(sea_df['period'])
                fig_sea = px.line(sea_df, x='period', y='ratio', color='keyword',
                                  title="패션 카테고리 입춘(2/3) 전후 트랜드 (동일 요청 기준)")
                try:
                    # 'period'가 datetime이므로 x값도 datetime 객체로 전달
                    ipchun_dt = pd.to_datetime('2025-02-03')
                    fig_sea.add_vline(x=ipchun_dt.timestamp() * 1000, line_dash="dash", line_color="red", annotation_text="입춘")
                except Exception:
                    try:
                        fig_sea.add_vline(x='2025-02-03', line_dash="dash", line_color="red", annotation_text="입춘")
                    except Exception as e:
                        st.warning(f"절기 세로선 표시 제한: {e}")
                st.plotly_chart(fig_sea, use_container_width=True)
        else:
            st.info("트랜드 데이터가 부족합니다.")

    # --- Tab 2: 쇼핑 인사이트 ---
    with tabs[1]:
        st.header("네이버 쇼핑 검색 결과 분석")
        
        shopping_dfs = []
        for kw in selected_keywords:
            if 'shopping' in main_data[kw]:
                df = main_data[kw]['shopping'].copy()
                df['keyword'] = kw
                df['lprice'] = pd.to_numeric(df['lprice'], errors='coerce')
                shopping_dfs.append(df)
        
        if shopping_dfs:
            full_shop_df = pd.concat(shopping_dfs)
            
            col1, col2 = st.columns(2)
            with col1:
                # [그래프 3] 가격 분포 히스토그램
                fig_hist = px.histogram(full_shop_df, x='lprice', color='keyword', barmode='overlay',
                                        title="키워드별 가격 분포", labels={'lprice': '최저가 (원)'})
                st.plotly_chart(fig_hist, use_container_width=True)
                
                # [표 2] 가격 기술 통계
                st.subheader("가격 기술통계")
                st.dataframe(full_shop_df.groupby('keyword')['lprice'].describe())

            with col2:
                # [그래프 4] 브랜드 점유율 (상위 10개)
                st.subheader("주요 브랜드 점유율")
                brand_data = full_shop_df.groupby(['keyword', 'brand']).size().reset_index(name='count')
                brand_data = brand_data.sort_values(by=['keyword', 'count'], ascending=[True, False]).groupby('keyword').head(10)
                fig_brand = px.bar(brand_data, x='count', y='brand', color='keyword', orientation='h',
                                   title="키워드별 상위 브랜드 (노출 건수)")
                st.plotly_chart(fig_brand, use_container_width=True)
                
                # [표 3] 쇼핑몰별 최저가 요약
                st.subheader("쇼핑몰별 평균 낙찰가(L-Price)")
                mall_stats = full_shop_df.groupby('mallName')['lprice'].agg(['mean', 'count']).sort_values(by='count', ascending=False).head(10)
                st.table(mall_stats)
        else:
            st.info("쇼핑 검색 데이터가 없습니다.")

    # --- Tab 3: 블로그 분석 ---
    with tabs[2]:
        st.header("블로그 기반 감성 및 관심 키워드 분석")
        
        for kw in selected_keywords:
            if 'blog' in main_data[kw]:
                st.subheader(f"'{kw}' 검색어 연관 키워드 (TF-IDF)")
                df_blog = main_data[kw]['blog']
                texts = df_blog['description'].str.replace('<b>', '').str.replace('</b>', '').fillna('').tolist()
                tfidf_res = extract_keywords_tfidf(texts)
                
                if not tfidf_res.empty:
                    # [그래프 5] TF-IDF Bar Chart
                    fig_tfidf = px.bar(tfidf_res.head(15), x='Score', y='Keyword', orientation='h',
                                       color='Score', title=f"'{kw}' 주요 핵심 키워드")
                    st.plotly_chart(fig_tfidf, use_container_width=True)
                    
                    # [표 4] 키워드 점수 데이터
                    with st.expander(f"'{kw}' 상세 키워드 점수 보기"):
                        st.dataframe(tfidf_res)
                else:
                    st.write("텍스트 데이터가 부족합니다.")

    # --- Tab 4: 인구통계 분석 ---
    with tabs[3]:
        st.header("인구통계학적 수요 비중 분석 (Demographics)")
        
        if special_data['trench_gender'] is not None and '트렌치코트' in selected_keywords:
            st.subheader("트렌치코트 성별/연령 관심도")
            
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                # [그래프 6] 성별 비중 Pie Chart
                g_df = special_data['trench_gender']
                # 'group' 컬럼이 성별 코드(f, m)
                g_avg = g_df.groupby('group')['ratio'].mean().reset_index()
                g_avg['gender'] = g_avg['group'].map({'f': '여성', 'm': '남성'})
                fig_g = px.pie(g_avg, values='ratio', names='gender', title="트렌치코트 전체 성별 비중")
                st.plotly_chart(fig_g, use_container_width=True)

            with col_d2:
                # [그래프 7] 연령대별 비중 Bar Chart
                a_df = special_data['trench_age']
                # 'group' 컬럼이 연령대 코드
                a_avg = a_df.groupby('group')['ratio'].mean().reset_index()
                age_map = {'10': '10대', '20': '20대', '30': '30대', '40': '40대', '50': '50대', '60': '60대+'}
                a_avg['age_group'] = a_avg['group'].astype(str).map(age_map)
                fig_a = px.bar(a_avg, x='age_group', y='ratio', title="트렌치코트 연령대별 관심도 (평균 지수)",
                               labels={'ratio': '평균 검색지수', 'age_group': '연령대'})
                st.plotly_chart(fig_a, use_container_width=True)
            
            # [표 5] 인구통계 요약 테이블
            st.subheader("인구통계 데이터 요약")
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.write("성별 비중 (%)")
                st.dataframe(g_avg.set_index('gender'))
            with col_t2:
                st.write("연령별 지수")
                st.dataframe(a_avg.set_index('age_group'))
        elif '트렌치코트' not in selected_keywords:
            st.info("'트렌치코트'를 선택하면 인구통계 데이터를 확인할 수 있습니다.")
        else:
            st.info("인구통계 데이터가 준비되지 않았습니다.")

    # --- Tab 5: 심층 분석 ---
    with tabs[4]:
        st.header("다변량 분석 및 세부 속성 탐색")
        
        if special_data['trench_attributes'] is not None and '트렌치코트' in selected_keywords:
            st.subheader("트렌치코트 세부 속성(핏/길이/성별) 트랜드")
            attr_df = special_data['trench_attributes'].copy()
            attr_df['period'] = pd.to_datetime(attr_df['period'])
            fig_attr = px.line(attr_df, x='period', y='ratio', color='keyword', title="트렌치코트 세부 속성별 추이")
            st.plotly_chart(fig_attr, use_container_width=True)
            
            # [그래프 8] 박스 플롯 (가격 비교)
            if shopping_dfs:
                st.subheader("주요 쇼핑몰별 가격 분포")
                full_shop_df = pd.concat(shopping_dfs)
                top_malls = full_shop_df['mallName'].value_counts().head(5).index
                df_malls = full_shop_df[full_shop_df['mallName'].isin(top_malls)]
                fig_box = px.box(df_malls, x='mallName', y='lprice', color='keyword', title="상위 5개 쇼핑몰 가격 분포")
                st.plotly_chart(fig_box, use_container_width=True)
        
        st.subheader("로 데이터(Raw Data) 미리보기")
        for kw in selected_keywords:
            with st.expander(f"'{kw}' 데이터 보기"):
                for dtype, df in main_data[kw].items():
                    st.write(f"[{dtype}]")
                    st.dataframe(df.head(10))

    # --- Tab 6: 런칭 전략 (Deck) ---
    with tabs[5]:
        st.header("트렌치코트 2026 봄 시즌 런칭 전략")
        st.info("데이터 분석 결과에 기반한 핵심 전략 시각화 자료입니다.")
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.image("images/strategy/market_growth_comparison.png", caption="시장 경쟁력: 트렌치코트 성장률 압도적 1위")
        with col_s2:
            st.image("images/strategy/attribute_demand_donut.png", caption="소비자 니즈: 숏 트렌치 및 블랙/네이비 선호")
            
        col_s3, col_s4 = st.columns(2)
        with col_s3:
            st.image("images/strategy/price_distribution_hist.png", caption="가격 전략: 10~13만원대 메인 볼륨 모델 최적")
        with col_s4:
            st.image("images/strategy/age_interest_bar.png", caption="핵심 타겟: 2545 여성 중심 (3040 강력 수요)")

        st.markdown("---")
        if os.path.exists("trench_coat_2026_strategy.md"):
             with open("trench_coat_2026_strategy.md", "r", encoding="utf-8") as f:
                 st.markdown(f.read(), unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.info("이 대시보드는 Naver API 실시간 수집 데이터를 기반으로 구성되었습니다.")
