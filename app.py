import streamlit as st
import pandas as pd
import time

try:
    from model import MovieRecommendationSystem
except ImportError:
    st.error("Error: 'model.py' not found. Please ensure you have copied the code from x1.ipynb and saved it as 'model.py'.")
    st.stop()


DATA_PATH = "data/" 


OCCUPATION_MAP = {
    0: "other", 1: "academic/educator", 2: "artist", 3: "clerk",
    4: "college/grad student", 5: "customer service", 6: "doctor/health care",
    7: "executive/managerial", 8: "farmer", 9: "homemaker", 10: "K-12 student",
    11: "lawyer", 12: "programmer", 13: "retired", 14: "sales/marketing",
    15: "scientist", 16: "self-employed", 17: "technician/engineer",
    18: "tradesman/craftsman", 19: "unemployed", 20: "writer"
}


AGE_MAP = {
    1: "Under 18", 18: "18-24", 25: "25-34",
    35: "35-44", 45: "45-49", 50: "50-55", 56: "56+"
}


@st.cache_resource
def load_recommendation_system(data_path):
    """
    Load and run the complete machine learning pipeline.
    This will be very slow (may take several minutes) but only runs once.
    """
    with st.spinner("Initializing recommendation system, this may take a few minutes..."):
        try:
            
            system = MovieRecommendationSystem(data_path)
        
            
            system.run_complete_pipeline()
            
            return system
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.error(f"Please check that the DATA_PATH variable is correct and that .dat files exist in '{data_path}'.")
            return None


system = load_recommendation_system(DATA_PATH)


if system is None:
    st.stop()


st.title("🎬 Movie Recommendation System")
st.write("This system is designed for new users. Please select the user's basic profile information:")


col1, col2, col3 = st.columns(3)

with col1:
    selected_age_label = st.selectbox(
        "Select Age Group:",
        list(AGE_MAP.values())
    )
    
    selected_age_id = [k for k, v in AGE_MAP.items() if v == selected_age_label][0]


with col2:
    selected_gender = st.selectbox(
        "Select Gender:",
        ["M", "F"]
    )

with col3:
    selected_occ_label = st.selectbox(
        "Select Occupation:",
        list(OCCUPATION_MAP.values())
    )
    
    selected_occ_id = [k for k, v in OCCUPATION_MAP.items() if v == selected_occ_label][0]



if st.button("🚀 Get Recommendations", type="primary"):
    
    
    new_user_info = {
        'gender': selected_gender,
        'age': selected_age_id,
        'occupation': selected_occ_id,
    }

    st.subheader(f"Recommendations for: {selected_age_label}, {selected_gender}, {selected_occ_label}")

    
    with st.spinner("Calculating recommendations..."):
        try:
            
            recommendations = system.recommend_for_new_user(new_user_info)
            
            if not recommendations:
                st.warning("No matching recommendations found.")
            
            
            for i, (movie_idx, pred_rating) in enumerate(recommendations, 1):
                movie_info = system.movies_df.iloc[movie_idx]
                title = movie_info['title']
                genres = movie_info['genres']
                
                st.markdown(f"**{i}. {title}**")
                st.write(f"&nbsp;&nbsp;&nbsp;&nbsp; **Genres**: {genres} | **Predicted Rating**: {pred_rating:.2f}")

        except Exception as e:
            st.error(f"An error occurred during recommendation: {e}")
