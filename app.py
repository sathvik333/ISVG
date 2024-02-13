import streamlit as st
from crackHead import get_top_N_images
from crackHead import plot_images_by_side
import tempfile
import os
from crackHead import get_top_N_images, plot_images_by_side, image_data_df
from video_gen import generate_video

st.set_page_config(
    layout="wide"
)

def main():
    st.markdown("<h1 style='text-align: center; color: white;'>Search Anything App</h1>", unsafe_allow_html=True)


    query = st.text_input("Enter your query:")
    if st.button("Search"):
        if len(query) > 0:
            Query = get_top_N_images(query)
            st.warning("Your query was "+query)
            st.subheader("Search Results:")
            for idx in Query.index.values:
                image = Query.iloc[idx].jpg
                caption = Query.iloc[idx].json
                sim_score = Query.iloc[idx].cos_sim
                sim_score = 100*float("{:.2f}".format(sim_score))
                st.image(image, caption=f"Caption: {caption}\nSimilarity: {sim_score}%")
        else:
            st.warning("Developer is not dumb are you?")

    vid_query = st.text_input("Enter your query for video generation:")
    if st.button("Generate Video"):
        if len(vid_query) > 0:
            # Generate the video
            video_path = generate_video(vid_query)
            st.subheader("Generated Video:")
            st.video(video_path)  # Display the video
        else:
            st.warning("Developer is not dumb are you?")
if __name__ == "__main__":
    main()
