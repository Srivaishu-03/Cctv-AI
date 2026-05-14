import streamlit as st
from visual import search_text, search_image
from agent import ai_agent

st.title("🔍 CCTV Missing Person Search")

st.write(
    "Search CCTV using text or image"
)

# SEARCH MODE

mode = st.radio(
    "Select Search Mode",
    ["Text", "Image"]
)

# TEXT SEARCH

if mode == "Text":

    query = st.text_input(
        "Enter text query"
    )
    
    if st.button("Search"):

        results = ai_agent(query)

        if len(results) == 0:

            st.warning("No matching objects found")

        else:

            st.success("Top Matches")

            for image, score in results:

                st.image(image,
                       caption=f"Score: {round(score, 2)}")

# IMAGE SEARCH

elif mode == "Image":

    image_file = st.file_uploader(
        "Upload Missing Person Image",
        type=["jpg", "png", "jpeg"]
    )

    if image_file is not None:

        from PIL import Image

        query_image = Image.open(
            image_file
        ).convert("RGB")

        st.image(
            query_image,
            caption="Query Image"
        )

        if st.button("Search"):

            results = search_image(
                query_image
            )

            st.success("Top Matches")

            for image, score in results:

                st.image(
                    image,
                    caption=f"Score: {round(score, 2)}"
                )