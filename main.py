import os
from dotenv import find_dotenv, load_dotenv
import streamlit as st
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain
from gtts import gTTS

# Загрузка переменных окружения
load_dotenv(find_dotenv())

def img2text(path_to_image):
    # Модель для извлечения описания изображения
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text(path_to_image)[0]["generated_text"]
    return text

def text2story(scenario, genre, tone, temperature=0.8, max_length=64):
    template = """
You are a creative and talented storyteller. I will provide a scenario derived from an image. 
Please produce only the final story text. The story should be no more than 50 words, 
and should not include the instructions or the prompt itself. Only print the final story.

Genre: {genre} 
Tone: {tone} 
Context: {scenario}

STORY:
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["scenario", "genre", "tone"]
    )

    story_llm = LLMChain(
        prompt=prompt,
        llm=HuggingFaceHub(
            repo_id="HuggingFaceH4/zephyr-7b-beta",
            huggingfacehub_api_token="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",
            model_kwargs={"temperature": temperature, "max_length": max_length}
        ),
        verbose=False
    )

    full_response = story_llm.predict(scenario=scenario, genre=genre, tone=tone)
    lines = full_response.split('\n')
    # Ищем строку STORY:
    start_index = 0
    for i, line in enumerate(lines):
        if "STORY:" in line:
            start_index = i + 1
            break
    # Берём текст только после STORY:
    clean_story = " ".join(lines[start_index:]).strip()
    return clean_story

def story_to_audio(story):
    # Генерируем аудиофайл из текста
    tts = gTTS(story, lang='en')
    audio_file = "output_story.wav"
    tts.save(audio_file)
    with open(audio_file, "rb") as f:
        audio_data = f.read()
    return audio_data

def main():
    st.set_page_config(page_title="Сreate your story", page_icon="✨")
    st.header("Сreate your story")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    genre = st.text_input("Enter the genre (e.g. fantasy, mystery, sci-fi):", "fantasy")
    tone = st.text_input("Enter the tone (e.g. cheerful, dark, whimsical):", "cheerful")

    temperature = st.slider("Temperature:", 0.0, 1.5, 0.8, 0.1)
    max_length = st.slider("Max Length (tokens):", 16, 128, 64, 1)

    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)

        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

        scenario = img2text(uploaded_file.name)

        story = text2story(scenario, genre, tone, temperature=temperature, max_length=max_length)

        with st.expander("Scenario"):
            st.write(scenario)
        with st.expander("Story"):
            st.write(story)

        if st.button("Play Story Audio"):
            audio_data = story_to_audio(story)
            st.audio(audio_data, format="audio/wav")

if __name__ == '__main__':
    main()
