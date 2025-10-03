import os
import streamlit as st
from dotenv import load_dotenv
from langchain.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import requests
from PIL import Image
from io import BytesIO
import re
from langchain.embeddings import OpenAIEmbeddings

ROOT_DIR = os.path.dirname(__name__)
DEVICE = "cpu"
st.set_page_config(
    page_title="Семантический поиск сериалов",
    layout="wide",  # ключевой параметр для широкой страницы
    initial_sidebar_state="expanded",  # или "collapsed", если нужно
)


def format_docs(docs):
    """Форматирует документы для передачи в промпт"""
    formatted = []

    for i, doc in enumerate(docs, 1):
        metadata = doc.metadata

        show_info = f"""
        === Сериал {i} ===
        Название: {metadata.get('title', 'Не указано')}
        Номер в рейтинге: {metadata.get('position_in_rating', 'Не указано')}
        Жанры: {metadata.get('genres', 'Не указано')}
        Год старта: {metadata.get('start_date', 'Не указано')}
        Год окончания: {metadata.get('end_date', 'Не указано')}
        Страна: {metadata.get('country', 'Не указано')}
        Длительность эпизода: {metadata.get('episode_duration_minutes', 'Не указано')}
        Эпизоды: {metadata.get('episodes', 'Не указано')}
        Ссылка: {metadata.get('url', 'Не указано')}
        Ссылка на постер: {metadata.get('poster_url', 'Не указано')}

        Описание: {doc.page_content[:300]}...
        """

        formatted.append(show_info)

    return "\n".join(formatted)


# ==== 1. Загружаем переменные окружения ====
QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API = st.secrets["QDRANT_API"]
GROQ_API = st.secrets["GROQ_API"]

# load_dotenv()
# QDRANT_API = os.getenv("QDRANT_API")
# QDRANT_URL = os.getenv("QDRANT_URL")
# GROQ_API = os.getenv("GROQ_API")

# ==== 2. Подключение к Qdrant Cloud ====
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API)

# ==== 3. Подключаем коллекцию через LangChain ====



model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
model_kwargs = {"device": DEVICE}
encode_kwargs = {"normalize_embeddings": True, "batch_size": 128}

embeddings_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)

vector_store = QdrantVectorStore(
    client=client, collection_name="Shows", embedding=embeddings_model
)

# ==== 4. Создаем RetrievalQA цепочку ====

os.environ["GROQ_API_KEY"] = GROQ_API

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=1.2, max_tokens=2000)

rag_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Ты веселый и душевный эксперт по сериалам.
    Твоя задача - проанализировать представленные сериалы и показать пользователю 5 наиболее подходящих из них с кратким комментарием. Сначала поприветсвуй пользователя. Отдавай предпочтение сериалам, которые выше в рейтинге. Комментарий должен содержать основную тему сериала, рейтинг сериала, количество серий, длину 1 серии.
    Если в данных указана длина серии 0, то пропускай. Если позиция в рейтинге больше 2000 нужно иронично намекнуть, что это может быть не очень хороший сериал. Также нужно показать ссылку на старницу сериала, которая находится в поле "ссылка". Заключительный комментарий должен быть о предложенных сериалах. Можно включить 1-2 сериала, которые не оказались в выдаче, но также могут подходить под сериал.    
    Помни: любой юмор должен быть добрым и не оскорбительным. Цель - сделать анализ интересным!

    Если среди предложенных сериалов есть что-то неподходящее под запрос - обязательно это отметь! 😄

    Стиль анализа:
    - Если рекомендуемый сериал широко известен, отмечай это
    - Если сериал или его персонажи связана с какими-либо мемами, шутками, словами песен и т.д. используй их в комментарии (например, песня "Ведьмаку заплатите чеканной монетой" из сериала Ведьмак)
    - Подмечай ключевые особенности сериалов
    - Структурируй ответ с эмодзи и комментариями
    - Отвечай на русском языке живым, но профессиональным тоном
    - Если сериал является мультфильмом или аниме надо это указать
    - Обязательно выводи 5 вариантов из полученной подборки
    - Проверяй сгенерированный текст на посторонние символы. Должны быть только русские буквы и слова
    Мне нужно, чтобы ты вернул ответ в cледующем формате:
    Блок перед списком | Текст про Сериал 1 | Текст про Сериал 2 | Текст про Сериал 3 | Текст про Сериал 4 | Текст про Сериал 5 | Заключительный блок

    Это нужно чтобы я потом смог превратить его в список при помощи команды split(' | ')
    
    Форматирование текста о сериалах:
    - 1 строка: Название сериала жирным, год начала в скобках не жирным
    - 2 строка: жанры
    - С новой строки комментарий по конкретному сериалу. Комментарий должен отражать основную тему и завязку сюжета сериала
    - С новой строки ссылка на сериал, если ссылки нет, не выводи
    - с новой строки Позиция в рейтинге
    - с новой строки длительность серии и количество серий в формате x серий по y минут
    """,
        ),
        (
            "human",
            """📊 ДАННЫЕ ДЛЯ АНАЛИЗА:
{context}

🎯 ЗАПРОС НА ЭКСПЕРТИЗУ: {question}""",
        ),
    ]
)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Создаем RAG цепочку
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }  # словарь, в котором ключи - это переменные, которые будут переданы в промпт
    | rag_prompt  # промпт для RAG
    | llm  # тут можно поставить любую llm-модель
    | StrOutputParser()  # для вывода ответа в читаемом виде
)

# ==== 5. Streamlit интерфейс ====
st.title("🎬 Семантический поиск сериалов")

query = st.text_input("Введите ваш запрос:")


def get_images(query, names):
    search_answer = vector_store.similarity_search_with_score(query=query, k=10)
    name_poster = {}
    for i, doc in enumerate(search_answer, 1):
        metadata_2 = doc[0].metadata
        name_poster[f"{metadata_2.get('title', i)}"] = metadata_2.get(
            "poster_url", "Not found"
        )
    result = {}

    for name in names:
        response = requests.get(name_poster[name])
        image = Image.open(BytesIO(response.content))
        result[name] = image
    return result


if query:
    with st.spinner("Ищу информацию..."):
        try:
            answer = rag_chain.invoke(query)
        except Exception as e:
            answer = f"❌ Ошибка: {e}"
    st.markdown("**Результат:**")
    # st.write(answer)
    split_text = answer.split("|")
    split_text = [c.strip() for c in split_text]
    names = []
    for i in range(1, min(6, len(split_text))):
        text = split_text[i]
        match = re.search(r"\*\*(.+?)\*\*", text)
        if match:
            title = match.group(1)
        names.append(title)
    images = get_images(query, names)
    text_0 = split_text[0]
    st.markdown(f"<p style='font-size:18px;'>{text_0}</p>", unsafe_allow_html=True)

    for i in range(1, min(6, len(split_text))):  # выводим максимум 5 блоков
        col1, col2 = st.columns([1, 3])  # 1/4 и 3/4
        with col1:
            st.image(images[names[i - 1]], use_container_width=True)
        with col2:
            # Формируем кликабельную ссылку с увеличенным шрифтом
            text = split_text[i]
            st.write(text)
        st.markdown("---")  # разделитель

    # Выводим оставшийся текст, если он есть
    if len(split_text) > 6:
        st.markdown(
            f"<p style='font-size:18px;'>{split_text[6]}</p>", unsafe_allow_html=True
        )
