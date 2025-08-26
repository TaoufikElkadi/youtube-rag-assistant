from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
import re

def extract_video_id(youtube_url):
    patterns = [
        r'(?:youtube\.com/watch\?v=)([^&\n?#]+)',
        r'(?:youtu\.be/)([^&\n?#]+)',
        r'(?:youtube\.com/embed/)([^&\n?#]+)',
        r'(?:youtube\.com/v/)([^&\n?#]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, youtube_url)
        if match:
            return match.group(1)
    
    raise ValueError("Invalid YouTube URL format")


def get_video_transcript(video_id, language="en"):
    try:
        api = YouTubeTranscriptApi()
        transcript_data = api.fetch(video_id, languages=[language])
        transcript = transcript_data.snippets
    except:
        try:
            api = YouTubeTranscriptApi()
            transcript_data = api.fetch(video_id)
            transcript = transcript_data.snippets
        except Exception as e:
            raise Exception(f"Could not retrieve transcript: {str(e)}")
    
    full_text = " ".join([item.text for item in transcript])
    return full_text

def process_youtube_video(youtube_url, language="en"):
    video_id = extract_video_id(youtube_url)

    transcript_text = get_video_transcript(video_id, language)

    doc = Document(
        page_content=transcript_text,
        metadata={"source": youtube_url,
                  "video_id": video_id,
                  "language":language}
    )

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)

    docs = text_splitter.split_documents([doc])

    embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    db_location = "./chroma_db"

    vector_store = Chroma(persist_directory=db_location, embedding_function=embeddings)
    vector_store.add_documents(docs)

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    return retriever








