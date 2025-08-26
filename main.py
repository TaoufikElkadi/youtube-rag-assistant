# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "langchain",
# ]
# ///

import os
from dotenv import load_dotenv
from youtube_processor import process_youtube_video
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()



def main():
    print("Hello from youtubevideoassistant!")

    youtube_url = input("Enter the YouTube URL: ")
    if not youtube_url:
        print("No URL provided. Exiting...")
        return

    print(f"Processing YouTube video: {youtube_url}")

    try:
        retriever = process_youtube_video(youtube_url=youtube_url)
        print("Video processed successfully.")
    except Exception as e:
        print(f"Error processing video: {e}")
        return

    chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    template = """
You are an expert research assistant specializing in analyzing startup and venture capital content from Y Combinator videos.

You will be given a user query and relevant transcript chunks from Y Combinator videos.
Use ONLY the provided transcript chunks to answer the query. 
If the answer is not found in the provided chunks, say: 
"I couldn’t find this information in the Y Combinator videos I have access to."

Make your answer:
- Concise and clear (2–4 paragraphs max).
- Faithful to the transcript (don’t make things up).
- Include specific names, examples, or advice mentioned in the videos when relevant.

---
**User Query:** {question}

**Relevant Transcript Chunks:**
{context}
"""
    prompt = ChatPromptTemplate.from_template(template)

    chain = prompt | chat

    while True:
        print("-" * 100)
        question = input("Enter your question: ")
        if question.lower() == "q":
            break
        try:
            context = retriever.invoke(question)
            res = chain.invoke({"question": question, "context": context})
            print(res.content)
        except Exception as e:
            print(f"Error processing question: {e}")




if __name__ == "__main__":
    main()
