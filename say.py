import time  
from openai import OpenAI
import os
from pydantic import BaseModel
from enum import Enum
from elevenlabs.client import ElevenLabs
import random
from typing import Iterator, List
import sounddevice as sd
import csv
import numpy as np
import random
from typing import List
import time
from audio_funcs import play_pcm_24000_audio, increase_pcm_volume_iterator
import asyncio
import os

from dotenv import load_dotenv



easy_ts = []
A_ID=os.getenv("A_ID")
B_ID=os.getenv("B_ID")

hard_ts = []

ELEVEN_KEY = os.getenv("ELEVEN_KEY")
client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

eleven_client = ElevenLabs(
  api_key=ELEVEN_KEY
)


class DifficultyLevel(Enum):
    EASY = "easy"
    HARD = "hard"

class Speaker(Enum):
    a = "a"
    b = "b"

class Dialogue(BaseModel):
    text:str
    speaker:Speaker
        
class Para(BaseModel):
    dialogues:list[Dialogue]
    difficulty:DifficultyLevel
    
class LangResponse(BaseModel):
    paras:list[Para]
    
    
class TopicResponse(BaseModel):
    topics:list[str]
    
    
async def pre_generate_and_play_audio(list_paras: List[Para]):
    random.shuffle(list_paras)
    assert len(list_paras) == 10, "TRY AGAIN!"
    easy_ts = []
    hard_ts = []
    # Pre-generate all audio
    audio_data = []
    for para in list_paras:
        para_audio = []
        for dlg in para.dialogues:
            voice_id = A_ID if dlg.speaker == Speaker.a else B_ID
            audio = eleven_client.generate(
                text=dlg.text,
                voice=voice_id,
                output_format="pcm_24000",
                model="eleven_multilingual_v2",
            )
            factor = 1 if dlg.speaker == Speaker.a else 1.5
            audio = increase_pcm_volume_iterator(audio, factor=factor, slow_up=dlg.speaker == Speaker.a)
            para_audio.append(audio)
        audio_data.append((para, para_audio))
        
        
    with sd.OutputStream(samplerate=24000, channels=1, dtype='int16') as stream:

        # Play audio with 1s delay between paragraphs
        for para, para_audio in audio_data:
            dfclty = para.difficulty
            para_start = time.time()
            print(f"Started playing paragraph (difficulty: {dfclty})")


            for audio in para_audio:
                play_pcm_24000_audio(audio, stream)
                #await asyncio.sleep(0.1)  # Small delay between dialogues

            para_end = time.time()
            print(f"Ended playing paragraph (difficulty: {dfclty})")

            if dfclty == DifficultyLevel.EASY:
                easy_ts.append({"start": para_start, "end": para_end})
            else:
                hard_ts.append({"start": para_start, "end": para_end})

            if para != audio_data[-1][0]:  # If not the last paragraph
                await asyncio.sleep(0.3)  # 1s delay between paragraphs

    print("Finished all paragraphs")
    return easy_ts, hard_ts

def _create_list():
    paras = []
    
    topic_prompt = f"""
    Come up with 10 topics suitable for a 1 minute conversation
    between 2 speakers A and B.
    Example outputs include:
    
    Speaker A inviting Speaker B to dinner.
    Speaker A accusing Speaker B of infidelity.
     
    """
    topic_response =  client.beta.chat.completions.parse(
        model='gpt-4o-mini',
        messages=[
            {'role': 'user', 'content': topic_prompt}
        ],
        temperature=0.5,
        response_format=TopicResponse
    )
    topics = topic_response.choices[0].message.parsed.topics
    print(topics)
    random.shuffle(topics)
    
    
    
    for i in range(10):
        topic = topics[i]
        difficulty = "easy" if i < 5 else "hard"
        hsk_level = "HSK Level 1" if difficulty == "easy" else "HSK Level 4"
        prompt = f"""
        Please generate a structured paragraph in Chinese, consisting of a dialogue between two speakers. The paragraph should be formatted as follows:
        {{
         "para": [
         {{
          "dialogues": [
           {{
            "text": "Chinese text for speaker 1",
            "speaker": "a"
           }},
           {{
            "text": "Chinese text for speaker 2",
            "speaker": "b"
           }},
           // ... more dialogue exchanges
          ],
          "difficulty": "{difficulty}"
         }}
         ]
        }}
        Requirements:
        1. The paragraph should be at {hsk_level}, marked as "difficulty": "{difficulty}". Only use words that are known by students of {hsk_level}
        2. The dialogue should contain about 140 words.
        3. Alternate between speakers a and b for each line of dialogue.
        4. Ensure that the content is appropriate and covers everyday situations or more advanced topics, depending on the difficulty level.
        5. For "easy" difficulty, use commonly used and easily recognizable phrases for beginners or intermediate learners of Chinese.
        6. For "hard" difficulty, use more challenging content typically known by intermediate to advanced learners.

        Please generate one such paragraph, adhering to the specified structure and requirements.
        The topic of this paragraph should be {topic} and the level of fluency should be at {hsk_level}
        """
        response =  client.beta.chat.completions.parse(
            model='gpt-4o-2024-08-06',
            messages=[
                {'role': 'user', 'content': prompt}
            ],
            temperature=0.9,
            response_format=Para
        )
        paras.append(response.choices[0].message.parsed)
    return paras

def save_timestamps_to_csv(easy_ts, hard_ts, filename='paragraph_timestamps.csv'):
    # Combine easy and hard timestamps with labels
    all_timestamps = [('Easy', ts) for ts in easy_ts] + [('Hard', ts) for ts in hard_ts]
    
    # Sort all timestamps by start time
    all_timestamps.sort(key=lambda x: x[1]['start'])

    # Open the CSV file for writing
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write the header
        writer.writerow(['Difficulty', 'Start Time (Unix)', 'End Time (Unix)', 'Duration (seconds)'])
        
        # Write the data
        for difficulty, ts in all_timestamps:
            start_time = ts['start']
            end_time = ts['end']
            duration = end_time - start_time
            writer.writerow([difficulty, start_time, end_time, f"{duration:.2f}"])

    print(f"Timestamp data has been saved to {filename}")

async def main():
    easy_timestamps, hard_timestamps = await pre_generate_and_play_audio(_create_list())
    save_timestamps_to_csv(easy_ts=easy_timestamps, hard_ts=hard_timestamps)

    print("Easy timestamps:", easy_timestamps)
    print("Hard timestamps:", hard_timestamps)
      
asyncio.run(main())
