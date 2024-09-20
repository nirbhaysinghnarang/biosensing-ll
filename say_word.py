import time  # for measuring time duration of API calls
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
import asyncio

from audio_funcs import increase_pcm_volume_iterator, play_pcm_24000_audio




easy_ts = []
hard_ts = []


ELEVEN_KEY = "sk_d468da565c6e9130fe61a1874b0709f64245a06a28319377"

eleven_client = ElevenLabs(
  api_key=ELEVEN_KEY# Defaults to ELEVEN_API_KEY
)

A_ID="VGcvPRjFP4qKhICQHO7d"
B_ID="hkfHEbBvdQFNX4uWHqRF"

class DifficultyLevel(Enum):
    EASY = "easy"
    HARD = "hard"


        
class Word(BaseModel):
    word:str
    diff:DifficultyLevel
    
class LangResponse(BaseModel):
    words:list[Word]
    




def _create_list(total=20, easy=10, hard=10):
    
    
    prompt = f"""
    Please generate a structured list of Chinese words in the following format:



    Requirements:
    1. Generate a total of {{total}} Chinese words.
    2. {{easy}} words should be at HSK Level 1, marked with diff="easy". These should be commonly used and easily recognizable by beginners or intermediate learners of Chinese.
    3. {{hard}} words should be at HSK Level 3, marked with diff="hard. These should be more challenging and typically known by intermediate to advanced learners.
    4. Ensure that the words cover a variety of everyday vocabulary and more advanced topics.

    Here's an example of how the response should look:

    words = [
        Word(word="你好", diff="easy"),
        Word(word="再见", diff="easy"),
        Word(word="谢谢", diff="easy"),
        # ... more easy words
        Word(word="建议", diff="hard"),
        Word(word="经验", diff="hard"),
        Word(word="幸福", diff="hard"),
        # ... more hard words
    ]

    Please generate the list of {{total}} words ({{easy}} easy, {{hard}} hard) adhering to the specified structure and requirements.
    """
    client = OpenAI(api_key="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")

    response =  client.beta.chat.completions.parse(
        model='gpt-4o-mini',
        messages=[
            {'role': 'user', 'content': prompt.format(total=total, hard=hard, easy=easy)}
        ],
        temperature=0,
        response_format=LangResponse
    )
    return response.choices[0].message.parsed.words


def save_timestamps_to_csv(easy_ts, hard_ts, filename=f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_words.csv"):
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

async def pre_generate_and_play_audio(words: List[Word]):
    random.shuffle(words)
    print(words)
    assert len(words) == 20, "TRY AGAIN!"
    easy_ts = []
    hard_ts = []
    # Pre-generate all audio
    audio_data = []
    
    for word in words:
        voice_id = A_ID if word.diff == DifficultyLevel.EASY else B_ID
        audio =  eleven_client.generate(
            text=word.word,
            voice=voice_id,
            output_format="pcm_24000",
            model="eleven_multilingual_v2",
        )
        factor = 1.5 if word.diff == DifficultyLevel.EASY else 1
        audio = increase_pcm_volume_iterator(audio, factor=factor, slow_up=word.diff == DifficultyLevel.EASY)
        audio_data.append((word,audio))
        
        
    with sd.OutputStream(samplerate=24000, channels=1, dtype='int16') as stream:

        
        
        for word, word_audio in audio_data:
            dfclty = word.diff
            word_start = time.time()
            print(f"Started playing paragraph (difficulty: {dfclty})")
            play_pcm_24000_audio(word_audio, stream)
            word_end = time.time()
            print(f"Ended playing paragraph (difficulty: {dfclty})") 
            if dfclty == DifficultyLevel.EASY:
                easy_ts.append({"start": word_start, "end": word_end})
            else:
                hard_ts.append({"start": word_start, "end": word_end})

      
    print("Finished all wods")
    return easy_ts, hard_ts




async def main():
    easy_timestamps, hard_timestamps = await pre_generate_and_play_audio(_create_list())
    save_timestamps_to_csv(easy_ts=easy_timestamps, hard_ts=hard_timestamps)

    print("Easy timestamps:", easy_timestamps)
    print("Hard timestamps:", hard_timestamps)
    
    
asyncio.run(main())
