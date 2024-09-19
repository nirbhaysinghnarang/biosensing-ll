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



easy_ts = []
hard_ts = []


ELEVEN_KEY = "sk_33dc4fbdf721579ab25f14cef5d98fc7a6398d80d5156e15"
client = OpenAI(api_key="sk-proj-PhhD323Mh5ND26TTfclbT3BlbkFJmqGD7QRJuSl4x0BUiEEN")

eleven_client = ElevenLabs(
  api_key=ELEVEN_KEY# Defaults to ELEVEN_API_KEY
)

A_ID="VGcvPRjFP4qKhICQHO7d"
B_ID="hkfHEbBvdQFNX4uWHqRF"

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
    
import numpy as np

def increase_pcm_volume_iterator(pcm_iterator: Iterator[bytes], factor: float, chunk_size: int = 4800) -> Iterator[bytes]:
    dtype = np.dtype('<i2')  # S16LE: Signed 16-bit Little Endian
    bytes_per_sample = 2
    max_value = np.iinfo(np.int16).max

    def process_chunk(chunk: bytes) -> bytes:
        if not chunk:
            print("Received empty chunk")
            return b''
        
        samples = len(chunk) // bytes_per_sample
        if samples == 0:
            print(f"Chunk too small: {len(chunk)} bytes")
            return chunk  
        
        
        aligned_chunk = chunk[:samples * bytes_per_sample]        
        audio_array = np.frombuffer(aligned_chunk, dtype=dtype)        
        audio_float = audio_array.astype(np.float32)        
        audio_float *= factor     
        print("Multiplied by factor")   
        np.clip(audio_float, -max_value, max_value, out=audio_float)        
        audio_increased = audio_float.astype(dtype)

        return audio_increased.tobytes()

    # Process chunks
    buffer = b''
    for chunk in pcm_iterator:
        buffer += chunk
        while len(buffer) >= chunk_size:
            yield process_chunk(buffer[:chunk_size])
            buffer = buffer[chunk_size:]
    
    # Process any remaining data
    if buffer:
        yield process_chunk(buffer)

def play_pcm_24000_audio(audio_iterator: Iterator[bytes],stream):
        print("Starting playback. Press Ctrl+C to stop.")
        try:
            for chunk in audio_iterator:
                audio_chunk = np.frombuffer(chunk, dtype=np.int16)
                stream.write(audio_chunk)
            #stream.write(np.zeros(int(stream.samplerate * 0.5), dtype=np.int16))  

        except KeyboardInterrupt:
            print("Playback stopped by user")
        except Exception as e:
            print(f"An error occurred: {e}")
            raise e

import random
from typing import List
import time
import asyncio

async def pre_generate_and_play_audio(list_paras: List[Para]):
    random.shuffle(list_paras)
    assert len(list_paras) == 6, "TRY AGAIN!"

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
            factor = 1 if dlg.speaker == Speaker.a else 2
            audio = increase_pcm_volume_iterator(audio, factor=factor)
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
    prompt = """
    Please generate a structured array of 6 paragraphs in Chinese, each consisting of a dialogue between two speakers. The array should be formatted as follows:
    {
     "para": [
     {
     "dialogues": [
     {
     "text": "Chinese text for speaker 1",
     "speaker": "a"
     },
     {
     "text": "Chinese text for speaker 2",
     "speaker": "b"
     },
     // ... more dialogue exchanges
     ],
     "difficulty": "easy"
     },
     // ... more paragraphs
     ]
    }
    Requirements:
    1. 3 paragraphs should be at HSK Level 1, marked as "difficulty": "easy". These should be commonly used and easily recognizable by beginners or intermediate learners of Chinese.
    2. 3 paragraphs should be at HSK Level 3, marked as "difficulty": "hard". These should be more challenging and typically known by intermediate to advanced learners.
    3. Each paragraph (dialogue exchange) should contain enough text to result in approximately 2 minutes of audio when spoken.
    4. Alternate between speakers a and b for each line of dialogue.
    5. Ensure that the content is appropriate and covers a variety of everyday situations and more advanced topics.
    Here are two examples to illustrate the expected format and content:
    Example 1 (HSK Level 1, Easy):
    {
     "para": [
     {
     "dialogues": [
     {
     "text": "王老师，好久不见！",
     "speaker": "a"
     },
     {
     "text": "好久不见！你也来这家饭店吃饭？",
     "speaker": "b"
     },
     {
     "text": "对，我和我的家人来吃饭。我的朋友说这里的广东菜很好吃。",
     "speaker": "a"
     },
     {
     "text": "你的爸爸妈妈也来了吗？",
     "speaker": "b"
     },
     {
     "text": "没有，他们喜欢在家吃。我的孩子们来了。那是我的老公，小孙；那个是我的大儿子，小宝，今年7岁了；这是我的小女儿，小贝，今年5岁了。",
     "speaker": "a"
     },
     {
     "text": "你们一家四口人真幸福！孩子们都上学了吗？",
     "speaker": "b"
     },
     {
     "text": "大儿子去年上小学了，小女儿明年上小学。",
     "speaker": "a"
     },
     {
     "text": "你老公也会说汉语吗？",
     "speaker": "b"
     },
     {
     "text": "哈哈，当然，他是中国人！",
     "speaker": "a"
     }
     ],
     "difficulty": "easy"
     }
     ]
    }
    Example 2 (HSK Level 3, Hard):
    {
     "para": [
     {
     "dialogues": [
     {
     "text": "你好，请问最近的公交车站在哪里？",
     "speaker": "a"
     },
     {
     "text": "最近的公交车站在友谊大厦。你往前走四百米，过马路，然后右转，再走三十米就能看到车站了。",
     "speaker": "b"
     },
     {
     "text": "那儿有车去豫园吗？",
     "speaker": "a"
     },
     {
     "text": "有的，你可以坐34路车直达，或者坐311路车到北京路，再转72路车。",
     "speaker": "b"
     },
     {
     "text": "要坐几个站？",
     "speaker": "a"
     },
     {
     "text": "34路要坐十个站，311路转72路一共要坐十三个站。",
     "speaker": "b"
     },
     {
     "text": "那等车大概要多久？",
     "speaker": "a"
     },
     {
     "text": "我用手机给你查查。34路还有45分钟到友谊大厦，311路要等一个小时。",
     "speaker": "b"
     },
     {
     "text": "都还要等很久呢。我坐地铁可以到豫园吗？",
     "speaker": "a"
     },
     {
     "text": "可以。你坐地铁二号线到体育馆站，再转十号线，到豫园站下。三号口出去走700米就到了。",
     "speaker": "b"
     },
     {
     "text": "地铁站离这里远不远？",
     "speaker": "a"
     },
     {
     "text": "不远，十分钟就走到了。",
     "speaker": "b"
     },
     {
     "text": "地铁站在哪儿？",
     "speaker": "a"
     },
     {
     "text": "往前走，但是不过马路，在前面的路口左拐。",
     "speaker": "b"
     },
     {
     "text": "谢谢你！",
     "speaker": "a"
     },
     {
     "text": "不客气。",
     "speaker": "b"
     }
     ],
     "difficulty": "hard"
     }
     ]
    }
    Please generate the array with 6 such paragraphs, adhering to the specified structure and requirements.
    """
    response =  client.beta.chat.completions.parse(
        model='gpt-4o-mini',
        messages=[
            {'role': 'user', 'content': prompt}
        ],
        temperature=0,
        response_format=LangResponse
    )
    return response.choices[0].message.parsed.paras

pre_generate_and_play_audio(_create_list())

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
