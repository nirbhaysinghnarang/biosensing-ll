from typing import Iterator
import numpy as np



def increase_pcm_volume_iterator(pcm_iterator: Iterator[bytes], factor: float, chunk_size: int = 4800, slow_up:bool = False, slow_up_factor: float = 1.1) -> Iterator[bytes]:
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
        np.clip(audio_float, -max_value, max_value, out=audio_float)        
        audio_increased = audio_float.astype(dtype)

        if slow_up:
            num_samples = len(audio_increased)
            time_original = np.arange(num_samples)
            time_stretched = np.linspace(0, num_samples - 1, int(num_samples * slow_up_factor))
            audio_float = np.interp(time_stretched, time_original, audio_increased.astype(np.float32))
            # Convert back to int16 before returning
            audio_stretched = np.round(audio_float).astype(np.int16)
            return audio_stretched.tobytes()
                
        
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

