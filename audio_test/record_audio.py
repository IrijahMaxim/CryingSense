"""
Audio Recording Module for CryingSense Testing

Records audio from the system default microphone for testing the CNN model.
Supports both real-time streaming and file-based recording.
"""

import os
import wave
import time
import numpy as np
import pyaudio
from datetime import datetime


class AudioRecorder:
    """Handles audio recording from system microphone."""
    
    def __init__(self, sample_rate=16000, chunk_size=1024, channels=1, 
                 audio_format=pyaudio.paInt16):
        """
        Initialize audio recorder.
        
        Args:
            sample_rate: Sample rate in Hz (default: 16000)
            chunk_size: Number of samples per buffer (default: 1024)
            channels: Number of audio channels (default: 1 for mono)
            audio_format: PyAudio format (default: paInt16)
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.audio_format = audio_format
        self.audio = pyaudio.PyAudio()
        
    def list_devices(self):
        """List all available audio input devices."""
        print("\nAvailable Audio Input Devices:")
        print("-" * 70)
        info = self.audio.get_host_api_info_by_index(0)
        num_devices = info.get('deviceCount')
        
        for i in range(num_devices):
            device_info = self.audio.get_device_info_by_host_api_device_index(0, i)
            if device_info.get('maxInputChannels') > 0:
                print(f"  Device {i}: {device_info.get('name')}")
                print(f"    Max Input Channels: {device_info.get('maxInputChannels')}")
                print(f"    Default Sample Rate: {device_info.get('defaultSampleRate')}")
                print()
    
    def record_audio(self, duration=5.0, device_index=None, save_path=None):
        """
        Record audio from microphone.
        
        Args:
            duration: Recording duration in seconds (default: 5.0)
            device_index: Specific device index (None for default)
            save_path: Path to save audio file (None to return array only)
        
        Returns:
            numpy array: Audio data
        """
        print(f"\nRecording for {duration} seconds...")
        print("Speak or make sounds into the microphone...")
        
        # Open stream
        stream = self.audio.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=self.chunk_size
        )
        
        frames = []
        num_chunks = int(self.sample_rate / self.chunk_size * duration)
        
        # Record audio
        for i in range(num_chunks):
            data = stream.read(self.chunk_size, exception_on_overflow=False)
            frames.append(data)
            
            # Progress indicator
            if i % 10 == 0:
                progress = (i + 1) / num_chunks * 100
                print(f"\rRecording progress: {progress:.1f}%", end='', flush=True)
        
        print("\rRecording complete!         ")
        
        # Stop and close stream
        stream.stop_stream()
        stream.close()
        
        # Convert to numpy array
        audio_data = b''.join(frames)
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # Save to file if path provided
        if save_path:
            self._save_wav(frames, save_path)
            print(f"Audio saved to: {save_path}")
        
        return audio_array
    
    def record_continuous(self, segment_duration=5.0, num_segments=None, 
                         output_dir='recordings', device_index=None):
        """
        Record continuous audio in segments.
        
        Args:
            segment_duration: Duration of each segment in seconds
            num_segments: Number of segments to record (None for infinite)
            output_dir: Directory to save recordings
            device_index: Specific device index (None for default)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nStarting continuous recording...")
        print(f"Segment duration: {segment_duration} seconds")
        print(f"Output directory: {output_dir}")
        print("Press Ctrl+C to stop\n")
        
        segment_count = 0
        try:
            while num_segments is None or segment_count < num_segments:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"recording_{timestamp}.wav"
                save_path = os.path.join(output_dir, filename)
                
                print(f"[Segment {segment_count + 1}]")
                self.record_audio(segment_duration, device_index, save_path)
                segment_count += 1
                
                print(f"Saved: {filename}\n")
                time.sleep(0.5)  # Brief pause between segments
                
        except KeyboardInterrupt:
            print("\n\nRecording stopped by user")
        
        print(f"\nTotal segments recorded: {segment_count}")
    
    def _save_wav(self, frames, filepath):
        """Save audio frames to WAV file."""
        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.audio_format))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(frames))
    
    def close(self):
        """Clean up PyAudio resources."""
        self.audio.terminate()


def main():
    """Main recording function with CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Record audio from microphone for CryingSense testing'
    )
    parser.add_argument('--duration', type=float, default=5.0,
                       help='Recording duration in seconds (default: 5.0)')
    parser.add_argument('--output', type=str, default='recordings',
                       help='Output directory for recordings')
    parser.add_argument('--continuous', action='store_true',
                       help='Enable continuous recording mode')
    parser.add_argument('--segments', type=int, default=None,
                       help='Number of segments for continuous mode (default: infinite)')
    parser.add_argument('--list-devices', action='store_true',
                       help='List available audio devices and exit')
    parser.add_argument('--device', type=int, default=None,
                       help='Specific device index to use')
    parser.add_argument('--sample-rate', type=int, default=16000,
                       help='Sample rate in Hz (default: 16000)')
    
    args = parser.parse_args()
    
    # Initialize recorder
    recorder = AudioRecorder(sample_rate=args.sample_rate)
    
    try:
        # List devices if requested
        if args.list_devices:
            recorder.list_devices()
            return
        
        print("="*70)
        print("CryingSense Audio Recorder")
        print("="*70)
        print(f"Sample Rate: {args.sample_rate} Hz")
        print(f"Recording Duration: {args.duration} seconds")
        print(f"Output Directory: {args.output}")
        print("="*70)
        
        # Create output directory
        os.makedirs(args.output, exist_ok=True)
        
        if args.continuous:
            # Continuous recording mode
            recorder.record_continuous(
                segment_duration=args.duration,
                num_segments=args.segments,
                output_dir=args.output,
                device_index=args.device
            )
        else:
            # Single recording mode
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.wav"
            save_path = os.path.join(args.output, filename)
            
            recorder.record_audio(
                duration=args.duration,
                device_index=args.device,
                save_path=save_path
            )
            
            print("\n" + "="*70)
            print("Recording Complete!")
            print("="*70)
            print(f"File saved: {save_path}")
    
    finally:
        recorder.close()


if __name__ == "__main__":
    main()
