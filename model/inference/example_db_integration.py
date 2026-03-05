"""
Example: Database Integration for CryingSense

This script demonstrates how to use the database integration feature
to automatically save cry classifications, audio sessions, and audio files.

Run this script after:
1. Setting up MongoDB (see database/README.md)
2. Installing dependencies: pip install -r database/requirements.txt
3. Training or downloading a model

Usage:
    python example_db_integration.py --audio test.wav --device-id ESP32-001
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from model.inference.predict import CryingSensePredictor


def example_single_prediction():
    """Example 1: Single prediction with database storage."""
    
    print("\n" + "="*70)
    print("Example 1: Single Prediction with Database")
    print("="*70)
    
    predictor = CryingSensePredictor(
        model_path='../saved_models/cryingsense_cnn_best.pth',
        save_to_db=True,
        device_id='ESP32-001',
        device_source='esp32'
    )
    
    # Perform inference
    result = predictor.predict_single('test_audio.wav')
    
    print(f"\nResults:")
    print(f"  Prediction: {result['prediction']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Is Cry: {result['is_cry']}")
    
    if result.get('classification_id'):
        print(f"  ✓ Saved to database")
        print(f"    Classification ID: {result['classification_id']}")
        print(f"    Audio File ID: {result.get('audio_file_id', 'N/A')}")
    

def example_session_monitoring():
    """Example 2: Session-based monitoring with multiple predictions."""
    
    print("\n" + "="*70)
    print("Example 2: Session-Based Monitoring")
    print("="*70)
    
    predictor = CryingSensePredictor(
        model_path='../saved_models/cryingsense_cnn_best.pth',
        save_to_db=True,
        device_id='ESP32-001',
        device_source='esp32'
    )
    
    # Start a monitoring session
    session_id = predictor.start_new_session()
    print(f"\n✓ Session started: {session_id}")
    
    # Simulate processing multiple audio files
    audio_files = ['cry1.wav', 'cry2.wav', 'cry3.wav']
    cry_count = 0
    
    print(f"\nProcessing {len(audio_files)} audio files...")
    for audio_file in audio_files:
        if os.path.exists(audio_file):
            result = predictor.predict_single(audio_file)
            
            if result['is_cry']:
                cry_count += 1
                print(f"  ✓ {audio_file}: {result['prediction']} ({result['confidence']:.2%})")
            else:
                print(f"  - {audio_file}: No cry detected")
        else:
            print(f"  ✗ {audio_file}: File not found")
    
    # End session
    predictor.end_current_session()
    print(f"\n✓ Session ended. {cry_count} cries detected and saved.")


def example_selective_saving():
    """Example 3: Selective database saving based on conditions."""
    
    print("\n" + "="*70)
    print("Example 3: Selective Database Saving")
    print("="*70)
    
    predictor = CryingSensePredictor(
        model_path='../saved_models/cryingsense_cnn_best.pth',
        save_to_db=False,  # Disabled by default
        device_id='ESP32-001',
        device_source='esp32',
        confidence_threshold=0.80  # Only alert on high confidence
    )
    
    # Process audio files
    audio_files = ['test1.wav', 'test2.wav', 'important.wav']
    
    print("\nProcessing with selective saving...")
    for audio_file in audio_files:
        if os.path.exists(audio_file):
            result = predictor.predict_single(audio_file)
            
            # Save only high-confidence detections
            if result['alert']:
                print(f"  ⚠️  HIGH CONFIDENCE: {audio_file}")
                print(f"     {result['prediction']} ({result['confidence']:.2%})")
                
                # Save this one to database
                predictor.predict_single(audio_file, save_to_db=True)
                print(f"     ✓ Saved to database")
            else:
                print(f"  - {audio_file}: {result['prediction']} ({result['confidence']:.2%}) - not saved")


def example_query_database():
    """Example 4: Query stored data from database."""
    
    print("\n" + "="*70)
    print("Example 4: Query Stored Classifications")
    print("="*70)
    
    try:
        from database.services import CryClassificationService
        
        service = CryClassificationService()
        
        # Get recent classifications
        recent = service.get_recent_classifications(limit=10, device_id='ESP32-001')
        
        print(f"\nRecent classifications (showing {len(recent)}):")
        for i, record in enumerate(recent, 1):
            classification = record.get('classification', {})
            print(f"  {i}. {classification.get('predicted_class')} "
                  f"({classification.get('confidence_score', 0):.2%}) - "
                  f"{record.get('timestamp')}")
        
        # Get statistics
        stats = service.get_statistics(device_id='ESP32-001', days=7)
        print(f"\nStatistics (last 7 days):")
        print(f"  Total classifications: {stats.get('total_count', 0)}")
        print(f"  Average confidence: {stats.get('average_confidence', 0):.2%}")
        
        if 'class_distribution' in stats:
            print(f"  Class distribution:")
            for cry_class, count in stats['class_distribution'].items():
                print(f"    - {cry_class}: {count}")
    
    except Exception as e:
        print(f"Error querying database: {e}")
        print("Make sure MongoDB is running and database is configured.")


def main():
    """Main function with command-line interface."""
    
    parser = argparse.ArgumentParser(
        description='CryingSense Database Integration Examples',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--audio', type=str, help='Path to audio file for testing')
    parser.add_argument('--device-id', type=str, default='ESP32-001',
                       help='Device identifier (default: ESP32-001)')
    parser.add_argument('--example', type=int, choices=[1, 2, 3, 4],
                       help='Run specific example (1-4)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("CryingSense Database Integration Examples")
    print("="*70)
    
    # Check if model exists
    model_path = Path(__file__).parent.parent / 'saved_models' / 'cryingsense_cnn_best.pth'
    if not model_path.exists():
        print(f"\n⚠️  Warning: Model not found at {model_path}")
        print("Please train a model or update the path in the examples.")
        return
    
    # Run examples
    if args.example == 1 or not args.example:
        example_single_prediction()
    
    if args.example == 2 or not args.example:
        example_session_monitoring()
    
    if args.example == 3 or not args.example:
        example_selective_saving()
    
    if args.example == 4 or not args.example:
        example_query_database()
    
    print("\n" + "="*70)
    print("Examples complete!")
    print("="*70)


if __name__ == "__main__":
    main()
