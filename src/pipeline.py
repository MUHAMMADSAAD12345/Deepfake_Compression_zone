import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.fer_model import FERModel
from src.models.ser_model import SERModel
from src.utils.preprocessing import preprocess_video
from src.reasoning.intra_modality import intra_modality_reasoning
from src.reasoning.inter_modality import inter_modality_reasoning
from src.reasoning.classifier import (
    compute_fake_probabilities,
    classify_modality,
    voting_classifier,
    generate_explanation,
)

class DeepfakeDetector:
    def __init__(self, fer_weights=None, ser_weights=None, model_type="vgg19"):
        print(f"[Detector] Initializing FER model ({model_type})...")
        self.fer = FERModel(fer_weights, model_type=model_type)
        print("[Detector] Initializing SER model...")
        self.ser = SERModel(ser_weights)
        print("[Detector] Ready.")

    def predict_video(self, video_path):
        print(f"[Detector] Processing video: {video_path}")
        data = preprocess_video(video_path)
        if "error" in data:
            return data
        return self._predict_from_data(data)

    def predict_from_frames_audio(self, face_images, segments, sync_map, frame_ts=None, segment_ts=None):
        """Run KB pipeline on pre-extracted face images and audio segments.
        Skips face detection and frame extraction."""
        if not face_images or not segments:
            return {"error": "No faces or audio provided"}
        
        visual_emotions, _ = self.fer.predict_batch(face_images)
        aural_emotions, _ = self.ser.predict_batch(segments)

        vt_results, vt_fake_idx = intra_modality_reasoning(visual_emotions)
        at_results, at_fake_idx = intra_modality_reasoning(aural_emotions)
        im_results, im_fake_idx = inter_modality_reasoning(
            visual_emotions, aural_emotions, sync_map
        )

        prob_dict = compute_fake_probabilities(vt_results, at_results, im_results)
        modality_results = classify_modality(prob_dict)
        final_prediction = voting_classifier(modality_results)
        explanation = generate_explanation(
            prob_dict, modality_results, vt_fake_idx, at_fake_idx, im_results
        )

        return {
            "prediction": final_prediction,
            "modality_results": modality_results,
            "probabilities": prob_dict,
            "explanation": explanation,
            "visual_emotions": visual_emotions,
            "aural_emotions": aural_emotions,
        }

    def _predict_from_data(self, data):
        face_images = data["face_images"]
        segments = data["audio_segments"]
        sync_map = data["sync_map"]
        return self.predict_from_frames_audio(face_images, segments, sync_map)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Multimodal Neurosymbolic Deepfake Detection")
    parser.add_argument("video", help="Path to input video")
    parser.add_argument("--fer_weights", default=None, help="Path to FER weights")
    parser.add_argument("--ser_weights", default=None, help="Path to SER weights")
    parser.add_argument("--model_type", default="vgg19", choices=["vgg19", "resnet50"], help="FER model type")
    args = parser.parse_args()

    detector = DeepfakeDetector(fer_weights=args.fer_weights, ser_weights=args.ser_weights, model_type=args.model_type)
    result = detector.predict_video(args.video)

    print("\n" + "=" * 60)
    print(result["explanation"])
    print("=" * 60)

if __name__ == "__main__":
    main()
