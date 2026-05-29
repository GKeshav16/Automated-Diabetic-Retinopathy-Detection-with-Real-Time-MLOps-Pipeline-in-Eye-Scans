#!/usr/bin/env python3
"""
Inference script for Diabetic Retinopathy Detection
"""

import argparse
import logging
import torch
import yaml
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessor import ImagePreprocessor
from src.models.architecture import RetinopathyDetectionModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SEVERITY_LEVELS = {
    0: "No Diabetic Retinopathy",
    1: "Mild Diabetic Retinopathy",
    2: "Moderate Diabetic Retinopathy",
    3: "Severe Diabetic Retinopathy",
    4: "Proliferative Diabetic Retinopathy"
}


class RetinopathyPredictor:
    """Make predictions using trained model"""
    
    def __init__(self, model_path: str, config_path: str = "config.yaml", device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load model
        self.model = RetinopathyDetectionModel(
            model_name=self.config['model']['model_name'],
            num_classes=self.config['model']['num_classes'],
            pretrained=False
        ).to(self.device)
        
        # Load checkpoint
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"✓ Model loaded from {model_path}")
        else:
            logger.warning(f"Model not found at {model_path}")
        
        self.model.eval()
        
        # Initialize preprocessor
        self.preprocessor = ImagePreprocessor(
            image_size=self.config['data']['image_size'],
            normalize=self.config['data']['normalize']
        )
    
    def predict(self, image_path: str) -> dict:
        """Make prediction on a single image"""
        try:
            # Preprocess image
            image = self.preprocessor.preprocess(image_path)
            image_tensor = torch.from_numpy(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                output = self.model(image_tensor)
                probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
                predicted_class = probabilities.argmax()
                confidence = probabilities[predicted_class]
            
            result = {
                'image_path': str(image_path),
                'predicted_grade': int(predicted_class),
                'severity': SEVERITY_LEVELS[int(predicted_class)],
                'confidence': float(confidence),
                'probabilities': {
                    SEVERITY_LEVELS[i]: float(prob)
                    for i, prob in enumerate(probabilities)
                }
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Prediction failed for {image_path}: {str(e)}")
            raise


def main(args):
    """Main inference function"""
    logger.info("=" * 60)
    logger.info("Diabetic Retinopathy Detection - Inference")
    logger.info("=" * 60)
    
    predictor = RetinopathyPredictor(
        model_path=args.model,
        config_path=args.config,
        device=args.device
    )
    
    if args.image:
        # Single image prediction
        image_path = args.image
        logger.info(f"\nPredicting for: {image_path}")
        
        result = predictor.predict(image_path)
        
        logger.info("\n" + "=" * 60)
        logger.info("PREDICTION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Severity: {result['severity']}")
        logger.info(f"Confidence: {result['confidence']:.2%}")
        logger.info("\nProbabilities:")
        for severity, prob in result['probabilities'].items():
            logger.info(f"  {severity}: {prob:.4f}")
        
    elif args.directory:
        # Batch predictions
        image_dir = Path(args.directory)
        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        
        logger.info(f"\nPredicting for {len(image_files)} images...")
        
        results = []
        for image_path in image_files:
            result = predictor.predict(str(image_path))
            results.append(result)
            logger.info(f"  {image_path.name}: {result['severity']} (confidence: {result['confidence']:.2%})")
        
        logger.info(f"\n✓ Processed {len(results)} images")
        
        # Save results
        if args.output:
            import json
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer Diabetic Retinopathy from images")
    parser.add_argument("--model", default="models/checkpoints/best_model.pt", help="Path to model checkpoint")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    parser.add_argument("--image", help="Path to single image for prediction")
    parser.add_argument("--directory", help="Directory of images for batch prediction")
    parser.add_argument("--output", help="Output JSON file for results")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    
    args = parser.parse_args()
    
    if not args.image and not args.directory:
        parser.error("Provide either --image or --directory")
    
    main(args)
