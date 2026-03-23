"""
Inference script for loading and using trained RPCE models.
"""
import torch
import argparse
from pathlib import Path

from models import AutoEncoder
from transport import predict_cate_rpce
from data import createJobsTensorDataset


def load_trained_model(checkpoint_path, device=None):
    """
    Load a trained RPCE model from checkpoint.
    
    Args:
        checkpoint_path (str): Path to model checkpoint
        device (str, optional): Device to load model on
    
    Returns:
        tuple: (model, checkpoint_dict)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Recreate model architecture
    model = AutoEncoder(
        input_dim=checkpoint['input_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        latent_dim=checkpoint['latent_dim']
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded from: {checkpoint_path}")
    print(f"Architecture: input_dim={checkpoint['input_dim']}, "
          f"hidden_dim={checkpoint['hidden_dim']}, "
          f"latent_dim={checkpoint['latent_dim']}")
    
    if 'metrics' in checkpoint:
        print("\nTraining Metrics:")
        for key, value in checkpoint['metrics'].items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.6f}")
    
    return model, checkpoint


def predict_on_new_data(model, X_new, X_rct_reference, batch_size=256, device=None):
    """
    Make CATE predictions on new data.
    
    Args:
        model (nn.Module): Trained RPCE model
        X_new (torch.Tensor): New covariates to predict on
        X_rct_reference (torch.Tensor): RCT reference distribution
        batch_size (int): Batch size for prediction
        device (str, optional): Device to use
    
    Returns:
        tuple: (cate_predictions, confidence_scores)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = model.to(device).eval()
    X_new = X_new.float()
    X_rct_reference = X_rct_reference.float()
    
    all_cate = []
    all_conf = []
    
    print(f"Predicting CATE for {X_new.shape[0]} samples...")
    
    for start in range(0, X_new.shape[0], batch_size):
        end = start + batch_size
        X_batch = X_new[start:end]
        
        cate_batch, conf_batch = predict_cate_rpce(
            model=model,
            x_obs=X_batch,
            x_rct=X_rct_reference.to(device),
            device=device
        )
        
        all_cate.append(cate_batch.cpu())
        all_conf.append(conf_batch.cpu())
        
        if (start // batch_size + 1) % 10 == 0:
            print(f"  Processed {end}/{X_new.shape[0]} samples...")
    
    cate = torch.cat(all_cate, dim=0)
    confidence = torch.cat(all_conf, dim=0)
    
    print("Prediction complete!")
    print(f"  Mean CATE: {cate.mean():.4f}")
    print(f"  Std CATE: {cate.std():.4f}")
    print(f"  Mean Confidence: {confidence.mean():.4f}")
    
    return cate, confidence


def recommend_treatment(cate_pred, confidence=None, threshold=0.0, confidence_threshold=0.5):
    """
    Recommend treatment based on CATE predictions.
    
    Args:
        cate_pred (torch.Tensor): CATE predictions
        confidence (torch.Tensor, optional): Confidence scores
        threshold (float): Treatment threshold
        confidence_threshold (float): Minimum confidence for recommendation
    
    Returns:
        dict: Treatment recommendations
    """
    cate_pred = cate_pred.cpu().numpy()
    
    # Basic recommendation
    treatment_recommended = (cate_pred > threshold)
    
    recommendations = {
        'treatment': treatment_recommended,
        'cate': cate_pred,
        'n_treat': int(treatment_recommended.sum()),
        'n_control': int((~treatment_recommended).sum()),
        'treatment_rate': float(treatment_recommended.mean())
    }
    
    # Add confidence-based filtering if available
    if confidence is not None:
        confidence = confidence.cpu().numpy()
        high_confidence = (confidence >= confidence_threshold)
        
        recommendations['high_confidence_mask'] = high_confidence
        recommendations['confidence'] = confidence
        recommendations['n_high_confidence'] = int(high_confidence.sum())
        recommendations['high_confidence_rate'] = float(high_confidence.mean())
        
        # Conservative recommendation: only recommend if high confidence
        conservative_treatment = treatment_recommended & high_confidence
        recommendations['conservative_treatment'] = conservative_treatment
        recommendations['conservative_treatment_rate'] = float(conservative_treatment.mean())
    
    return recommendations


def main():
    """Command-line interface for model inference."""
    parser = argparse.ArgumentParser(description='RPCE Model Inference')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to .npz file with new data')
    parser.add_argument('--rct_reference_path', type=str, required=True,
                       help='Path to .npz file with RCT reference data')
    parser.add_argument('--output_path', type=str, default='predictions.pt',
                       help='Path to save predictions')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for inference')
    parser.add_argument('--threshold', type=float, default=0.0,
                       help='Treatment recommendation threshold')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                       help='Minimum confidence for recommendation')
    
    args = parser.parse_args()
    
    # Load model
    print("="*80)
    print("Loading model...")
    print("="*80)
    model, checkpoint = load_trained_model(args.model_path)
    
    # Load data
    print("\n" + "="*80)
    print("Loading data...")
    print("="*80)
    
    # Load new data to predict on
    new_dataset = createJobsTensorDataset(
        args.data_path,
        split_by_e=False,
        return_type="both"
    )
    X_new = new_dataset.tensors[0]
    print(f"New data: {X_new.shape[0]} samples, {X_new.shape[1]} features")
    
    # Load RCT reference
    rct_dataset = createJobsTensorDataset(
        args.rct_reference_path,
        split_by_e=True,
        return_type="rct"
    )
    X_rct_ref = rct_dataset.tensors[0]
    print(f"RCT reference: {X_rct_ref.shape[0]} samples")
    
    # Make predictions
    print("\n" + "="*80)
    print("Making predictions...")
    print("="*80)
    
    cate, confidence = predict_on_new_data(
        model=model,
        X_new=X_new,
        X_rct_reference=X_rct_ref,
        batch_size=args.batch_size
    )
    
    # Generate recommendations
    print("\n" + "="*80)
    print("Generating treatment recommendations...")
    print("="*80)
    
    recommendations = recommend_treatment(
        cate_pred=cate,
        confidence=confidence,
        threshold=args.threshold,
        confidence_threshold=args.confidence_threshold
    )
    
    print(f"\nRecommendation Summary:")
    print(f"  Threshold: {args.threshold}")
    print(f"  Samples recommended for treatment: {recommendations['n_treat']} "
          f"({recommendations['treatment_rate']:.1%})")
    print(f"  Samples recommended for control: {recommendations['n_control']}")
    
    if 'conservative_treatment_rate' in recommendations:
        print(f"\nHigh-Confidence Recommendations:")
        print(f"  Confidence threshold: {args.confidence_threshold}")
        print(f"  High-confidence samples: {recommendations['n_high_confidence']} "
              f"({recommendations['high_confidence_rate']:.1%})")
        print(f"  Conservative treatment rate: {recommendations['conservative_treatment_rate']:.1%}")
    
    # Save predictions
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'cate_predictions': cate,
        'confidence_scores': confidence,
        'recommendations': recommendations,
        'args': vars(args)
    }, output_path)
    
    print(f"\nPredictions saved to: {output_path}")
    print("\n" + "="*80)
    print("Inference complete!")
    print("="*80)


if __name__ == "__main__":
    main()
