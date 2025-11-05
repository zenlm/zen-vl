#!/usr/bin/env python3
"""
Upload trained Zen VL models to HuggingFace
"""
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder

def upload_model(model_path, repo_id, commit_message="Upload trained model"):
    """Upload a model directory to HuggingFace"""
    
    print(f"\n{'='*60}")
    print(f"Uploading {repo_id}")
    print(f"From: {model_path}")
    print(f"{'='*60}\n")
    
    if not Path(model_path).exists():
        print(f"❌ Model path does not exist: {model_path}")
        return False
    
    try:
        # Create repo if it doesn't exist
        api = HfApi()
        create_repo(repo_id, repo_type="model", exist_ok=True)
        
        # Upload all files
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message
        )
        
        print(f"✅ Successfully uploaded {repo_id}")
        print(f"📍 https://huggingface.co/{repo_id}\n")
        return True
        
    except Exception as e:
        print(f"❌ Failed to upload {repo_id}: {e}")
        return False

def main():
    """Upload all trained models"""
    
    print("="*60)
    print("Zen VL Model Upload to HuggingFace")
    print("="*60)
    
    # Model paths
    base_dir = Path("/Users/z/work/zen/zen-vl")
    
    models = [
        (base_dir / "instruct/finetuned", "zenlm/zen-vl-4b-instruct", "Upload zen-vl-4b-instruct trained model"),
        (base_dir / "agent/finetuned", "zenlm/zen-vl-4b-agent", "Upload zen-vl-4b-agent trained model"),
    ]
    
    uploaded = []
    failed = []
    
    for model_path, repo_id, message in models:
        if model_path.exists():
            success = upload_model(str(model_path), repo_id, message)
            if success:
                uploaded.append(repo_id)
            else:
                failed.append(repo_id)
        else:
            print(f"⏭️  Skipping {repo_id} - model not found at {model_path}")
            failed.append(repo_id)
    
    # Summary
    print("\n" + "="*60)
    print("Upload Summary")
    print("="*60)
    
    if uploaded:
        print(f"\n✅ Successfully uploaded {len(uploaded)} models:")
        for repo in uploaded:
            print(f"   - https://huggingface.co/{repo}")
    
    if failed:
        print(f"\n❌ Failed/Skipped {len(failed)} models:")
        for repo in failed:
            print(f"   - {repo}")
    
    print(f"\n🎉 Upload complete!")

if __name__ == "__main__":
    main()
