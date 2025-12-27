#!/usr/bin/env python3
"""
Quick verification script to check if a model is compatible with export_qkv_gqa.py

Usage:
    python verify_model_compatibility.py <model-name>
    
Examples:
    python verify_model_compatibility.py google/gemma-2-2b-it
    python verify_model_compatibility.py meta-llama/Llama-2-7b-hf
"""

import sys
import torch
from transformers import AutoConfig

def verify_model(model_name):
    """
    Verify if a model is compatible with export_qkv_gqa.py
    
    Returns: (is_compatible, issues, warnings)
    """
    issues = []
    warnings = []
    
    print(f"\n{'='*70}")
    print(f"Verifying Model Compatibility")
    print(f"{'='*70}")
    print(f"Model: {model_name}\n")
    
    # Step 1: Load config
    print("[1/4] Loading model config...")
    try:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        print("  ✓ Config loaded successfully")
    except Exception as e:
        issues.append(f"Cannot load config: {e}")
        print(f"  ✗ Error: {e}")
        return False, issues, warnings
    
    # Step 2: Check model architecture
    print("\n[2/4] Checking architecture...")
    model_type = getattr(config, 'model_type', 'unknown')
    architectures = getattr(config, 'architectures', ['unknown'])
    print(f"  Model type: {model_type}")
    print(f"  Architectures: {architectures}")
    
    # Known compatible architectures
    compatible_types = ['llama', 'gemma', 'mistral', 'mixtral', 'qwen2']
    if model_type.lower() not in compatible_types:
        warnings.append(f"Model type '{model_type}' not in known compatible list: {compatible_types}")
        print(f"  ⚠ Warning: Unknown model type (may still work)")
    else:
        print(f"  ✓ Model type '{model_type}' is known to be compatible")
    
    # Step 3: Check attention configuration
    print("\n[3/4] Checking attention configuration...")
    
    # Check for number of attention heads
    num_heads = (getattr(config, 'num_attention_heads', None) or 
                 getattr(config, 'num_query_heads', None))
    if num_heads is None:
        issues.append("Cannot find num_attention_heads or num_query_heads in config")
        print(f"  ✗ Error: No attention heads config found")
    else:
        print(f"  ✓ Number of attention heads: {num_heads}")
    
    # Check for number of KV heads (for GQA)
    num_kv_heads = (getattr(config, 'num_key_value_heads', None) or
                    getattr(config, 'num_kv_heads', None) or
                    num_heads)
    print(f"  ✓ Number of KV heads: {num_kv_heads}")
    
    if num_kv_heads and num_heads:
        if num_kv_heads < num_heads:
            print(f"  ✓ Uses Group Query Attention (GQA): {num_heads // num_kv_heads} queries per KV head")
        elif num_kv_heads == num_heads:
            print(f"  ℹ Uses Multi-Head Attention (MHA): same number of Q and KV heads")
        else:
            warnings.append(f"Unusual config: num_kv_heads ({num_kv_heads}) > num_heads ({num_heads})")
    
    # Check hidden size
    hidden_size = getattr(config, 'hidden_size', None)
    if hidden_size is None:
        issues.append("Cannot find hidden_size in config")
        print(f"  ✗ Error: No hidden_size found")
    else:
        print(f"  ✓ Hidden size: {hidden_size}")
    
    # Check head dimension
    head_dim = getattr(config, 'head_dim', None)
    if head_dim is None and num_heads and hidden_size:
        head_dim = hidden_size // num_heads
        print(f"  ✓ Head dimension (computed): {head_dim}")
    elif head_dim:
        print(f"  ✓ Head dimension (explicit): {head_dim}")
    
    # Step 4: Check projection layer structure
    print("\n[4/4] Checking projection layer naming...")
    
    # We can't easily check this without loading the model, so just provide info
    print("  ℹ Expected projection layers: q_proj, k_proj, v_proj")
    print("  ℹ Expected attention module: self_attn or attention")
    print("  → Full verification requires loading the model (use test_gemma_arch.py)")
    
    # Summary
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    
    if issues:
        print(f"\n✗ INCOMPATIBLE - {len(issues)} critical issue(s) found:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        is_compatible = False
    else:
        print(f"\n✓ LIKELY COMPATIBLE")
        is_compatible = True
    
    if warnings:
        print(f"\n⚠ {len(warnings)} warning(s):")
        for i, warning in enumerate(warnings, 1):
            print(f"  {i}. {warning}")
    
    if is_compatible:
        print(f"\nNext steps:")
        print(f"  1. (Optional) Run full test: python test_gemma_arch.py {model_name}")
        print(f"  2. Export data: python export_qkv_gqa.py --model-path {model_name} --layer-id 0")
        
    print("")
    return is_compatible, issues, warnings


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python verify_model_compatibility.py <model-name>")
        print("\nExamples:")
        print("  python verify_model_compatibility.py google/gemma-2-2b-it")
        print("  python verify_model_compatibility.py meta-llama/Llama-2-7b-hf")
        print("  python verify_model_compatibility.py mistralai/Mistral-7B-v0.1")
        print("\nNote: google/gemma-3-4b-it does not exist. Use gemma-2-XXb-it instead.")
        sys.exit(1)
    
    model_name = sys.argv[1]
    
    # Check for common mistakes
    if 'gemma-3' in model_name.lower():
        print("\n" + "!"*70)
        print("WARNING: Gemma-3 models do not exist as of December 2025!")
        print("!"*70)
        print("\nDid you mean one of these?")
        print("  - google/gemma-2-2b-it  (2B parameters)")
        print("  - google/gemma-2-9b-it  (9B parameters)")
        print("  - google/gemma-2-27b-it (27B parameters)")
        print("\nProceeding with verification anyway...\n")
    
    is_compatible, issues, warnings = verify_model(model_name)
    
    # Exit code: 0 if compatible, 1 if not
    sys.exit(0 if is_compatible else 1)
