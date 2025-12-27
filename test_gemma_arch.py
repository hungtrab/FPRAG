"""
Quick test to check if Gemma model architecture is correctly detected.
"""

import torch
from transformers import AutoModelForCausalLM, AutoConfig

def test_model_architecture(model_name):
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print('='*60)
    
    # Load config only (faster)
    print("\n1. Loading config...")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    
    print(f"\nConfig attributes:")
    print(f"  model_type: {config.model_type}")
    print(f"  architectures: {config.architectures}")
    
    # Check attention heads config
    num_heads = getattr(config, 'num_attention_heads', 
                       getattr(config, 'num_query_heads', None))
    num_kv_heads = getattr(config, 'num_key_value_heads',
                           getattr(config, 'num_kv_heads', num_heads))
    hidden_size = config.hidden_size
    head_dim = getattr(config, 'head_dim', hidden_size // num_heads)
    
    print(f"\nAttention configuration:")
    print(f"  num_attention_heads: {num_heads}")
    print(f"  num_key_value_heads: {num_kv_heads}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  head_dim: {head_dim}")
    print(f"  GQA groups: {num_heads // num_kv_heads if num_kv_heads else 1}")
    
    # Load model to check layer structure
    print("\n2. Loading model (checking first layer only)...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"  bfloat16 failed, trying float16...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    
    # Check layer structure
    if hasattr(model, 'model'):
        base_model = model.model
    else:
        base_model = model
    
    print(f"\nModel structure:")
    print(f"  has model.model: {hasattr(model, 'model')}")
    print(f"  has base_model.layers: {hasattr(base_model, 'layers')}")
    
    if hasattr(base_model, 'layers'):
        layer0 = base_model.layers[0]
        print(f"\nLayer 0 attributes: {[attr for attr in dir(layer0) if not attr.startswith('_')][:10]}")
        
        # Check attention module
        has_self_attn = hasattr(layer0, 'self_attn')
        has_attention = hasattr(layer0, 'attention')
        print(f"\n  has self_attn: {has_self_attn}")
        print(f"  has attention: {has_attention}")
        
        if has_self_attn:
            attn = layer0.self_attn
            print(f"  Using: self_attn")
        elif has_attention:
            attn = layer0.attention
            print(f"  Using: attention")
        else:
            print(f"  ERROR: No attention module found!")
            return
        
        # Check projection modules
        print(f"\nAttention module attributes: {[attr for attr in dir(attn) if not attr.startswith('_')][:15]}")
        print(f"  has q_proj: {hasattr(attn, 'q_proj')}")
        print(f"  has k_proj: {hasattr(attn, 'k_proj')}")
        print(f"  has v_proj: {hasattr(attn, 'v_proj')}")
        
        if hasattr(attn, 'q_proj'):
            print(f"\nProjection shapes:")
            print(f"  q_proj.weight: {attn.q_proj.weight.shape}")
            print(f"  k_proj.weight: {attn.k_proj.weight.shape}")
            print(f"  v_proj.weight: {attn.v_proj.weight.shape}")
    
    print(f"\n✓ Architecture detection successful!")
    
if __name__ == '__main__':
    import sys
    
    # Test with the model specified in command line or default
    # Note: google/gemma-3-4b-it may not exist yet. Latest is Gemma-2 series.
    # Common Gemma models: google/gemma-2-2b-it, google/gemma-2-9b-it, google/gemma-2-27b-it
    model_name = sys.argv[1] if len(sys.argv) > 1 else 'google/gemma-2-2b-it'
    
    print(f"Note: If you're looking for google/gemma-3-4b-it, it may not exist yet.")
    print(f"Latest available Gemma models are Gemma-2 series (2b, 9b, 27b).")
    print(f"Testing with: {model_name}\n")
    
    test_model_architecture(model_name)
