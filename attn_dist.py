import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from typing import List, Dict, Tuple, Optional
import argparse
import os
import csv
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class AttentionAnalyzer:
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-chat-hf", token: Optional[str] = None):
        """
        Initialize the attention analyzer with a language model.
        
        Args:
            model_name: HuggingFace model identifier
            token: HuggingFace access token (if None, will try to get from HF_TOKEN env var or .env file)
        """
        # Get token from parameter, environment variable, or .env file
        if token is None:
            token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=token
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            token=token
        )
        # Set attention implementation to 'eager' to support output_attentions
        if hasattr(self.model.config, 'attn_implementation'):
            self.model.config.attn_implementation = 'eager'
        # Also set it on the model if the method exists
        if hasattr(self.model, 'set_attn_implementation'):
            self.model.set_attn_implementation('eager')
        self.model.eval()
        
        # Store attention weights for each layer
        self.attention_weights = {}
        self.hooks = []
        
        # Register hooks to capture attention weights
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture attention weights from each layer."""
        def make_hook(layer_idx):
            def hook(module, input, output):
                # For LLaMA models, attention weights are in the output tuple
                # The attention weights are typically in the second element (if present)
                # We need to extract from the attention module itself
                if hasattr(module, 'attn_weights'):
                    self.attention_weights[layer_idx] = module.attn_weights.detach().cpu()
            return hook
        
        # Register hooks on attention modules
        for idx, layer in enumerate(self.model.model.layers):
            if hasattr(layer, 'self_attn'):
                hook = layer.self_attn.register_forward_hook(make_hook(idx))
                self.hooks.append(hook)
    
    def _get_attention_weights(self, input_ids, attention_mask=None):
        """
        Forward pass to extract attention weights.
        Uses transformers' output_attentions feature first, falls back to manual computation if needed.
        """
        self.attention_weights = {}
        
        # Try to get attention weights from model outputs first
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                output_attentions=True
            )
        
        # Extract attention weights from outputs
        if hasattr(outputs, 'attentions') and outputs.attentions:
            for layer_idx, attn in enumerate(outputs.attentions):
                if attn is not None:
                    # attn shape: [batch, num_heads, seq_len, seq_len] or [batch, seq_len, seq_len]
                    # Ensure it's in the right format
                    if len(attn.shape) == 4:
                        self.attention_weights[layer_idx] = attn.detach().cpu()
                    elif len(attn.shape) == 3:
                        # If single head, add head dimension
                        attn = attn.unsqueeze(1)
                        self.attention_weights[layer_idx] = attn.detach().cpu()
        
        # If still no weights, manually compute from model internals
        if not self.attention_weights:
            print("Warning: Could not extract attention weights from model outputs. Using manual computation.")
            self.attention_weights = self._compute_attention_weights_manual(input_ids, attention_mask)
        
        return self.attention_weights
    
    def _compute_attention_weights_manual(self, input_ids, attention_mask=None):
        """
        Manually compute attention weights by intercepting the forward pass.
        This is a fallback method when output_attentions doesn't work.
        """
        attention_weights_dict = {}
        
        # Use the model's forward pass but intercept attention computation
        # We'll use hooks to capture attention weights during forward pass
        captured_weights = {}
        
        def make_hook(layer_idx):
            def hook_fn(module, input_tuple, output):
                # Try to extract attention weights from the module's internal state
                # This depends on the specific implementation
                pass
            return hook_fn
        
        # Register hooks
        hooks = []
        for idx, layer in enumerate(self.model.model.layers):
            if hasattr(layer, 'self_attn'):
                hook = layer.self_attn.register_forward_hook(make_hook(idx))
                hooks.append(hook)
        
        # Run forward pass through model to get hidden states
        with torch.no_grad():
            # Get embeddings
            hidden_states = self.model.model.embed_tokens(input_ids)
            if hasattr(self.model.model, 'layers'):
                # Process through each layer manually
                for layer_idx, layer in enumerate(self.model.model.layers):
                    # Use the layer's forward but we need to extract attention
                    # For now, compute manually
                    attn_module = layer.self_attn
                    
                    # Apply layer norm first (if exists)
                    if hasattr(layer, 'input_layernorm'):
                        residual = hidden_states
                        hidden_states = layer.input_layernorm(hidden_states)
                    else:
                        residual = hidden_states
                    
                    # Compute Q, K, V
                    batch_size, seq_len, hidden_size = hidden_states.shape
                    # Get num_heads from config (Llama models have this in config)
                    num_heads = self.model.config.num_attention_heads
                    head_dim = hidden_size // num_heads
                    
                    query_states = attn_module.q_proj(hidden_states)
                    key_states = attn_module.k_proj(hidden_states)
                    value_states = attn_module.v_proj(hidden_states)
                    
                    # Reshape for multi-head attention
                    query_states = query_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
                    key_states = key_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
                    value_states = value_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
                    
                    # Compute attention scores
                    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / np.sqrt(head_dim)
                    
                    # Apply causal mask (upper triangular should be -inf for causal)
                    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=attn_weights.device, dtype=torch.bool), diagonal=1)
                    attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
                    
                    # Apply attention mask if provided
                    if attention_mask is not None:
                        expanded_mask = attention_mask[:, None, None, :].expand(batch_size, num_heads, seq_len, seq_len)
                        attn_weights = attn_weights.masked_fill(expanded_mask == 0, float('-inf'))
                    
                    # Apply softmax
                    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(hidden_states.dtype)
                    
                    # Store attention weights
                    attention_weights_dict[layer_idx] = attn_weights.detach().cpu()
                    
                    # Continue forward pass
                    attn_output = torch.matmul(attn_weights, value_states)
                    attn_output = attn_output.transpose(1, 2).contiguous()
                    attn_output = attn_output.view(batch_size, seq_len, -1)
                    attn_output = attn_module.o_proj(attn_output)
                    
                    # Add residual
                    hidden_states = residual + attn_output
                    
                    # Apply post-attention layer norm and MLP (simplified)
                    if hasattr(layer, 'post_attention_layernorm'):
                        residual = hidden_states
                        hidden_states = layer.post_attention_layernorm(hidden_states)
                        hidden_states = layer.mlp(hidden_states)
                        hidden_states = residual + hidden_states
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return attention_weights_dict
    
    def compute_ae(self, attention_weights: torch.Tensor) -> float:
        """
        Compute Attention Entropy (AE) for a single layer.
        
        Args:
            attention_weights: Tensor of shape [batch, num_heads, seq_len, seq_len]
        
        Returns:
            AE value (scalar)
        """
        # attention_weights: [batch, num_heads, seq_len, seq_len]
        batch_size, num_heads, seq_len, _ = attention_weights.shape
        
        # Check for NaN or Inf values
        if torch.isnan(attention_weights).any() or torch.isinf(attention_weights).any():
            # Replace NaN and Inf with small epsilon
            attention_weights = attention_weights.clone()
            attention_weights[torch.isnan(attention_weights)] = 1e-10
            attention_weights[torch.isinf(attention_weights)] = 1e-10
        
        # Normalize attention weights per query position (each row should sum to 1)
        # Sum over key dimension (last dim) for each query
        attn_sums = attention_weights.sum(dim=-1, keepdim=True)  # [batch, num_heads, seq_len, 1]
        # Avoid division by zero
        attn_sums = torch.clamp(attn_sums, min=1e-10)
        attention_weights_normalized = attention_weights / attn_sums
        
        # Ensure attention weights are non-negative and not too small
        attention_weights_normalized = torch.clamp(attention_weights_normalized, min=1e-10, max=1.0)
        
        # Compute entropy: -A * log(A) for each position
        # For each query position, compute entropy over key positions
        log_attn = torch.log(attention_weights_normalized)
        entropy = -attention_weights_normalized * log_attn
        
        # Sum over key positions for each query position, then average over queries, heads, and batches
        # AE = average entropy over all query positions, heads, and batches
        # For each query position, entropy is sum over key positions
        ae = entropy.sum() / (batch_size * num_heads * seq_len)
        
        # Check if result is valid
        if torch.isnan(ae) or torch.isinf(ae) or ae.item() < 0:
            return 0.0
        
        return ae.item()
    
    def compute_hta(
        self, 
        attention_weights: torch.Tensor, 
        harmful_token_indices: List[int],
        safe_token_indices: List[int]
    ) -> float:
        """
        Compute Harmful Token Attention (HTA) ratio.
        
        Args:
            attention_weights: Tensor of shape [batch, num_heads, seq_len, seq_len]
            harmful_token_indices: List of token indices that are harmful
            safe_token_indices: List of token indices that are safe
        
        Returns:
            HTA ratio (μ(T_harm) / μ(T_safe))
        """
        # attention_weights: [batch, num_heads, seq_len, seq_len]
        # A_{h,i,j} means attention from token i to token j in head h
        # We need to compute: μ(T) = 1/|T| * sum_h sum_{j in T} sum_i A_{h,i,j}
        # This is the average attention RECEIVED by tokens in T
        
        batch_size, num_heads, seq_len, _ = attention_weights.shape
        
        # Sum over query positions (i) for each key position (j)
        # Shape: [batch, num_heads, seq_len]
        attention_received = attention_weights.sum(dim=2)  # Sum over i (query positions)
        
        # Compute μ(T_harm)
        if len(harmful_token_indices) > 0:
            harm_attentions = []
            for j in harmful_token_indices:
                if 0 <= j < seq_len:
                    harm_attentions.append(attention_received[:, :, j])
            if harm_attentions:
                mu_harm = torch.stack(harm_attentions, dim=-1).sum() / (batch_size * num_heads * len(harmful_token_indices))
            else:
                mu_harm = torch.tensor(0.0)
        else:
            mu_harm = torch.tensor(0.0)
        
        # Compute μ(T_safe)
        if len(safe_token_indices) > 0:
            safe_attentions = []
            for j in safe_token_indices:
                if 0 <= j < seq_len:
                    safe_attentions.append(attention_received[:, :, j])
            if safe_attentions:
                mu_safe = torch.stack(safe_attentions, dim=-1).sum() / (batch_size * num_heads * len(safe_token_indices))
            else:
                mu_safe = torch.tensor(1e-10)  # Avoid division by zero
        else:
            mu_safe = torch.tensor(1e-10)
        
        # HTA = μ(T_harm) / μ(T_safe)
        hta = (mu_harm / mu_safe).item()
        
        return hta
    
    def _word_indices_to_token_indices(
        self, 
        prompt: str, 
        word_indices: List[int]
    ) -> List[int]:
        """
        Convert word indices (0-indexed) to token indices (0-indexed).
        
        Args:
            prompt: Input text prompt
            word_indices: List of word indices (0-indexed) in the original prompt
        
        Returns:
            List of token indices (0-indexed) corresponding to the given word indices
        """
        # Tokenize with offset mapping to get character positions
        encoding = self.tokenizer(
            prompt,
            return_offsets_mapping=True,
            return_attention_mask=False,
            return_tensors=None  # Return as lists, not tensors
        )
        
        # Get offset_mapping - handle different return formats
        offset_mapping = encoding['offset_mapping']
        if isinstance(offset_mapping, list):
            # Check if it's a list of lists (batch) or list of tuples (single sequence)
            if len(offset_mapping) > 0:
                first_elem = offset_mapping[0]
                # If first element is a tuple/list of 2 elements, it's a single sequence
                # If first element is a list of tuples, it's a batch
                if isinstance(first_elem, (list, tuple)) and len(first_elem) == 2:
                    # Single sequence: list of (start, end) tuples - use directly
                    offsets = offset_mapping
                elif isinstance(first_elem, list) and len(first_elem) > 0:
                    # Batch: first element is the sequence
                    offsets = offset_mapping[0]
                else:
                    offsets = offset_mapping
            else:
                offsets = []
        elif isinstance(offset_mapping, torch.Tensor):
            # Convert tensor to list
            offsets = offset_mapping[0].cpu().tolist() if offset_mapping.dim() > 1 else offset_mapping.cpu().tolist()
        else:
            # Try to convert to list
            try:
                offsets = list(offset_mapping)
            except:
                offsets = []
        
        # Split prompt into words and get their character positions
        # Use a more robust method to find word boundaries
        import re
        words = prompt.split()
        word_char_ranges = []
        char_pos = 0
        for word in words:
            # Find word position using regex to match word boundaries
            # This handles cases where a word might appear as substring of another word
            pattern = r'\b' + re.escape(word) + r'\b'
            match = re.search(pattern, prompt[char_pos:])
            if match:
                word_start = char_pos + match.start()
                word_end = word_start + len(word)
                word_char_ranges.append((word_start, word_end))
                char_pos = word_end
            else:
                # Fallback: simple find
                word_start = prompt.find(word, char_pos)
                if word_start != -1:
                    word_end = word_start + len(word)
                    word_char_ranges.append((word_start, word_end))
                    char_pos = word_end
        
        # Map tokens to words based on character offsets
        word_ids = []
        for i, offset_pair in enumerate(offsets):
            try:
                # Handle different formats: tuple, list, or array
                if isinstance(offset_pair, (list, tuple)) and len(offset_pair) >= 2:
                    token_start = int(offset_pair[0])
                    token_end = int(offset_pair[1])
                elif hasattr(offset_pair, '__getitem__') and hasattr(offset_pair, '__len__') and len(offset_pair) >= 2:
                    token_start = int(offset_pair[0])
                    token_end = int(offset_pair[1])
                else:
                    token_start, token_end = 0, 0
            except (TypeError, IndexError, ValueError, AttributeError) as e:
                print(f"Warning: Could not parse offset at index {i}: {offset_pair}, error: {e}")
                token_start, token_end = 0, 0
            if token_start == token_end == 0:
                # Special token (BOS/EOS/PAD)
                word_ids.append(None)
            else:
                # Find which word this token belongs to
                # A token belongs to a word if its start position is within the word's range
                # Also check if token overlaps with word (token_end > word_start and token_start < word_end)
                token_word_idx = None
                for word_idx, (word_start, word_end) in enumerate(word_char_ranges):
                    # Token overlaps with word if: token_start < word_end AND token_end > word_start
                    if token_start < word_end and token_end > word_start:
                        token_word_idx = word_idx
                        break
                word_ids.append(token_word_idx)
        
        # Convert word indices to token indices
        token_indices = []
        for word_idx in word_indices:
            if word_idx < 0 or word_idx >= len(words):
                print(f"Warning: Word index {word_idx} is out of range (0-{len(words)-1})")
                continue
            # Find all tokens that belong to this word
            for token_idx, mapped_word_idx in enumerate(word_ids):
                if mapped_word_idx == word_idx:
                    token_indices.append(token_idx)
        
        return sorted(list(set(token_indices)))  # Remove duplicates and sort
    
    def analyze(
        self, 
        prompt: str,
        harmful_token_indices: Optional[List[int]] = None,
        harmful_word_indices: Optional[List[int]] = None,
        safe_token_indices: Optional[List[int]] = None,
        safe_word_indices: Optional[List[int]] = None
    ) -> Dict[int, Dict[str, float]]:
        """
        Analyze attention patterns for a given prompt.
        
        Args:
            prompt: Input text prompt
            harmful_token_indices: List of token indices (0-indexed) that are harmful
            harmful_word_indices: List of word indices (0-indexed) that are harmful (alternative to harmful_token_indices)
            safe_token_indices: List of token indices (0-indexed) that are safe
            safe_word_indices: List of word indices (0-indexed) that are safe (alternative to safe_token_indices)
        
        Returns:
            Dictionary mapping layer index to {'ae': float, 'hta': float}
        """
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)
        
        # Get tokenized text
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        seq_len = len(tokens)
        words = prompt.split()
        
        # Print original prompt first
        print(f"\nOriginal prompt: {prompt}")
        
        # Convert word indices to token indices if provided
        if harmful_word_indices is not None:
            if harmful_token_indices is not None:
                print("Warning: Both harmful_token_indices and harmful_word_indices provided. Using harmful_word_indices.")
            harmful_token_indices = self._word_indices_to_token_indices(prompt, harmful_word_indices)
        
        if safe_word_indices is not None:
            if safe_token_indices is not None:
                print("Warning: Both safe_token_indices and safe_word_indices provided. Using safe_word_indices.")
            safe_token_indices = self._word_indices_to_token_indices(prompt, safe_word_indices)
        
        # If harmful_token_indices is provided, safe_token_indices is automatically
        # set to all other indices. Otherwise, use default heuristic.
        if harmful_token_indices is None:
            # Default: assume middle tokens might be harmful, first and last are safe
            # Use indices 1, 3, ... (skip first token which is often BOS)
            harmful_token_indices = [i for i in range(1, seq_len) if i % 2 == 0]
            safe_token_indices = [i for i in range(seq_len) if i not in harmful_token_indices]
        else:
            # If harmful indices are specified, safe indices are all others (unless explicitly provided)
            if safe_token_indices is None:
                safe_token_indices = [i for i in range(seq_len) if i not in harmful_token_indices]
        
        # Print formatted output with actual words and tokens
        if harmful_word_indices is not None:
            harmful_words = [words[i] for i in harmful_word_indices if 0 <= i < len(words)]
            print(f"Harmful words: {harmful_words}")
        
        harmful_tokens = [tokens[i] for i in harmful_token_indices if 0 <= i < len(tokens)]
        print(f"Harmful tokens: {harmful_tokens}")
        
        # Extract attention weights
        attention_weights_dict = self._get_attention_weights(input_ids, attention_mask)
        
        # Compute metrics for each layer
        results = {}
        for layer_idx in sorted(attention_weights_dict.keys()):
            attn_weights = attention_weights_dict[layer_idx]
            # attn_weights: [batch, num_heads, seq_len, seq_len]
            
            # Compute AE
            ae = self.compute_ae(attn_weights)
            
            hta = self.compute_hta(attn_weights, harmful_token_indices, safe_token_indices)
            
            results[layer_idx] = {
                'ae': ae,
                'hta': hta
            }
        
        return results


def load_prompts_from_csv(csv_path: str) -> List[Dict]:
    """
    Load prompts from CSV file.
    
    CSV format:
    prompt,harmful_indices,harmful_token_indices
    "prompt text",,
    "another prompt","1,3",
    "yet another","","2,4"
    
    Note: 
    - harmful_indices: word indices (0-indexed) - will be converted to token indices (default, recommended)
    - harmful_token_indices: token indices (0-indexed) - use only if you need direct token-level control
    - If harmful_indices or harmful_token_indices is specified, safe_indices will be automatically
      set to all other token indices.
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        List of dictionaries with 'prompt', 'harmful_indices', 'harmful_word_indices' keys
    """
    prompts = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt_data = {
                'prompt': row['prompt'].strip(),
                'harmful_word_indices': row.get('harmful_indices', '').strip() if row.get('harmful_indices') else None,  # harmful_indices is treated as word indices
                'harmful_token_indices': row.get('harmful_token_indices', '').strip() if row.get('harmful_token_indices') else None
            }
            prompts.append(prompt_data)
    return prompts


def parse_indices(indices_str: Optional[str]) -> Optional[List[int]]:
    """Parse comma-separated indices string to list of integers."""
    if not indices_str or indices_str.strip() == '':
        return None
    return [int(x.strip()) for x in indices_str.split(',') if x.strip()]


def save_all_results(output_dir: Path, all_prompt_results: List[Dict], model: str):
    """
    Save all prompt analysis results to a single JSON file.
    
    Args:
        output_dir: Directory to save results
        all_prompt_results: List of dictionaries with 'prompt' and 'results' keys
        model: Model name
    
    Returns:
        Path to saved JSON file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_{timestamp}.json"
    output_path = output_dir / filename
    
    # Prepare data structure
    output_data = {
        'model': model,
        'timestamp': timestamp,
        'prompts': []
    }
    
    for prompt_result in all_prompt_results:
        output_data['prompts'].append({
            'prompt': prompt_result['prompt'],
            'layers': prompt_result['results']
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Analyze attention patterns for jailbreak prompts")
    parser.add_argument("--prompt", type=str, default=None,
                       help="Input prompt to analyze (if not using CSV)")
    parser.add_argument("--csv", type=str, default="data/jailbreak_prompts.csv",
                       help="Path to CSV file containing prompts")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf",
                       help="Model name or path")
    parser.add_argument("--token", type=str, default=None,
                       help="HuggingFace access token (overrides .env file)")
    parser.add_argument("--harmful-indices", type=str, default=None,
                       help="Comma-separated list of harmful word indices (0-indexed, overrides CSV). Will be converted to token indices. Safe indices will be automatically set to all other tokens. (Default: word indices)")
    parser.add_argument("--harmful-token-indices", type=str, default=None,
                       help="Comma-separated list of harmful token indices (0-indexed, overrides CSV). Use only if you need direct token-level control. Safe indices will be automatically set to all other tokens.")
    parser.add_argument("--output-dir", type=str, default="outputs/results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer
    print("Initializing model...")
    analyzer = AttentionAnalyzer(model_name=args.model, token=args.token)
    print("Model loaded successfully!\n")
    
    # Determine if using CSV or single prompt
    if args.prompt:
        # Single prompt mode
        prompts = [{
            'prompt': args.prompt,
            'harmful_word_indices': args.harmful_indices,  # --harmful-indices is treated as word indices
            'harmful_token_indices': args.harmful_token_indices
        }]
    else:
        # CSV mode
        csv_path = Path(args.csv)
        if not csv_path.exists():
            print(f"Error: CSV file not found: {csv_path}")
            print(f"Please create the file or use --prompt for single prompt analysis.")
            return
        
        print(f"Loading prompts from {csv_path}...")
        prompts = load_prompts_from_csv(str(csv_path))
        print(f"Loaded {len(prompts)} prompts from CSV.\n")
    
    # Process each prompt
    all_results = []
    for idx, prompt_data in enumerate(prompts, 1):
        prompt = prompt_data['prompt']
        # Use command-line argument if provided, otherwise use CSV value
        # Both CSV's 'harmful_indices' and --harmful-indices are treated as word indices (default behavior)
        harmful_word_indices = parse_indices(args.harmful_indices) if args.harmful_indices else parse_indices(prompt_data.get('harmful_word_indices'))
        harmful_token_indices = parse_indices(args.harmful_token_indices) if args.harmful_token_indices else parse_indices(prompt_data.get('harmful_token_indices'))
        
        print(f"\n{'='*60}")
        print(f"Processing prompt {idx}/{len(prompts)}")
        print(f"{'='*60}")
        
        # Analyze prompt (safe_indices will be automatically computed from harmful_indices)
        results = analyzer.analyze(
            prompt=prompt,
            harmful_token_indices=harmful_token_indices,
            harmful_word_indices=harmful_word_indices,
            safe_token_indices=None  # Will be auto-computed
        )
        
        # Print summary
        print("\n=== Summary ===")
        print("Layer\tAE\t\tHTA")
        print("-" * 40)
        ae_values = []
        hta_values = []
        for layer_idx in sorted(results.keys()):
            ae_val = results[layer_idx]['ae']
            hta_val = results[layer_idx]['hta']
            print(f"{layer_idx}\t{ae_val:.6f}\t{hta_val:.6f}")
            ae_values.append(ae_val)
            hta_values.append(hta_val)
        
        # Print average across all layers
        if ae_values and hta_values:
            avg_ae = sum(ae_values) / len(ae_values)
            avg_hta = sum(hta_values) / len(hta_values)
            print("-" * 40)
            print(f"Average\t{avg_ae:.6f}\t{avg_hta:.6f}")
        
        all_results.append({
            'prompt': prompt,
            'results': results
        })
    
    # Save all results to a single JSON file
    if all_results:
        output_path = save_all_results(output_dir, all_results, args.model)
        print(f"\n{'='*60}")
        print(f"Completed processing {len(prompts)} prompt(s)")
        print(f"All results saved to: {output_path}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
