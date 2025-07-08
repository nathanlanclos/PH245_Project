# === File: run_esm_embeddings.py ===

import torch
import numpy as np
import os
import gc
import json
import warnings
from tqdm import tqdm
# ESM and Hugging Face imports
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProteinTensor, LogitsConfig, ESM3InferenceClient, ESMProtein
from huggingface_hub import login
import pandas as pd

# Suppress specific warnings if needed (e.g., Biotite ExperimentalWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='biotite')
warnings.filterwarnings("ignore", category=FutureWarning) # Often from huggingface_hub/transformers

class ESM3FeatureExtractor:
    """
    Feature extractor for ESM-3 embeddings.
    
    This class handles:
    - Hugging Face authentication
    - Loading the ESM-3 model
    - Generating embeddings for protein sequences
    - Saving results to CSV or pickle files
        """
    def __init__(self, hf_token: str = None, skip_existing: bool = True, device: str = None):
        """
            Initialize the feature extractor.

            Args:
                hf_token (str): Hugging Face token for model access (optional, overrides HF_TOKEN env var).
                skip_existing (bool): If True, skip sequences with existing embeddings.(optional, default: True)
                Device (str): Device to use for model inference ('cuda', 'cpu', or specific cuda device like 'cuda:0'). (optional, defaults to CUDA if available)

        """
        # --- Handle Hugging Face Login ---
        token = hf_token or os.environ.get("HF_TOKEN")
        if not token:
            print("Error: Hugging Face token not found.")
            print("Please provide it via the --hf_token argument or set the HF_TOKEN environment variable.")
            try:
                login()
                print("Hugging Face Hub login successful.")
            except Exception as e:
                print(f"Error logging into Hugging Face Hub: {e}")
                exit(1)
        try:
            login(token=token)
            print("Hugging Face Hub login successful.")
        except Exception as e:
            print(f"Error logging into Hugging Face Hub: {e}")
            exit(1)
        
        # --- Skip Existing ---
        self.skip_existing = skip_existing

        # --- Setup Device ---
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

    def generate_embedding(self, sequence: str, model: ESM3InferenceClient, expected_dim: int) -> torch.Tensor | None:
        """Generates embedding for a single sequence."""
        try:
            protein_input = ESMProtein(sequence=sequence)
            tensor_input = model.encode(protein_input)
            if len(tensor_input) == 0:
                # print(f"Debug: Encoding produced empty tensor for seq: {sequence[:20]}...")
                return None # Encoding failed or sequence invalid for tokenizer

            with torch.no_grad():
                logits_output = model.logits(tensor_input, LogitsConfig(return_embeddings=True))

            if logits_output.embeddings is None:
                # print(f"Debug: Embeddings are None for seq: {sequence[:20]}...")
                return None

            # Embeddings shape: [1, seq_len_with_special_tokens, embedding_dim]
            embeddings_tensor = logits_output.embeddings[0] # Remove batch dim
            # Exclude BOS/EOS tokens
            residue_embeddings = embeddings_tensor[1:-1, :]

            # Verify length and dimension
            if residue_embeddings.shape[0] != len(sequence):
                print(f"Warning: Embedding length ({residue_embeddings.shape[0]}) mismatch with sequence length ({len(sequence)}). Skipping.")
                return None
            if residue_embeddings.shape[1] != expected_dim:
                print(f"Warning: Embedding dimension ({residue_embeddings.shape[1]}) mismatch with expected ({expected_dim}). Skipping.")
                return None

            # Return tensor on CPU, float32
            return residue_embeddings.cpu().float()

        except Exception as e:
            # print(f"Debug: Error during embedding generation for seq {sequence[:20]}...: {e}")
            # import traceback; traceback.print_exc() # Uncomment for detailed debug
            return None

    def generate_ESM_embeddings(self, input_file: str, output_path: str, model_name:str = "esm3_sm_open_v1", batch_size: int = 8):
        # --- Load Model ---
        print(f"Loading model: {model_name}...")
        try:
            model: ESM3InferenceClient = ESM3.from_pretrained(model_name)
            print(f"Model {model_name} loaded successfully.")
            dtype = torch.bfloat16 if self.device.type == 'cuda' else torch.float32
            model = model.to(self.device, dtype=dtype)

            model = model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            exit(1)

        # --- Determine Embedding Dimension ---
        try:
            # Try to infer dimension (might need adjustment based on model structure)
            _ = next(model.parameters())
            # This is a heuristic, might fail for some models/versions
            embedding_dim = model.transformer.blocks[0].ffn[1].in_features
        except Exception:
            # Fallback based on known value for common models
            if model_name == "esm3_sm_open_v1":
                embedding_dim = 1536
            else:
                # Attempt a forward pass with a dummy sequence if possible, or ask user
                print("Warning: Could not automatically determine embedding dimension.")
                print("Attempting fallback for esm3_sm_open_v1 (1536).")
                embedding_dim = 1536 # Default fallback
                # Consider adding an argument --embedding_dim if needed
        print(f"Using embedding dimension: {embedding_dim}")

        # --- Process Files ---
        print(f"Scanning input file: {input_file}")
        try:
            if isinstance(input_file, pd.DataFrame):
                df = input_file.copy()
            elif input_file.endswith('.csv'):
                df = pd.read_csv(input_file)
            elif input_file.endswith('.pkl'):
                df = pd.read_pickle(input_file)
            else:
                print("Input must be a DataFrame, .csv, or .pkl file.")
                exit(1)
            if 'sequence' not in df.columns:
                print("Error: Input file must contain a 'sequence' column.")
                exit(1)
            all_sequences = [(row['gene'], row['sequence']) for _, row in df.iterrows()]

            print(f"Found {len(all_sequences)} sequences.")
        except FileNotFoundError:
            print(f"Error: Input file not found: {input_file}")
            exit(1)
        except Exception as e:
            print(f"Error listing input file: {e}")
            exit(1)

        processed_count = 0
        skipped_count = 0
        error_count = 0

        processed_embeddings = []
        # Use tqdm for progress bar
        for i, (gene, sequence) in enumerate(tqdm(all_sequences, desc="Generating Embeddings")):

            # Load sequence
            if sequence is None:
                error_count += 1
                continue # Skip if sequence loading failed

            # Generate embedding
            embedding_tensor = self.generate_embedding(sequence, model, embedding_dim)

            if embedding_tensor is not None:
                # Mean pooling over sequence length (dim=0)
                pooled = embedding_tensor.mean(dim=0)
                embedding_list = pooled.tolist()
                processed_embeddings.append(embedding_list)
                processed_count += 1
            else:
                print(f"Failed to generate embedding for {gene}. Skipping save.")
                processed_embeddings.append(None)
                error_count += 1

            # Memory cleanup periodically
            if (i + 1) % batch_size == 0:
                gc.collect()
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

        # Add embeddings to DataFrame
        df['esm_embedding'] = processed_embeddings

        # Print shape of embeddings list (rows, embedding_dim)
        valid_embs = [e for e in processed_embeddings if e is not None]
        if valid_embs:
            print(f"Embeddings list shape: ({len(valid_embs)}, {len(valid_embs[0])})")
        else:
            print("No valid embeddings generated.")

        # Final summary
        print("\n--- Embedding Generation Summary ---")
        print(f"Successfully processed and saved: {processed_count}")
        print(f"Skipped (already existing):       {skipped_count}")
        print(f"Failed (loading/embedding/saving): {error_count}")
        print(f"Total files considered:          {len(all_sequences)}")
        print(f"Embeddings saved in:             {output_path}")
        print("Script finished.")

        # Save if requested
        if output_path:
            if output_path.endswith('.csv'):
                df['esm_embedding'] = df['esm_embedding'].apply(lambda x: json.dumps(x) if x is not None else None)
                df.to_csv(output_path, index=False)
            elif output_path.endswith('.pkl'):
                df.to_pickle(output_path)
            else:
                raise ValueError("Output must be a .csv or .pkl file.")

        return df