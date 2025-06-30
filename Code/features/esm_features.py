import numpy as np
import pandas as pd
from tqdm import tqdm
import esm
import json
import torch

class ESMFeatureExtractor:
    """
    A class for extracting sequence-based features from protein sequences
    using Biopython's ProteinAnalysis and ProtLearn.

    Currently supports:
        - Molecular weight
        - Aromaticity
        - GRAVY (hydropathy)
        - Isoelectric point
        - Flexibility
        - Length
        - CTDC composition features

    Optional:
        Adds binary solubility column based on a threshold.
    """

    def __init__(self, solubility_threshold=0.5):
        """
        Initialize the feature extractor.

        Args:
            solubility_threshold (float): Threshold to convert solubility scores to binary labels.
        """
        self.solubility_threshold = solubility_threshold

    def generate_ESM_embeddings(self, input_data, output_path=None):
        """
        Generate ESM-based features for a set of protein sequences.

        Args:
            input_data (str or DataFrame): Input CSV/PKL file path or DataFrame with columns: 
                                                    'gene' and 'sequence'.
            output_path (str, optional): Optional path to save the output CSV/PKL file.

        Returns:
            DataFrame: The input DataFrame with appended feature columns.
        """
        if isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        elif input_data.endswith('.csv'):
            df = pd.read_csv(input_data)
        elif input_data.endswith('.pkl'):
            df = pd.read_pickle(input_data)
        else:
            raise ValueError("Input must be a DataFrame, .csv, or .pkl file.")
        # Convert to list
        sequences_train = [(row['gene'], row['sequence']) for _, row in df.iterrows()]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        # Load the ESM-2 model and alphabet
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        model.eval().to(device)  # Disable dropout for deterministic results

        # Prepare a list to store embeddings
        embeddings_list = []
        if device.type == "cuda":
            batch_size = 32
        else:
            batch_size = 1
        
        # Loop through each sequence with tqdm to show progress
        for i in tqdm(range(0, len(sequences_train), batch_size), desc="Generating ESM Embeddings"):
            batch = sequences_train[i:i+batch_size]
            _, _, batch_tokens = batch_converter(batch)
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33])
            embedding = results["representations"][33].mean(dim=1)  # Mean pooling
            embeddings_list.append(embedding.cpu().numpy()[0])

        # Add embeddings to the DataFrame and save
        df['esm_embedding'] = embeddings_list

        # Save if requested
        if output_path:
            if output_path.endswith('.csv'):
                df['esm_embedding'] = df['esm_embedding'].apply(lambda x: json.dumps(x.tolist()))
                df.to_csv(output_path, index=False)
            elif output_path.endswith('.pkl'):
                df.to_pickle(output_path)
            else:
                raise ValueError("Output must be a .csv or .pkl file.")

        return df
