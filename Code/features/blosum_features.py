import numpy as np
import pandas as pd
import blosum as bl
from tqdm import tqdm
import json

class BlosumFeatureExtractor:
    def __init__(self, blosum_matrix = None):
        """
        Initializes the Blosum Matrix for embedding generation.

        Args:
            blosum_matrix (dict): BLOSUM matrix to calculate scores and embeddings. Defaults to Blosum62 from Blosum library
        """
        if blosum_matrix is None:
            self.blosum_matrix = bl.BLOSUM(62, default=0)
        else:
            self.blosum_matrix = blosum_matrix
    # Define a function to calculate the BLOSUM score for a sequence
    def generate_blosum_embeddings(self, input_data, output_path = None):
        """
        Adds BLOSUM score and BLOSUM embeddings to a DataFrame from a CSV or pickle file.

        Args:
            input_data (str or DataFrame): Input CSV/PKL file path or DataFrame with columns: 
                                                    'gene' and 'sequence'.
            output_path (str, optional): Path to save the output file.

        Returns:
            pd.DataFrame: DataFrame with BLOSUM score and embeddings added.
        """
        # Load the input file
        if isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        elif input_data.endswith('.csv'):
            df = pd.read_csv(input_data)
        elif input_data.endswith('.pkl'):
            df = pd.read_pickle(input_data)
        else:
            raise ValueError("Unsupported file format. Only .csv, .pkl, or pandas dataframe are supported.")

        # Ensure the file has a 'sequence' column
        if 'sequence' not in df.columns:
            raise ValueError("The input file must contain a 'sequence' column.")
        
        tqdm.pandas(desc="Computing BLOSUM features")

        # Add BLOSUM score column
        df['blosum_score'] = df['sequence'].progress_apply(lambda seq: self.calculate_blosum_score(seq))

        # Add BLOSUM embedding column
        df['blosum62_embedding'] = df['sequence'].progress_apply(lambda seq: self.calculate_blosum_embedding(seq))

        # Save the updated DataFrame
        if output_path is not None: 
            if output_path.endswith('.csv'):
                df['blosum62_embedding'] = df['blosum62_embedding'].apply(lambda x: json.dumps(x.tolist()))
                df.to_csv(output_path, index=False)
            elif output_path.endswith('.pkl'):
                df.to_pickle(output_path)

        return df
    
    def calculate_blosum_score(self, sequence):
        """
        Calculates the cumulative BLOSUM score for a given protein sequence based on consecutive amino acid pairs.

        The function iterates over each adjacent pair of amino acids in the sequence and sums their substitution scores
        using the provided BLOSUM matrix. This can provide an estimate of sequence stability or conservation
        based on observed evolutionary substitutions.

        Args:
            sequence (str): A string of amino acids representing a protein sequence.

        Returns:
            float: The average BLOSUM score for the sequence.
        """
        score = 0
        for i in range(len(sequence) - 1):
            score += self.blosum_matrix[sequence[i]][sequence[i+1]]
        return score/(len(sequence)-1)

    def calculate_blosum_embedding(self, sequence):
        """
        Generates a fixed-length BLOSUM62-based embedding vector for a given protein sequence.

        The function creates a (20 x sequence length) matrix where each column is the BLOSUM62 substitution vector
        of an amino acid in the input sequence. It then performs mean pooling across the sequence length
        to produce a 20-dimensional embedding vector representing the sequence.

        Args:
            sequence (str): A string of amino acids representing a protein sequence.

        Returns:
            np.ndarray: A NumPy array of shape (20,) representing the average BLOSUM embedding for the sequence.
        """
        # Define the standard amino acids in order
        AMINO_ACIDS = ['A', 'R', 'N', 'D', 'C', 
                    'Q', 'E', 'G', 'H', 'I', 
                    'L', 'K', 'M', 'F', 'P', 
                    'S', 'T', 'W', 'Y', 'V']
        embedding_matrix = np.stack([
            np.array([self.blosum_matrix[aa][ref] for ref in AMINO_ACIDS])
            for aa in sequence
        ], axis=1)  # shape = (20, len(sequence))

        pooled_embedding = embedding_matrix.mean(axis=1)  # shape = (20,)
        return pooled_embedding


