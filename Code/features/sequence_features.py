import numpy as np
import pandas as pd
import protlearn.features
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from tqdm import tqdm
import json

class SequenceFeatureExtractor:
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

    def generate_sequence_features(self, input_data, output_path=None):
        """
        Generate sequence-based features for a set of protein sequences.

        Args:
            input_data (str or DataFrame): Input CSV/PKL file path or DataFrame with columns:
                                           'sequence' and optionally 'solubility'.
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

        if 'solubility' in df.columns:
            df['binary_solubility'] = (df['solubility'] > self.solubility_threshold).astype(int)

        # Initialize feature lists
        molecular_weights, aromaticities, gravies = [], [], []
        isoelectric_points, lengths = [], []
        flex_mean, flex_std, flex_min, flex_max = [], [], [], []
        ctdc_dict = {}

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Sequence Features"):
            sequence = row['sequence']
            analysis = ProteinAnalysis(sequence)

            molecular_weights.append(analysis.molecular_weight())
            aromaticities.append(analysis.aromaticity())
            gravies.append(analysis.gravy())
            isoelectric_points.append(analysis.isoelectric_point())
            flex = analysis.flexibility()
            flex_mean.append(np.mean(flex))
            flex_std.append(np.std(flex))
            flex_min.append(np.min(flex))
            flex_max.append(np.max(flex))
            lengths.append(float(protlearn.features.length(sequence)[0]))
            ctdc_values, ctdc_labels = protlearn.features.ctdc(sequence)
            ctdc_values = np.squeeze(ctdc_values)

            for i, label in enumerate(ctdc_labels):
                if label not in ctdc_dict:
                    ctdc_dict[label] = []
                ctdc_dict[label].append(ctdc_values[i])

        # Assign feature columns
        df['molecular_weight'] = molecular_weights
        df['aromaticity'] = aromaticities
        df['gravy'] = gravies
        df['isoelectric_point'] = isoelectric_points
        df['flexibility_mean'] = flex_mean
        df['flexibility_std'] = flex_std
        df['flexibility_min'] = flex_min
        df['flexibility_max'] = flex_max
        df['length'] = lengths

        for label, values in ctdc_dict.items():
            df[label] = values

        # Save if requested
        if output_path:
            if output_path.endswith('.csv'):
                df.to_csv(output_path, index=False)
            elif output_path.endswith('.pkl'):
                df.to_pickle(output_path)
            else:
                raise ValueError("Output must be a .csv or .pkl file.")

        return df