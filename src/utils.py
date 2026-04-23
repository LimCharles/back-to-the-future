import torch
from .hmm import HMM
from .sohmm import SOHMM
from .chmm import CHMM
import pandas as pd
from typing import Tuple

torch.set_float32_matmul_precision('high')


def load_hmm_model(hmm_model_path: str, device: str = 'cuda:0') -> HMM:
    """
    Load the pretrained first-order HMM model.

    Args:
        hmm_model_path (str): Path to the saved HMM model.
        device (str): Device to load the model on.

    Returns:
        HMM: Loaded HMM model.
    """
    hmm_model = HMM.from_pretrained(hmm_model_path, local_files_only=True).to(device)
    hmm_model.eval()
    return hmm_model


def load_sohmm_model(sohmm_model_path: str, device: str = 'cuda:0') -> SOHMM:
    """
    Load the pretrained second-order HMM (SOHMM / SHMM) model.

    Matches the sohmm-branch checkpoint format: alpha_exp (H,H,H),
    beta (H,V) in log-space, gamma (H,H) in log-space. Uses
    PyTorchModelHubMixin.from_pretrained to deserialize
    safetensors/pytorch_model.bin.

    Args:
        sohmm_model_path (str): Path to the saved SOHMM directory.
        device (str): Device to load the model on.

    Returns:
        SOHMM: Loaded SOHMM model.
    """
    sohmm_model = SOHMM.from_pretrained(sohmm_model_path, local_files_only=True).to(device)
    sohmm_model.eval()
    return sohmm_model


def load_chmm_model(chmm_model_path: str, device: str = 'cuda:0') -> CHMM:
    """
    Load the pretrained clone-hidden HMM (CHMM) model.

    Uses ``CHMM.from_pretrained`` (reference-format ``model.pt`` payload with
    ``config``, ``gamma``, ``pair_codes``, ``transition_values``,
    ``transition_floor``). Falls back to dense ``alpha_exp`` payloads via
    the reference's ``_pair_codes_from_dense`` path.

    Args:
        chmm_model_path (str): Path to the saved CHMM directory.
        device (str): Device to load the model on.

    Returns:
        CHMM: Loaded CHMM model in eval mode.
    """
    chmm_model = CHMM.from_pretrained(chmm_model_path, map_location=device)
    chmm_model.eval()
    return chmm_model


def load_weights(weights_file: str, device: str = "cpu") -> torch.Tensor:
    """
    Load weights from CSV file.

    Args:
        weights_file (str): Path to the weights CSV file with columns 'Token ID' and 'Coefficient'.
        device (str): Device to load the tensors onto.

    Returns:
        torch.Tensor: weights_tensor of shape (V,) where V is the vocab size.
    """
    try:
        df_weights = pd.read_csv(weights_file)
    except Exception as e:
        raise FileNotFoundError(f"Failed to read weights file '{weights_file}': {e}")

    required_columns = {'Token ID', 'Coefficient'}
    if not required_columns.issubset(df_weights.columns):
        raise ValueError(f"Weights CSV must contain columns: {required_columns}")

    df_weights = df_weights.sort_values('Token ID').reset_index(drop=True)

    expected_token_ids = list(range(len(df_weights)))
    actual_token_ids = df_weights['Token ID'].tolist()
    if actual_token_ids != expected_token_ids:
        raise ValueError("Token IDs in weights file are not sequential starting from 0.")

    coefficients = df_weights['Coefficient'].values
    weights_tensor = torch.tensor(coefficients, dtype=torch.float32, device=device)

    return weights_tensor
