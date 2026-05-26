import os
import json
import joblib
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
from recsys.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ReciprocalDataset(Dataset):
    """
    Consumes outputs from Prompt A pipeline and prepares them for the
    Two-Tower Retrieval and Reciprocal Scorer models.
    """
    def __init__(self, dataset_name: str, split: str = 'train', artifacts_path: str = 'data/artifacts', inference: bool = False):
        self.dataset_name = dataset_name
        self.split = split
        self.artifacts_path = artifacts_path
        self.inference = inference
        
        self._load_data()
        self._load_feature_metadata()
        self._load_advanced_artifacts() # New: Graph & Sequences
        self._build_tensors()
        
        # Pre-allocate advanced features to tensors indexed by unique user to save memory
        unique_users = np.unique(self.user_ids)
        user_to_idx = {uid: i for i, uid in enumerate(unique_users)}
        
        num_users = len(unique_users)
        graph_feats = np.zeros((num_users, settings.graph_embedding_dim), dtype=np.float32)
        seq_feats = np.zeros((num_users, settings.sequence_window_size, settings.sequence_hidden_dim), dtype=np.float32)
        
        for uid, i in user_to_idx.items():
            if uid in self.graph_features:
                graph_feats[i] = self.graph_features[uid]
            if uid in self.user_sequences:
                seq_feats[i] = self.user_sequences[uid]
                
        self.graph_feat_tensor = torch.from_numpy(graph_feats)
        self.seq_feat_tensor = torch.from_numpy(seq_feats)
        
        # Map each row to the unique user index
        self.row_to_user_idx = np.array([user_to_idx[uid] for uid in self.user_ids], dtype=np.int64)
        
        self.length = len(self.df)
        del self.df
        del self.graph_features
        del self.user_sequences
        
    def _load_data(self):
        base_name = self.dataset_name
        if base_name.endswith('.csv'):
            base_name = base_name[:-4]
        elif base_name.endswith('.edges'):
            base_name = base_name[:-6]
            
        cleaned_filename = f"splits_privacy_cleaned_{base_name}.parquet"
            
        filepath = os.path.join(self.artifacts_path, cleaned_filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Expected split dataset at {filepath}")
            
        self.df = pd.read_parquet(filepath)
        
        if not self.inference and 'split' in self.df.columns:
            self.df = self.df[self.df['split'] == self.split].reset_index(drop=True)
            
        logger.info(f"Loaded {len(self.df)} rows for split '{self.split}' from {self.dataset_name}")

    def _load_feature_metadata(self):
        vocab_path = os.path.join(self.artifacts_path, f"vocab_{self.dataset_name}.json")
        scaler_path = os.path.join(self.artifacts_path, f"scalers_{self.dataset_name}.joblib")
        
        if os.path.exists(vocab_path):
            with open(vocab_path, 'r') as f:
                self.vocabularies = json.load(f)
        else:
            self.vocabularies = {}
            
        if os.path.exists(scaler_path):
            self.scalers = joblib.load(scaler_path)
        else:
            self.scalers = {}

    def _load_advanced_artifacts(self):
        # Consume precomputed artifacts from Prompt A
        graph_path = os.path.join(self.artifacts_path, f"graph_embeddings_{self.dataset_name}.joblib")
        seq_path = os.path.join(self.artifacts_path, f"user_sequences_{self.dataset_name}.joblib")
        
        if os.path.exists(graph_path):
            feats = joblib.load(graph_path)
            self.graph_features = {k: v for k, v in feats.items()}
            logger.info(f"Loaded graph features for {self.dataset_name}")
        else:
            logger.warning(f"Graph features missing for {self.dataset_name}. Defaulting to zero tensors.")
            self.graph_features = {}
            
        if os.path.exists(seq_path):
            seqs = joblib.load(seq_path)
            self.user_sequences = {k: v for k, v in seqs.items()}
            logger.info(f"Loaded user sequences for {self.dataset_name}")
        else:
            logger.warning(f"User sequences missing for {self.dataset_name}. Defaulting to zero tensors.")
            self.user_sequences = {}


    def _build_tensors(self):
        # We need to map dataframe rows to user features, candidate features, pairwise, and context features.
        # This implementation uses the vocabularies to encode categorical variables into integers for embeddings.
        # Numerical features are scaled.
        
        self.user_features_num = []
        self.user_features_cat = {}
        
        self.cand_features_num = []
        self.cand_features_cat = {}
        
        self.forward_labels = []
        self.reverse_labels = []
        self.mutual_labels = []
        self.propensities = []
        
        cat_cols = list(self.vocabularies.keys())
        num_cols = list(self.scalers.keys())
        
        if not self.inference:
            # 1) Label Construction
            if 'u_likes_c' in self.df.columns and 'c_likes_u' in self.df.columns:
                self.forward_labels = self.df['u_likes_c'].values.astype(np.float32)
                self.reverse_labels = self.df['c_likes_u'].values.astype(np.float32)
                self.mutual_labels = (self.forward_labels * self.reverse_labels).astype(np.float32)
            else:
                # Fallback to single generic label
                label_col = None
                for col in ['match', 'rating', 'label', 'swiped_right']:
                    if col in self.df.columns:
                        label_col = col
                        break
                if label_col:
                    logger.warning(f"Using generic label '{label_col}' as fallback for forward/reverse/mutual labels.")
                    generic_labels = self.df[label_col].values.astype(np.float32)
                    
                    # Ensure labels are in [0, 1] for BCELoss
                    min_val, max_val = generic_labels.min(), generic_labels.max()
                    if max_val > 1.0 or min_val < 0.0:
                        logger.warning(f"Scaling generic labels from [{min_val}, {max_val}] to [0, 1]")
                        if max_val > min_val:
                            generic_labels = (generic_labels - min_val) / (max_val - min_val)
                        else:
                            generic_labels = np.zeros_like(generic_labels)
                            
                    self.forward_labels = generic_labels
                    self.reverse_labels = generic_labels
                    self.mutual_labels = generic_labels
                else:
                    logger.warning("No label columns found. Defaulting to 0.")
                    n = len(self.df)
                    self.forward_labels = np.zeros(n, dtype=np.float32)
                    self.reverse_labels = np.zeros(n, dtype=np.float32)
                    self.mutual_labels = np.zeros(n, dtype=np.float32)
                    
            # 2) Propensity Score Calculation
            if 'propensity' in self.df.columns:
                self.propensities = self.df['propensity'].values.astype(np.float32)
            else:
                # Dynamic calculation based on target occurrence frequency
                target_col = 'target_id' if 'target_id' in self.df.columns else 'cand_id'
                if target_col in self.df.columns:
                    counts = self.df[target_col].value_counts()
                    total = counts.sum()
                    freqs = (counts / total).to_dict()
                    self.propensities = self.df[target_col].map(freqs).fillna(1e-5).values.astype(np.float32)
                else:
                    self.propensities = np.ones(len(self.df), dtype=np.float32) * 0.1
        else:
            n = len(self.df)
            self.forward_labels = np.zeros(n, dtype=np.float32)
            self.reverse_labels = np.zeros(n, dtype=np.float32)
            self.mutual_labels = np.zeros(n, dtype=np.float32)
            self.propensities = np.ones(n, dtype=np.float32)
            
        # Process categories
        for col in cat_cols:
            vocab = {val: idx + 1 for idx, val in enumerate(self.vocabularies[col])} # 0 for unknown
            encoded = self.df[col].map(vocab).fillna(0).astype(np.int64).values
            # Heuristic separation: if it starts with 'candidate_' it's candidate feature
            if col.startswith('target_') or col.startswith('cand_'):
                self.cand_features_cat[col] = torch.tensor(encoded)
            else:
                self.user_features_cat[col] = torch.tensor(encoded)
                
        # Process numericals
        for col in num_cols:
            scaler = self.scalers[col]
            vals = self.df[[col]].fillna(0.0)
            scaled = scaler.transform(vals).astype(np.float32).flatten()
            if col.startswith('target_') or col.startswith('cand_'):
                self.cand_features_num.append(scaled)
            else:
                self.user_features_num.append(scaled)
                
        if self.user_features_num:
            self.user_features_num = torch.tensor(np.stack(self.user_features_num, axis=1))
        else:
            self.user_features_num = torch.zeros((len(self.df), 1))
            
        if self.cand_features_num:
            self.cand_features_num = torch.tensor(np.stack(self.cand_features_num, axis=1))
        else:
            self.cand_features_num = torch.zeros((len(self.df), 1))
            
        # Extract user_ids and cand_ids as numpy arrays to avoid slow iloc in __getitem__
        self.user_ids = np.zeros(len(self.df), dtype=object)
        for col in ['user_id', 'User_ID', 'userId']:
            if col in self.df.columns:
                self.user_ids = self.df[col].values
                break
                
        self.cand_ids = np.zeros(len(self.df), dtype=object)
        for col in ['target_id', 'cand_id', 'targetId']:
            if col in self.df.columns:
                self.cand_ids = self.df[col].values
                break

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        u_idx = self.row_to_user_idx[idx]
        return {
            'user_num': self.user_features_num[idx],
            'user_cat': {k: v[idx] for k, v in self.user_features_cat.items()},
            'cand_num': self.cand_features_num[idx],
            'cand_cat': {k: v[idx] for k, v in self.cand_features_cat.items()},
            'graph_feat': self.graph_feat_tensor[u_idx],
            'sequence': self.seq_feat_tensor[u_idx],
            'forward_label': self.forward_labels[idx],
            'reverse_label': self.reverse_labels[idx],
            'mutual_label': self.mutual_labels[idx],
            'propensity': self.propensities[idx],
            'user_id': self.user_ids[idx],
            'cand_id': self.cand_ids[idx]
        }
