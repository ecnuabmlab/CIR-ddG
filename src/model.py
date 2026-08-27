"""CIR-DDG: Cross-Chain Interaction Residual module.

A model-agnostic plug-in that adds explicit cross-chain binding geometry
to any ΔΔG backbone predictor via an additive residual correction:

    ΔΔG(mut) = f_base(mut) + s · m(mut) · r_φ(g(mut))

where:
    f_base: any backbone predictor's output
    g(mut): 22d cross-chain geometry descriptor
    r_φ:    lightweight MLP mapping geometry to scalar correction
    m(mut): interface mask (1 if mutation is at antibody-antigen interface)
    s:      antisymmetry sign (+1 forward, -1 reverse mutation)
"""

import torch
import torch.nn as nn


class CrossChainResidual(nn.Module):
    """Cross-Chain Interaction Residual (CCIR) module.

    Maps 22d interface geometry features to a scalar ΔΔG correction,
    active only at antibody-antigen interface sites.

    Architecture:
        Linear(22 → hidden) → GELU → Dropout → Linear(hidden → 1)

    Args:
        geom_dim: Dimension of input geometry features (default 22).
        hidden_dim: Hidden layer dimension (default 64).
        dropout: Dropout rate (default 0.1).
    """

    def __init__(self, geom_dim: int = 22, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(geom_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        base_prediction: torch.Tensor,
        geometry: torch.Tensor,
        interface_mask: torch.Tensor,
        antisymmetry_sign: float = 1.0,
    ) -> torch.Tensor:
        """Apply cross-chain residual correction to base predictions.

        Args:
            base_prediction: Backbone model's ΔΔG predictions, shape (B,).
            geometry: Standardized geometry features, shape (B, 22).
            interface_mask: Binary mask, 1 for interface mutations, shape (B,).
            antisymmetry_sign: +1 for forward mutations, -1 for reverse.

        Returns:
            Corrected predictions, shape (B,).
        """
        h = self.encoder(geometry)
        residual = self.output(h).squeeze(-1)
        correction = antisymmetry_sign * interface_mask.float() * residual
        return base_prediction + correction


class CIR_DDG(nn.Module):
    """Full CIR-DDG model: backbone MLP + cross-chain residual.

    For use when training from scratch on backbone features (e.g., frozen
    encoder features from Pythia/ESM-IF/ProteinMPNN).

    Architecture:
        Branch 1 (main): Linear(x_dim → h) → GELU → Drop → Linear(h → h) → GELU → Drop → Linear(h → 1)
        Branch 2 (CCIR): CrossChainResidual(22 → 64 → 1)

    Args:
        feature_dim: Dimension of backbone features.
        hidden_dim: Main head hidden dimension (default 256).
        residual_hidden: CCIR hidden dimension (default 64).
        dropout: Dropout rate (default 0.1).
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        residual_hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.main_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.residual = CrossChainResidual(
            geom_dim=22, hidden_dim=residual_hidden, dropout=dropout
        )

    def forward(
        self,
        features: torch.Tensor,
        geometry: torch.Tensor,
        interface_mask: torch.Tensor,
        antisymmetry_sign: float = 1.0,
    ) -> torch.Tensor:
        """Predict ΔΔG with cross-chain residual correction.

        Args:
            features: Backbone features, shape (B, feature_dim).
            geometry: Standardized geometry features, shape (B, 22).
            interface_mask: Binary mask for interface mutations, shape (B,).
            antisymmetry_sign: +1 for forward, -1 for reverse mutations.

        Returns:
            ΔΔG predictions, shape (B,).
        """
        base = self.main_head(features).squeeze(-1)
        return self.residual(base, geometry, interface_mask, antisymmetry_sign)
