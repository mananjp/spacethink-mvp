"""Domain-specific verifier implementations conforming to VerifierProtocol."""
from evaluate.verifiers.reaction_wheel import ReactionWheelVerifier
from evaluate.verifiers.astro_catalog import AstroCatalogVerifier

__all__ = ["ReactionWheelVerifier", "AstroCatalogVerifier"]
