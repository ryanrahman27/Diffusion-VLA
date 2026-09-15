"""Backbone adapters for the diffusion-backbone study.

Every variant exposes the *same* interface (see base.BackboneAdapter) so the
downstream flow-matching action expert is identical across A/B/C/D. This is the
mechanism that makes the backbone the sole controlled variable.
"""
