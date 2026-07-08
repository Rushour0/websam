"""Export wrappers: thin ``torch.nn.Module``s that cut HF models at the
WebSAM graph boundaries (see ``websam_export.spec``).

Import the model-specific wrappers from their submodule, e.g.::

    from websam_export.wrappers.edgetam import EdgeTamMemoryAttentionWrapper

Heavy deps (torch/transformers) are only required when a submodule is
imported, so the base package stays light.
"""
