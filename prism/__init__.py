# PRISM: inference-only package for text-to-motion generation.
# No mmotion dependency required.
#
# Usage:
#   from prism.pipelines.prism_from_pretrained import load_prism_pipeline_from_pretrained
#   pipe = load_prism_pipeline_from_pretrained("path/to/checkpoint")
#   smplx_dict = pipe(prompts="A person walks forward.", num_frames_per_segment=129)

__version__ = "0.1.0"
