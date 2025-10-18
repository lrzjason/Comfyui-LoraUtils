import node_helpers
import comfy.utils
import comfy.sd
import folder_paths
import torch
import re
import os
import time
from safetensors.torch import save_file


class LoadLoraOnly:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "The name of the LoRA file to load."}),
            }
        }

    RETURN_TYPES = ("LORA",)
    RETURN_NAMES = ("lora",)
    FUNCTION = "load_lora"

    CATEGORY = "LoraUtils"
    DESCRIPTION = "Load a LoRA file without applying it to any models. Use with MergeLoraToModel or other nodes to apply the LoRA later."

    def load_lora(self, lora_name):
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                self.loaded_lora = None

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        return (lora,)


class LoraLayersOperation:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora": ("LORA", {"tooltip": "The LoRA state dictionary to modify."}),
                "layer_pattern": ("STRING", {"default": ".*transformer_blocks\\.(\\d+)\\.", "multiline": False, "tooltip": "Regex pattern to match layer names. Use groups to extract layer indices."}),
                "layer_indices": ("STRING", {"default": "59", "multiline": False, "tooltip": "Comma-separated list of layer indices to operate on, with support for ranges (e.g., '59', '10,11,12', or '50-53')."}),
                "scale_factor": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip": "Scale factor to apply. Use 0 to zero out layers."}),
            }
        }

    RETURN_TYPES = ("LORA",)
    RETURN_NAMES = ("modified_lora",)
    FUNCTION = "modify_lora"

    CATEGORY = "LoraUtils"
    DESCRIPTION = "Modify specific layers in a LoRA by zeroing them out (when scale=0) or scaling them (otherwise) based on pattern matching."

    def modify_lora(self, lora, layer_pattern, layer_indices, scale_factor):
        # Parse layer indices from string, supporting ranges like "50-53" and comma-separated list
        def parse_indices(indices_str):
            indices = []
            parts = indices_str.split(",")
            for part in parts:
                part = part.strip()
                if "-" in part:
                    # Handle range notation like "50-53"
                    try:
                        start, end = part.split("-")
                        start_idx = int(start.strip())
                        end_idx = int(end.strip())
                        if start_idx > end_idx:
                            raise ValueError(f"Invalid range: {part} (start index greater than end index)")
                        indices.extend(range(start_idx, end_idx + 1))
                    except ValueError:
                        raise ValueError(f"Invalid range format: {part}. Use format like '10-15'.")
                else:
                    # Handle single integer
                    try:
                        indices.append(int(part))
                    except ValueError:
                        raise ValueError(f"Invalid layer index: {part}")
            return indices
        
        try:
            layer_indices_list = parse_indices(layer_indices)
        except ValueError as e:
            raise ValueError(f"Error parsing layer indices: {str(e)}")
        
        layer_set = set(layer_indices_list)
        modified_keys = []
        
        # Compile the pattern
        pattern = re.compile(layer_pattern)
        
        # Work on a copy of the state dict to avoid modifying original
        modified_lora = {}
        for key, value in lora.items():
            modified_lora[key] = value.clone()  # Clone tensors to avoid modifying original

        for key in modified_lora:
            match = pattern.search(key)
            if match:
                # Try to extract layer index - looking for first captured group
                layer_id = None
                if len(match.groups()) > 0:
                    try:
                        layer_id = int(match.group(1))
                    except ValueError:
                        # If grouping didn't extract a number, try looking for digits in the full match
                        digit_match = re.search(r'\d+', match.group(0))
                        if digit_match:
                            layer_id = int(digit_match.group())
                
                if layer_id is not None and layer_id in layer_set:
                    if scale_factor == 0:
                        # Zero out the layer when scale factor is 0
                        modified_lora[key] = torch.zeros_like(modified_lora[key])
                        modified_keys.append(key)
                    else:
                        # Scale the layer by the scale factor
                        modified_lora[key] = modified_lora[key] * scale_factor
                        modified_keys.append(key)

        print(f"Modified {len(modified_keys)} parameters in layers: {sorted(layer_set)} using pattern '{layer_pattern}'")
        return (modified_lora,)


class MergeLoraToModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model to apply the LoRA to."}),
                "lora": ("LORA", {"tooltip": "The loaded LoRA to apply."}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip": "How strongly to modify the diffusion model. This value can be negative."}),
            },
            "optional": {
                "clip": ("CLIP", {"tooltip": "The CLIP model to apply the LoRA to (optional)."}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip": "How strongly to modify the CLIP model. This value can be negative."}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    OUTPUT_TOOLTIPS = ("The modified diffusion model.", "The modified CLIP model.")
    FUNCTION = "apply_lora"

    CATEGORY = "LoraUtils"
    DESCRIPTION = "Apply a pre-loaded LoRA to diffusion and CLIP models. This allows separation of loading and applying LoRAs."

    def apply_lora(self, model, lora, strength_model, clip=None, strength_clip=1.0):
        # Both model and clip are provided
        if clip is None:
            strength_clip = 0
        
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
        return (model_lora, clip_lora)


class LoraStatViewer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora": ("LORA", {"tooltip": "The loaded LoRA to analyze."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lora_info",)
    FUNCTION = "view_lora_stats"

    CATEGORY = "LoraUtils"
    DESCRIPTION = "View information about LoRA layers to help define layer patterns for LoraLayersOperation."

    def view_lora_stats(self, lora):
        result = []
        result.append("=== LoRA Statistics ===")
        result.append(f"Total number of keys: {len(lora.keys())}")
        
        # Group keys by pattern
        key_types = {}
        for key in lora.keys():
            # Extract the layer type (like 'lora.down.weight', 'lora.up.weight', etc.)
            # Common pattern: prefix.layer_type
            key_parts = key.split('.')
            if len(key_parts) >= 3:
                layer_type = '.'.join(key_parts[-3:])  # Get last 3 parts as layer type
            else:
                layer_type = key  # If not enough parts, use the full key
            
            if layer_type not in key_types:
                key_types[layer_type] = []
            key_types[layer_type].append(key)
        
        result.append("\nLayer types found in LoRA:")
        for layer_type, keys in key_types.items():
            result.append(f"  {layer_type}: {len(keys)} keys")

        # Show some example keys to help with pattern matching
        result.append("\nFirst 10 keys (helpful for pattern creation):")
        for i, key in enumerate(list(lora.keys())[:10]):
            result.append(f"  [{i}] {key}")
        
        # Look for transformer blocks if present
        transformer_keys = [key for key in lora.keys() if 'transformer_blocks' in key]
        if transformer_keys:
            result.append(f"\nFound {len(transformer_keys)} transformer block related keys")
            unique_blocks = set()
            for key in transformer_keys:
                # Find pattern like transformer_blocks.59. or similar
                matches = re.findall(r'transformer_blocks\.(\d+)', key)
                for match in matches:
                    unique_blocks.add(int(match))
            
            if unique_blocks:
                sorted_blocks = sorted(list(unique_blocks))
                result.append(f"Transformer block indices present: {sorted_blocks[:10]}{'...' if len(sorted_blocks) > 10 else ''}")
                if len(sorted_blocks) <= 10:
                    result.append(f"All transformer block indices: {sorted_blocks}")
        
        result.append("========================")
        
        # Add the full state dictionary
        result.append("\nFull LoRA State Dictionary Keys:")
        for i, key in enumerate(lora.keys()):
            result.append(f"  [{i}] {key}")
        
        # Join all results into a single string
        output_string = "\n".join(result)
        
        return (output_string,)


class SaveLora:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora": ("LORA", {"tooltip": "The modified LoRA state dictionary to save."}),
                "filename": ("STRING", {"default": "my_lora.safetensors", "tooltip": "Filename to save the LoRA as (e.g. my_lora.safetensors)."}),                
            },
            "optional": {
                "output_dir": ("STRING", {"default": folder_paths.get_output_directory(), "tooltip": "Directory to save the LoRA to. Defaults to ComfyUI output directory if not provided."}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_lora"
    OUTPUT_NODE = True
    CATEGORY = "LoraUtils"

    def save_lora(self, lora, filename, output_dir=None):
        if output_dir is None:
            output_dir = self.output_dir
            
        # Ensure the directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Create the full path
        full_output_path = os.path.join(output_dir, filename)
        
        # Add .safetensors extension if not present
        if not full_output_path.lower().endswith('.safetensors'):
            full_output_path += '.safetensors'
        
        # Save the lora state dict as safetensors (this will overwrite if file exists)
        save_file(lora, full_output_path)
        
        print(f"LoRA saved to: {full_output_path}")
        return {}


NODE_CLASS_MAPPINGS = {
    "LoadLoraOnly": LoadLoraOnly,
    "LoraLayersOperation": LoraLayersOperation,
    "MergeLoraToModel": MergeLoraToModel,
    "LoraStatViewer": LoraStatViewer,
    "SaveLora": SaveLora,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadLoraOnly": "Load LoRA Only",
    "LoraLayersOperation": "LoRA Layers Operation",
    "MergeLoraToModel": "Merge LoRA to Model",
    "LoraStatViewer": "LoRA Stat Viewer",
    "SaveLora": "Save LoRA",
}