import node_helpers
import comfy.utils
import comfy.sd
import folder_paths
import torch
import re
import os
import time
from safetensors.torch import save_file
import json

import torch
import torch.nn.functional as F

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
            if indices_str.strip() == "":
                return []
            parts = indices_str.split(",")
            for part in parts:
                part = part.strip()
                
                # skip empty parts
                if not part:
                    continue
                
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

        # check if layer index is empty, return original lora
        if not layer_indices_list:
            return (modified_lora,)
        
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

class CreateLoraMappingJson:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "loraA": ("LORA", {"tooltip": "Lora A as a source for the mapping."}),
                "loraB": ("LORA", {"tooltip": "Lora B as a target for the mapping."}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("mapping_json",)
    FUNCTION = "create_mapping_json"
    CATEGORY = "LoraUtils"
    DESCRIPTION = "Create a mapping json that maps keys from LoRA A to LoRA B for conversion purposes by identifying common layer structures."

    def create_mapping_json(self, loraA, loraB):
        
        # Create a mapping from loraA keys to loraB keys
        mapping_json = {}
        
        # Get all keys from both LoRAs
        keysA = list(loraA.keys())
        keysB = list(loraB.keys())
        
        # First, map common keys directly
        for key in keysA:
            if key in keysB:
                mapping_json[key] = key
        
        # For keys that are not common, we need to find similar patterns
        # Group keys by their core structure, ignoring different prefixes/suffixes
        # Common LoRA patterns to look for:
        # - transformer_blocks (diffusion models)
        # - lora_up, lora_down
        # - lora_A, lora_B
        # - etc.
        
        # Create a function to extract the core layer structure from a key
        def extract_core_structure(key):
            # Look for transformer blocks pattern
            transformer_match = re.search(r'transformer_blocks\.(\d+)', key)
            if transformer_match:
                block_num = transformer_match.group(1)
                # Extract parts of the key around the transformer block
                # This helps match keys like "input_blocks.1.1.transformer_blocks.0.attn1.to_q.lora_up.weight" 
                # with "diffusion_model_transformer_blocks_0_attn_to_q.lora_linear_layer.weight"
                base_pattern = f"transformer_blocks.{block_num}"
                return base_pattern
            
            # Look for time or timestep related patterns
            time_match = re.search(r'(time_emb|time_mix)', key, re.IGNORECASE)
            if time_match:
                return time_match.group(0)
            
            # Look for input/output/middle blocks in UNet
            block_match = re.search(r'(input|output|middle)_blocks', key)
            if block_match:
                # For blocks with numbers like input_blocks.1.1
                block_pattern = re.search(r'(input|output|middle)_blocks\.(\d+)(\.(\d+))?', key)
                if block_pattern:
                    main_block = block_pattern.group(1)
                    block_num = block_pattern.group(2)
                    sub_block = block_pattern.group(4) if block_pattern.group(4) else ""
                    if sub_block:
                        return f"{main_block}_blocks.{block_num}.{sub_block}"
                    else:
                        return f"{main_block}_blocks.{block_num}"
            
            # Look for diffusion_model patterns
            diff_match = re.search(r'diffusion_model', key)
            if diff_match:
                # Extract more specific structure for diffusion model
                layer_match = re.search(r'diffusion_model\.([^\.]+\.(\d+))', key)
                if layer_match:
                    return f"diffusion_model.{layer_match.group(1)}"
                
            # If no specific pattern found, return a simplified version
            # Remove common LoRA specific parts to find core structure
            simplified = re.sub(r'\.lora_(up|down|A|B)(\.weight|\.bias)?', '', key)
            simplified = re.sub(r'_lora(_(up|down|A|B))?(_weight|_bias)?', '', simplified)
            
            return simplified
        
        # Group keys by their core structure
        structure_map_A = {}
        structure_map_B = {}
        
        for key in keysA:
            if key not in mapping_json:  # Skip already mapped keys
                core_structure = extract_core_structure(key)
                if core_structure not in structure_map_A:
                    structure_map_A[core_structure] = []
                structure_map_A[core_structure].append(key)
        
        for key in keysB:
            # We only care about unmapped keys in B for potential mapping
            core_structure = extract_core_structure(key)
            if core_structure not in structure_map_B:
                structure_map_B[core_structure] = []
            structure_map_B[core_structure].append(key)
        
        # Now try to map keys with the same core structure
        for structure in structure_map_A:
            if structure in structure_map_B:
                keys_a_list = structure_map_A[structure]
                keys_b_list = structure_map_B[structure]
                
                # Map keys with similar patterns but potentially different suffixes/prefixes
                min_len = min(len(keys_a_list), len(keys_b_list))
                for i in range(min_len):
                    mapping_json[keys_a_list[i]] = keys_b_list[i]
        
        # Convert mapping to JSON string with indentation
        mapping_json_string = json.dumps(mapping_json, indent=4)
        
        return (mapping_json_string,)

class ConvertLoraKeys:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora": ("LORA", {"tooltip": "Lora to be converted."}),
                "mapping_json": ("STRING", {"tooltip": "Mapping json which contains current lora keys and the values as target keys."}),
            }
        }
        
    RETURN_TYPES = ("LORA",)
    RETURN_NAMES = ("converted_lora",)
    FUNCTION = "convert_lora"
    CATEGORY = "LoraUtils"
    DESCRIPTION = "Convert lora keys to match the new naming convention."
    def convert_lora(self, lora, mapping_json):
        # convert mapping_json string to dict
        mapping_dict = json.loads(mapping_json)

        new_lora = {}
        for key, value in mapping_dict.items():
            print(f"From key: {key}, To target key: {value}")
            new_lora[value] = lora[key].clone()
            
        return (new_lora,)

class LoraAdd:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "loraA": ("LORA", {"tooltip": "Lora A to add."}),
                "loraB": ("LORA", {"tooltip": "Lora B to add."}),
                "alpha_a": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip": "Weight for LoRA A."}),
                "alpha_b": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip": "Weight for LoRA B."}),
                "target_rank": ("INT", {"default": -1, "min": -1, "max": 1024, "step": 1, "tooltip": "Target rank for merged LoRA. Use -1 to automatically determine based on minimum rank."}),
            }
        }

    RETURN_TYPES = ("LORA",)
    RETURN_NAMES = ("merged_lora",)
    FUNCTION = "add_lora"
    CATEGORY = "LoraUtils"
    DESCRIPTION = "Combine two LoRAs with different ranks via SVD-based rank alignment. This allows merging LoRAs with different ranks."

    def extract_alpha(self, lora_sd: dict, down_key: str) -> float:
        """
        Try to extract lora_alpha from common naming patterns.
        Assume alpha is stored as:
          - `{prefix}.alpha` (kohya)
          - not stored → use rank (diffusers default: alpha = rank)
        """
        # Common patterns
        base_key = re.sub(r"\.lora_down\.weight$", "", down_key)
        alpha_key = f"{base_key}.alpha"
        
        if alpha_key in lora_sd:
            alpha = lora_sd[alpha_key]
            if isinstance(alpha, torch.Tensor):
                alpha = alpha.item()
            return float(alpha)
        else:
            # Fallback: alpha = rank (diffusers default behavior)
            rank = lora_sd[down_key].shape[0]
            return float(rank)

    def absorb_alpha(self, lora_sd: dict):
        """Absorb lora_alpha / rank into lora_down (or up). Modifies in-place."""
        down_keys = [k for k in lora_sd if ".lora_down." in k and k.endswith(".weight")]
        for down_key in down_keys:
            up_key = down_key.replace(".lora_down.", ".lora_up.")
            if up_key not in lora_sd:
                continue
            alpha = self.extract_alpha(lora_sd, down_key)
            rank = lora_sd[down_key].shape[0]
            scale = alpha / rank
            # Absorb into lora_down (arbitrary choice; up would also work)
            lora_sd[down_key] = lora_sd[down_key] * scale
        return lora_sd

    def low_rank_approximation(self, lora_down: torch.Tensor, lora_up: torch.Tensor, target_rank: int):
        r, in_f = lora_down.shape
        out_f, _ = lora_up.shape
        k = min(target_rank, r, in_f, out_f)
        
        if k == r:
            return lora_down, lora_up

        # For typical LoRA (r <= 256), construct ΔW directly
        if r <= 256:
            delta_w = lora_up @ lora_down  # (out, in)
            U, S, Vt = torch.linalg.svd(delta_w, full_matrices=False)
            U_k = U[:, :k]
            S_k = S[:k]
            Vt_k = Vt[:k, :]
            
            sqrt_S = torch.sqrt(S_k).unsqueeze(1)  # (k, 1)
            new_up   = U_k * sqrt_S.T               # (out, k)
            new_down = sqrt_S * Vt_k                # (k, in)
            return new_down, new_up
        else:
            # Use randomized SVD for large r (rare)
            # For this implementation, we'll use the basic SVD approach
            delta_w = lora_up @ lora_down  # (out, in)
            U, S, Vt = torch.linalg.svd(delta_w, full_matrices=False)
            U_k = U[:, :k]
            S_k = S[:k]
            Vt_k = Vt[:k, :]
            
            sqrt_S = torch.sqrt(S_k).unsqueeze(1)  # (k, 1)
            new_up   = U_k * sqrt_S.T               # (out, k)
            new_down = sqrt_S * Vt_k                # (k, in)
            return new_down, new_up

    def add_lora(self, loraA, loraB, alpha_a=1.0, alpha_b=1.0, target_rank=-1):
        # Create copies to avoid modifying originals
        loraA_copy = {k: v.clone() for k, v in loraA.items()}
        loraB_copy = {k: v.clone() for k, v in loraB.items()}
        
        # Absorb alpha scaling
        loraA_copy = self.absorb_alpha(loraA_copy)
        loraB_copy = self.absorb_alpha(loraB_copy)
        
        # Find all down keys to identify LoRA layers
        down_keys_a = {k for k in loraA_copy if ".lora_down." in k and k.endswith(".weight")}
        down_keys_b = {k for k in loraB_copy if ".lora_down." in k and k.endswith(".weight")}
        common_down_keys = down_keys_a & down_keys_b

        # Process LoRA layers with rank alignment
        for down_key in common_down_keys:
            up_key = down_key.replace(".lora_down.", ".lora_up.")
            if up_key not in loraA_copy or up_key not in loraB_copy:
                continue

            down_a, up_a = loraA_copy[down_key], loraA_copy[up_key]
            down_b, up_b = loraB_copy[down_key], loraB_copy[up_key]

            # Determine target rank
            r_a, r_b = down_a.shape[0], down_b.shape[0]
            k = target_rank if target_rank != -1 else min(r_a, r_b)

            # Align A to target rank
            if r_a != k:
                down_a, up_a = self.low_rank_approximation(down_a, up_a, k)
                loraA_copy[down_key] = down_a
                loraA_copy[up_key] = up_a

            # Align B to target rank
            if r_b != k:
                down_b, up_b = self.low_rank_approximation(down_b, up_b, k)
                loraB_copy[down_key] = down_b
                loraB_copy[up_key] = up_b

            # Fuse the aligned tensors
            fused_down = alpha_a * down_a + alpha_b * down_b
            fused_up   = alpha_a * up_a   + alpha_b * up_b

            loraA_copy[down_key] = fused_down
            loraA_copy[up_key]   = fused_up

            # Preserve alpha if exists (set to target rank, conventional)
            base_key = re.sub(r"\.lora_down\.weight$", "", down_key)
            alpha_key = f"{base_key}.alpha"
            if alpha_key in loraA or alpha_key in loraB:
                loraA_copy[alpha_key] = torch.tensor(float(k))  # common convention

        # Now add remaining non-LoRA keys normally
        for key in loraB_copy.keys():
            if key not in loraA_copy:
                loraA_copy[key] = loraB_copy[key]

        return (loraA_copy, )


class LoraSimpleAdd:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "loraA": ("LORA", {"tooltip": "First LoRA to add (base LoRA)."}),
                "loraB": ("LORA", {"tooltip": "Second LoRA to add (will be added to base LoRA)."}),
                "alpha_a": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip": "Weight for LoRA A (multiplier for first LoRA)."}),
                "alpha_b": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip": "Weight for LoRA B (multiplier for second LoRA)."}),
            }
        }

    RETURN_TYPES = ("LORA",)
    RETURN_NAMES = ("combined_lora",)
    FUNCTION = "simple_add_lora"
    CATEGORY = "LoraUtils"
    DESCRIPTION = "Combine two LoRAs with the same ranks by adding their values together. Simple implementation that directly adds tensors with specified weights."

    def simple_add_lora(self, loraA, loraB, alpha_a=1.0, alpha_b=1.0):
        # Create a copy of loraA to avoid modifying the original
        combined_lora = {k: v.clone() for k, v in loraA.items()}
        
        # Add tensors from loraB to the combined_lora
        for key, tensor in loraB.items():
            if key in combined_lora:
                # If both LoRAs have the same key, add them together with their respective weights
                combined_lora[key] = alpha_a * combined_lora[key] + alpha_b * tensor
            else:
                # If the key is only in loraB, add it with its weight
                combined_lora[key] = alpha_b * tensor
        
        return (combined_lora,)

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
                "remove_zero_layers": ("BOOLEAN", {"default": False, "tooltip": "Remove layers where all values are zero before saving."}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_lora"
    OUTPUT_NODE = True
    CATEGORY = "LoraUtils"

    def save_lora(self, lora, filename, remove_zero_layers=False, output_dir=None):
        if output_dir is None:
            output_dir = self.output_dir
            
        # Ensure the directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Create the full path
        full_output_path = os.path.join(output_dir, filename)
        # Add .safetensors extension if not present
        if not full_output_path.lower().endswith('.safetensors'):
            full_output_path += '.safetensors'

        if remove_zero_layers:
            # Filter out layers where all values are zero
            filtered_lora = {}
            zero_layers = []
            for key, tensor in lora.items():
                if isinstance(tensor, torch.Tensor):
                    if not torch.allclose(tensor, torch.zeros_like(tensor), atol=1e-12):
                        filtered_lora[key] = tensor
                    else:
                        zero_layers.append(key)
                else:
                    # Skip non-tensor items or optionally warn/log
                    filtered_lora[key] = tensor  # or skip if undesired

            if zero_layers:
                print(f"[SaveLora] Removed {len(zero_layers)} zero-only layers: {zero_layers}")
        else:
            filtered_lora = lora

        # Save the filtered lora state dict
        save_file(filtered_lora, full_output_path)

        print(f"LoRA saved to: {full_output_path} (original {len(lora)} layers → {len(filtered_lora)} layers)")
        return {}


NODE_CLASS_MAPPINGS = {
    "LoadLoraOnly": LoadLoraOnly,
    "LoraLayersOperation": LoraLayersOperation,
    "MergeLoraToModel": MergeLoraToModel,
    "LoraStatViewer": LoraStatViewer,
    "SaveLora": SaveLora,
    "LoraAdd": LoraAdd,
    "LoraSimpleAdd": LoraSimpleAdd,
    "ConvertLoraKeys": ConvertLoraKeys,
    "CreateLoraMappingJson": CreateLoraMappingJson,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadLoraOnly": "Load LoRA Only",
    "LoraLayersOperation": "LoRA Layers Operation",
    "MergeLoraToModel": "Merge LoRA to Model",
    "LoraStatViewer": "LoRA Stat Viewer",
    "SaveLora": "Save LoRA",
    "LoraAdd": "Lora Add",
    "LoraSimpleAdd": "Lora Simple Add",
    "ConvertLoraKeys": "Convert Lora Keys",
    "CreateLoraMappingJson": "Create Lora Mapping Json",
}