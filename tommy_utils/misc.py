import numpy as np
from typing import List, Optional, Tuple

##########################################################
###### Function to generate matched encoding arrays ######
##########################################################

def unify_arrays(string_lists: List[List[str]], 
                 array_lists: List[List[np.ndarray]], 
                 check_axis: Optional[int] = None,
                 debug: bool = False,
                 list_names: Optional[List[str]] = None) -> Tuple[List[str], List[List[np.ndarray]]]:
    """
    Creates unified arrays with consistent ordering, filling in zeros for missing arrays.
    
    Args:
        string_lists: Lists of layer names.
        array_lists: Lists of arrays corresponding to the layer names.
        check_axis: Optional axis to check for size consistency.
        debug: If True, prints debugging information about added arrays.
        list_names: Optional list of names for each input list (e.g., ["List 1", "List 2"]).
    
    Returns:
        Tuple containing:
        - List of layer names in sorted order.
        - List of lists of arrays corresponding to the layer names.
    
    Raises:
        ValueError: If non-substituted arrays do not match their original shapes.
    """
    # Validate list_names
    if list_names is not None and len(list_names) != len(string_lists):
        raise ValueError("Length of list_names must match the number of input lists.")
    
    # Default list names if not provided
    if list_names is None:
        list_names = [f"List {i+1}" for i in range(len(string_lists))]
    
    # Step 1: Get unique layer names in sorted order
    unified_layers = get_unified_layers(string_lists)
    
    # Step 2: Create a dictionary mapping layer names to their reference shapes
    sizes = get_layer_sizes(string_lists, array_lists, unified_layers)
    
    # Step 3: Create a consistent set of arrays for each list
    result = create_unified_arrays(string_lists, array_lists, unified_layers, sizes, check_axis, debug, list_names)
    
    # Step 4: Verify that non-substituted arrays match their original shapes
    verify_non_substituted_arrays(string_lists, array_lists, result, unified_layers)
    
    return unified_layers, result


def get_unified_layers(string_lists: List[List[str]]) -> List[str]:
    """Returns a sorted list of unique layer names from all input lists."""
    return sorted(set().union(*string_lists))


def get_layer_sizes(string_lists: List[List[str]], 
                    array_lists: List[List[np.ndarray]], 
                    unified_layers: List[str]) -> dict:
    """Returns a dictionary mapping layer names to their reference shapes."""
    sizes = {}
    for layers, arrays in zip(string_lists, array_lists):
        for layer, arr in zip(layers, arrays):
            if layer not in sizes:
                sizes[layer] = arr.shape
    return sizes


def create_unified_arrays(string_lists: List[List[str]], 
                          array_lists: List[List[np.ndarray]], 
                          unified_layers: List[str], 
                          sizes: dict, 
                          check_axis: Optional[int],
                          debug: bool,
                          list_names: List[str]) -> List[List[np.ndarray]]:
    """Creates unified arrays for each list, filling in zeros for missing layers."""
    result = []
    for i, (layers, arrays) in enumerate(zip(string_lists, array_lists)):
        if debug:
            print(f"\nProcessing {list_names[i]}:")
        
        # Map existing arrays by layer name
        current_arrays = dict(zip(layers, arrays))
        
        # Build new arrays for all unified layers
        new_arrays = []
        for layer in unified_layers:
            if layer in current_arrays:
                arr = current_arrays[layer]
                # Check size consistency along the specified axis
                if check_axis is not None and arr.shape[check_axis] != sizes[layer][check_axis]:
                    raise ValueError(f"Size mismatch at axis {check_axis} for {layer}")
                new_arrays.append(arr)
                if debug:
                    print(f"  Retained array for {layer}: shape {arr.shape}")
            else:
                # Create a zero-filled array with the same shape as the reference, but update axis 0
                zero_shape = list(sizes[layer])
                zero_shape[0] = arrays[0].shape[0]  # Match axis 0 size with the current list
                zero_array = np.zeros(zero_shape)
                new_arrays.append(zero_array)
                if debug:
                    print(f"  Added zero-filled array for {layer}: shape {zero_array.shape}")
        
        result.append(new_arrays)
    return result


def verify_non_substituted_arrays(string_lists: List[List[str]], 
                                  array_lists: List[List[np.ndarray]], 
                                  result: List[List[np.ndarray]], 
                                  unified_layers: List[str]):
    """Verifies that non-substituted arrays match their original shapes."""
    for original_layers, original_arrays, result_arrays in zip(string_lists, array_lists, result):
        for layer, original_arr in zip(original_layers, original_arrays):
            # Find the corresponding array in the result
            result_arr = result_arrays[unified_layers.index(layer)]
            if not np.array_equal(original_arr, result_arr):
                raise ValueError(f"Non-substituted array for {layer} does not match the original array.")