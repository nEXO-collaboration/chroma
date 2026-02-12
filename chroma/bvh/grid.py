import numpy as np

from chroma.bvh.bvh import BVH, CHILD_BITS
from chroma.gpu.bvh import create_leaf_nodes, merge_nodes_detailed, concatenate_layers, collapse_chains

MAX_CHILD = 2**(32 - CHILD_BITS) - 1

def count_unique_in_sorted(a):
    return (np.ediff1d(a) > 0).sum() + 1

def make_recursive_grid_bvh(mesh, target_degree=3, verbose=False):
    '''Returns a BVH created using a 'recursive grid' method.

    This method is somewhat similar to the original Chroma BVH generator, 
    with a few adjustments:
      * Every triangle is boxed individually by a leaf node in the BVH
      * Triangles are not rearranged in Morton order since the
        leaf nodes are sorted instead.
      * The GPU is used to assist in the calculation of the tree to
        speed things up.

    '''
    world_coords, leaf_nodes, morton_codes = create_leaf_nodes(mesh)

    # rearrange in morton order
    argsort = morton_codes.argsort()
    leaf_nodes = leaf_nodes[argsort]
    morton_codes = morton_codes[argsort]

    # Create parent layers
    layers = [leaf_nodes]
    while len(layers[0]) > 1:        
        top_layer = layers[0]

        # Figure out how many bits to shift in morton code to achieve desired
        # degree
        nnodes = len(top_layer)
        
        nunique = count_unique_in_sorted(morton_codes)
        while nnodes / float(nunique) < target_degree and nunique > 1:
            morton_codes >>= 1
            nunique = count_unique_in_sorted(morton_codes)

        # Determine the grouping of child nodes into parents
        morton_delta = np.ediff1d(
            morton_codes,
            to_begin=np.array([1], dtype=morton_codes.dtype)
        ).astype(np.uint64)
        parent_morton_codes = morton_codes[morton_delta > 0]
        first_child = np.argwhere(morton_delta > 0).flatten().astype(np.uint32)
        nchild = np.ediff1d(first_child, to_end=nnodes - first_child[-1]).astype(np.uint32)
        
        # Expand groups that have too many children
        excess_children = np.argwhere(nchild > MAX_CHILD).flatten()
        if len(excess_children) > 0:
            if verbose:
                print('Expanding %d parent nodes' % len(excess_children))
            parent_morton_parts = np.split(parent_morton_codes, excess_children)
            first_child_parts = np.split(first_child, excess_children)
            nchild_parts = np.split(nchild, excess_children)

            new_parent_morton_parts = parent_morton_parts[:1]
            new_first_child_parts = first_child_parts[:1]

            for morton_part, first_part, nchild_part in zip(parent_morton_parts[1:], 
                                                            first_child_parts[1:],
                                                            nchild_parts[1:]):
                extra_first = np.arange(first_part[0], 
                                        first_part[0]+nchild_part[0], 
                                        MAX_CHILD).astype(first_part.dtype)
                new_first_child_parts.extend([extra_first, first_part[1:]])
                new_parent_morton_parts.extend([np.repeat(morton_part[0], 
                                                          len(extra_first)), morton_part[1:]])

            # Explicitly cast all lists to uint64 dtype to ensure final array has correct type
            # (Empty array is implicitly float64 dtype)
            parent_morton_codes = np.concatenate([p.astype(np.uint64) for p in new_parent_morton_parts])
            first_child = np.concatenate(new_first_child_parts)
            nchild = np.ediff1d(first_child, to_end=nnodes - first_child[-1]).astype(np.uint32)
              
        if nunique > 1: # Yes, I'm pedantic
            plural = 's'
        else:
            plural = ''
        if verbose:
            print('Merging %d nodes to %d parent%s' % (nnodes, nunique, plural))

        assert (nchild > 0).all()
        assert (nchild <= MAX_CHILD).all()
        
        # Merge children
        parents = merge_nodes_detailed(top_layer, first_child, nchild)
        layers = [parents] + layers
        morton_codes = parent_morton_codes

    nodes, layer_bounds = concatenate_layers(layers)
    nodes = collapse_chains(nodes, layer_bounds)
    return BVH(world_coords, nodes, layer_bounds[:-1])
