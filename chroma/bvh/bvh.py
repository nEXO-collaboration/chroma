'''chroma.bvh: Bounding Volume Hierarchy generation and manipulation.'''

import numpy as np
from pycuda.gpuarray import vec

uint4 = vec.uint4 # pylint: disable-msg=C0103,E1101

CHILD_BITS = 28
# NumPy 2.x raises on out-of-range integer casts to uint32.  Mask before cast
# so this stays valid across NumPy 1.x and 2.x.
NCHILD_MASK = np.uint32((0xFFFF << CHILD_BITS) & 0xFFFFFFFF)

def unpack_nodes(nodes):
    '''Creates a numpy record array with the contents of nodes
    array unpacked into separate fields.

      ``nodes``: ndarray(shape=n, dtype=uint4)
         BVH node array in the packed x,y,z,w format.

    Returns ndarray(shape=n, dtype=[('xlo', np.uint16), ('xhi', np.uint16),
                                    ('ylo', np.uint16), ('yhi', np.uint16),
                                    ('zlo', np.uint16), ('zhi', np.uint16),
                                    ('child', np.uint32), ('nchild', np.uint16)])
    '''
    unpacked_dtype = np.dtype([('xlo', np.uint16), ('xhi', np.uint16),
                               ('ylo', np.uint16), ('yhi', np.uint16),
                               ('zlo', np.uint16), ('zhi', np.uint16),
                               ('child', np.uint32), ('nchild', np.uint16)])
    unpacked = np.empty(shape=len(nodes), dtype=unpacked_dtype)
    
    for axis in ['x', 'y', 'z']:
        unpacked[axis+'lo'] = nodes[axis] & 0xFFFF
        unpacked[axis+'hi'] = nodes[axis] >> 16
    unpacked['child'] = nodes['w'] & ~NCHILD_MASK
    unpacked['nchild']  = nodes['w'] >> CHILD_BITS

    return unpacked

class OutOfRangeError(Exception):
    '''The world coordinates cannot be transformed into fixed point
    coordinates because they exceed the range of an unsigned 16-bit
    fixed point number.'''
    def __init__(self, msg):
        Exception.__init__(self, msg)

class WorldCoords(object):
    '''A simple helper object that represents the transformation
    between floating point world coordinates and unsigned 16-bit fixed
    point coordinates.

    This class has a ``world_origin`` (3-vector), and a ``world_scale``
    (a scalar) property such that:
    
       \vec{r} = ``world_scale`` * (\vec{f} + ``world_origin``)

    Where \vec{r} is a vector in the real coordinate system and \vec{f}
    is a vector in the fixed point coordinate system.
    '''
    MAX_INT = 2**16 - 1

    def __init__(self, world_origin, world_scale):
        '''Transformation from fixed point to world coordinates.

          ``world_origin``: ndarray(shape=3)
              World x,y,z coordinates of (0,0,0) in fixed point coordinates.

          ``world_scale``: float
              Multiplicative scale factor to go from fixed point distances
              to real world distances.
        '''
        self.world_origin = np.array(world_origin, dtype=np.float32)
        self.world_scale = np.float32(world_scale)

    def world_to_fixed(self, world):
        '''Convert a vector, or array of vectors to the fixed point
        representation.
        
          ``world``: ndarray(shape=3) or ndarray(shape=(n,3))
             Vector or array of vectors in real world coordinates to convert to
             fixed point.

          .. warning: Conversion to fixed point rounds to nearest integer.

          Returns ndarray(shape=3, dtype=np.uint16) or 
                  ndarray(shape=(n,3), dtype=np.uint16)
        '''
        fixed = ((np.asarray(world, dtype=np.float64) - self.world_origin)
                 / self.world_scale).round()
        if int(fixed.max()) > WorldCoords.MAX_INT or fixed.min() < 0:
            raise OutOfRangeError('range = (%f, %f)' 
                                  % (fixed.min(), fixed.max()))
        else:
            return fixed.astype(np.uint16)

    def fixed_to_world(self, fixed):
        '''Convert a vector, or array of vectors to the world representation.
        
          ``fixed``: ndarray(shape=3, dtype=np.uint16) or 
                   ndarray(shape=(n,3), dtype=np.uint16)
             Vector or array of vectors in unsigned fixed point
             coordinates to convert to world coordinates.

          Returns ndarray(shape=3) or ndarray(shape=(n,3))    
        '''
        return np.asarray(fixed) * self.world_scale + self.world_origin


class BVH(object):
    '''A bounding volume hierarchy for a triangle mesh.

    For the purposes of Chroma, a BVH is a tree with the following properties:

      * Each node consists of an axis-aligned bounding box, a child ID
        number, and a boolean flag indicating whether the node is a
        leaf.  The bounding box is represented as a lower and upper
        bound for each Cartesian axis.

      * All nodes are stored in a 1D array with the root node first.

      * A node with a bounding box that has no surface area (upper and
        lower bounds equal for all axes) is a dummy node that should
        be ignored.  Dummy nodes are used to pad the tree to satisfy
        the fixed degree requirement described below, and have no
        children.

      * If the node is a leaf, then the child ID number refers to the
        ID number of the triangle this node contains.

      * If the node is not a leaf (an "inner" node), then the child ID
        number indicates the offset in the node array of the first
        child.  The other children of this node will be stored
        immediately after the first child.

      * All inner nodes have the same number of children, called the
        "degree" (technically the "out-degree") of the tree.  This
        avoid the requirement to save the degree with the node.

      * For simplicity, we also require nodes at the same depth
        in the tree to be contiguous, and the layers to be in order
        of increasing depth.

      * All nodes satisfy the bounding volume hierarchy constraint:
        their bounding boxes contain the bounding boxes of all their
        children.

    For space reasons, the BVH bounds are internally represented using
    16-bit unsigned fixed point coordinates.  Normally, we would want
    to hide that from you, but we would like to avoid rounding issues
    and high memory usage caused by converting back and forth between
    floating point and fixed point representations.  For similar
    reasons, the node array is stored in a packed record format that
    can be directly mapped to the GPU.  In general, you will not need
    to manipulate the contents of the BVH node array directly.
    '''

    def __init__(self, world_coords, nodes, layer_offsets):
        '''Create a BVH object with the given properties.

           ``world_coords``: chroma.bvh.WorldCoords
              Transformation from fixed point to world coordinates.
              
           ``nodes``: ndarray(shape=n, dtype=chroma.bvh.uint4) 
              List of nodes.  x,y,z attributes of array are the lower
              and upper limits of the bounding box (lower is the least
              significant 16 bits), and the w attribute is the child
              ID with the leaf state set in bit 31.  First node is root
              node.

           ``layer_offsets``: list of ints
              Offset in node array for the start of each layer.  The first
              entry must be 0, since the root node is first, and the 
              second entry must be 1, for the first child of the root node,
              unless the root node is also a leaf node.
        '''
        self.world_coords = world_coords
        self.nodes = nodes
        self.layer_offsets = layer_offsets

        # for convenience when slicing in get_layer
        self.layer_bounds = list(layer_offsets) + [len(nodes)]

    def get_layer(self, layer_number):
        '''Returns a BVHLayerSlice object corresponding to the given layer
        number in this BVH, with the root node at layer 0.
        '''
        layer_slice = slice(self.layer_bounds[layer_number],
                            self.layer_bounds[layer_number+1])
        return BVHLayerSlice(world_coords=self.world_coords,
                             nodes=self.nodes[layer_slice])

    def layer_count(self):
        '''Returns the number of layers in this BVH'''
        return len(self.layer_offsets)

    def __len__(self):
        '''Returns the number of nodes in this BVH'''
        return len(self.nodes)

def node_areas(nodes):
    '''Returns the areas of each node in an array of nodes in fixed point units.

       ``nodes``: ndarray(dtype=uint4)
          array of packed BVH nodes
    '''
    unpacked = unpack_nodes(nodes)
    delta = np.empty(shape=len(nodes), 
                     dtype=[('x', float), ('y', float), ('z', float)])
    for axis in ['x', 'y', 'z']:
        delta[axis] = unpacked[axis+'hi'] - unpacked[axis+'lo']    

    half_area = delta['x']*delta['y'] + delta['y']*delta['z'] \
        + delta['z']*delta['x']

    return 2.0 * half_area

class BVHLayerSlice(object):
    '''A single layer in a bounding volume hierarchy represented as a slice
    in the node array of the parent.  

    .. warning: Modifications to the node values in the BVHLayerSlice
    object affect the parent storage!

    A BVHLayerSlice has the same properties as the ``BVH`` class,
    except no ``layer_offsets`` list.
    '''

    def __init__(self, world_coords, nodes):
        self.world_coords = world_coords
        self.nodes = nodes

    def __len__(self):
        '''Return number of nodes in this layer'''
        return len(self.nodes)

    def areas_fixed(self):
        '''Return array of surface areas of all the nodes in this layer in
        fixed point units.'''
        return node_areas(self.nodes)

    def area_fixed(self):
        '''Return the surface area of all the nodes in this layer in
        fixed point units.'''
        return node_areas(self.nodes).sum()

    def area(self):
        '''Return the surface area of all the nodes in this layer in world
        units.'''
        return self.area_fixed().sum() * self.world_coords.world_scale**2

    def get_bounds(self):
        """Returns the lower and upper bounds of this layer in the BVH."""
        node_info = unpack_nodes(self.nodes)
        fixed_lower = \
            np.dstack([node_info[s] for s in ['xlo','ylo','zlo']]).squeeze()
        fixed_upper = \
            np.dstack([node_info[s] for s in ['xhi','yhi','zhi']]).squeeze()

        lower_bounds = self.world_coords.fixed_to_world(fixed_lower)
        upper_bounds = self.world_coords.fixed_to_world(fixed_upper)

        return np.atleast_2d(lower_bounds), np.atleast_2d(upper_bounds)
