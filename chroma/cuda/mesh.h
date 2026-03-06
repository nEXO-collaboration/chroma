#ifndef __MESH_H__
#define __MESH_H__

#include "intersect.h"
#include "geometry.h"

#include "stdio.h"

#define STACK_SIZE 1000

/* Tests the intersection between a ray and a node in the bounding volume
   hierarchy. If the ray intersects the bounding volume and `min_distance`
   is less than zero or the distance from `origin` to the intersection is
   less than `min_distance`, return true, else return false. */
__device__ bool
intersect_node(const float3 &neg_origin_inv_dir, const float3 &inv_dir,
	       Geometry *g, const Node &node, const float min_distance=-1.0f)
{
    float distance_to_box;

    if (intersect_box(neg_origin_inv_dir, inv_dir, node.lower, node.upper,
		      distance_to_box)) {
	if (min_distance < 0.0f)
	    return true;

	if (distance_to_box > min_distance)
	    return false;

	return true;
    }
    else {
	return false;
    }
}

__device__ float3
triangle_surface_normal(Geometry *g, int triangle_id)
{
    Triangle t = get_triangle(g, triangle_id);
    float3 edge1 = t.v1 - t.v0;
    float3 edge2 = t.v2 - t.v1;
    return normalize(cross(edge1, edge2));
}

/* Finds the intersection between a ray and `geometry`. If the ray does
   intersect the mesh and the index of the intersected triangle is not equal
   to `last_hit_triangle`, set `min_distance` to the distance from `origin` to
   the intersection and return the index of the triangle which the ray
   intersected, else return -1. */
__device__ int
intersect_mesh(const float3 &origin, const float3& direction, Geometry *g,
	       float &min_distance, int last_hit_triangle = -1,
	       float3 *surface_normal = NULL)
{
    int triangle_index = -1;

    float distance;
    min_distance = -1.0f;
    unsigned int material_code = 0;

    if (surface_normal != NULL)
        *surface_normal = make_float3(0.0f, 0.0f, 0.0f);

    Node root = get_node(g, 0);

    float3 neg_origin_inv_dir = -origin / direction;
    float3 inv_dir = 1.0f / direction;

    if (!intersect_node(neg_origin_inv_dir, inv_dir, g, root, min_distance))
	return -1;

    unsigned int child_ptr_stack[STACK_SIZE];
    unsigned int nchild_ptr_stack[STACK_SIZE];
    child_ptr_stack[0] = root.child;
    nchild_ptr_stack[0] = root.nchild;

    int curr = 0;

    unsigned int count = 0;
    unsigned int tri_count = 0;

    while (curr >= 0) {
	unsigned int first_child = child_ptr_stack[curr];
	unsigned int nchild = nchild_ptr_stack[curr];
	curr--;

	for (unsigned int i=first_child; i < first_child + nchild; i++) {
	    Node node = get_node(g, i);
	    count++;

	    if (intersect_node(neg_origin_inv_dir, inv_dir, g, node, min_distance)) {

		if (node.nchild == 0) { /* leaf node */
		    // This node wraps a triangle

		    if (node.child != last_hit_triangle) {
			// Can't hit same triangle twice in a row
			tri_count++;
			Triangle t = get_triangle(g, node.child);			
			if (intersect_triangle(origin, direction, t, distance)) {
			    const float distance_tolerance =
			        fmaxf(1.0e-4f, 1.0e-5f * fmaxf(fabsf(distance), fabsf(min_distance)));

			    if (triangle_index == -1 || distance < min_distance - distance_tolerance) {
				triangle_index = node.child;
				min_distance = distance;
                                material_code = g->material_codes[node.child];
                                if (surface_normal != NULL) {
                                    *surface_normal = triangle_surface_normal(g, node.child);
                                }
			    } // if hit triangle is closer than previous hits
                            else if (surface_normal != NULL &&
                                     triangle_index != -1 &&
                                     fabsf(distance - min_distance) <= distance_tolerance &&
                                     g->material_codes[node.child] == material_code) {
                                float3 candidate_normal = triangle_surface_normal(g, node.child);
                                if (dot(candidate_normal, *surface_normal) < 0.0f)
                                    candidate_normal = -candidate_normal;
                                *surface_normal += candidate_normal;
                            }

			} // if hit triangle
			
		    } // if not hitting same triangle as last step

		} else {
		    curr++;
		    child_ptr_stack[curr] = node.child;
		    nchild_ptr_stack[curr] = node.nchild;
		} // leaf or internal node?
	    } // hit node?
	    
	    if (curr >= STACK_SIZE) {
	    	printf("warning: intersect_mesh() aborted; node > tail\n");
	    	break;
	    }
	} // loop over children, starting with first_child

    } // while nodes on stack

    //if (blockIdx.x == 0 && threadIdx.x == 0) {
    //  printf("node count: %d\n", count);
    //  printf("triangle count: %d\n", tri_count);
    //}

    if (triangle_index != -1 && surface_normal != NULL) {
        float normal_length = norm(*surface_normal);
        if (normal_length > CHROMA_EPSILON)
            *surface_normal /= normal_length;
    }

    return triangle_index;
}

extern "C"
{

__global__ void
distance_to_mesh(int nthreads, float3 *_origin, float3 *_direction,
		 Geometry *g, float *_distance)
{
    __shared__ Geometry sg;

    if (threadIdx.x == 0)
	sg = *g;

    __syncthreads();

    int id = blockIdx.x*blockDim.x + threadIdx.x;

    if (id >= nthreads)
	return;

    g = &sg;

    float3 origin = _origin[id];
    float3 direction = _direction[id];
    direction /= norm(direction);

    float distance;

    int triangle_index = intersect_mesh(origin, direction, g, distance);

    if (triangle_index != -1)
	_distance[id] = distance;
}

__global__ void
color_solids(int first_triangle, int nthreads, int *solid_id_map,
	     bool *solid_hit, unsigned int *solid_colors, Geometry *g)
{
    int id = blockIdx.x*blockDim.x + threadIdx.x;

    if (id >= nthreads)
	return;

    int triangle_id = first_triangle + id;
    int solid_id = solid_id_map[triangle_id];
    if (solid_hit[solid_id])
	g->colors[triangle_id] = solid_colors[solid_id];
}

} // extern "C"

#endif
