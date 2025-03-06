#ifndef __GEOMETRY_TYPES_H__
#define __GEOMETRY_TYPES_H__

struct Material
{
    float *refractive_index;
    float *absorption_length; 
    float *scattering_length; 
    float **comp_reemission_prob; 
    float **comp_reemission_wvl_cdf;
    float **comp_reemission_time_cdf;
    float **comp_absorption_length;
    unsigned int num_comp;
    unsigned int wavelength_n;
    float wavelength_step;
    float wavelength_start;
    unsigned int time_n;
    float time_step;
    float time_start;
};

enum { SURFACE_DEFAULT,
       SURFACE_COMPLEX,
       SURFACE_WLS,
       SURFACE_DICHROIC, 
       SURFACE_DIELECTRIC_METAL,
       SURFACE_SIPM_EMPIRICAL};

struct DichroicProps
{
    float *angles;
    float **dichroic_reflect;
    float **dichroic_transmit;
    unsigned int nangles;
};

struct SiPMEmpiricalProps
{
    float *angles;
    float **sipmEmpirical_reflect;
    float **sipmEmpirical_transmit;  // New field for transmission data
    float **sipmEmpirical_relativePDE;
    float diffuseRefl;  
    float FillFactor;  // Changed from Absorption to FillFactor
    unsigned int nangles;
};

struct Surface
{
    float *detect;
    float *absorb;
    float *reemit;
    float *reflect_diffuse;
    float *reflect_specular;
    float *eta;
    float *k;
    float *reemission_cdf;
    DichroicProps *dichroic_props;
    SiPMEmpiricalProps *sipmEmpirical_props;
 
    unsigned int model;
    unsigned int wavelength_n;
    unsigned int transmissive;
    float wavelength_step;
    float wavelength_start;
    float thickness;
};

struct Triangle
{
    float3 v0, v1, v2;
};

enum { INTERNAL_NODE, LEAF_NODE, PADDING_NODE };
const unsigned int CHILD_BITS = 28;
const unsigned int NCHILD_MASK = (0xFFFFu << CHILD_BITS);

struct Node
{
    float3 lower;
    float3 upper;
    unsigned int child;
    unsigned int nchild;
};

struct Geometry
{
    float3 *vertices;
    uint3 *triangles;
    unsigned int *material_codes;
    unsigned int *colors;
    uint4 *primary_nodes;
    uint4 *extra_nodes;
    Material **materials;
    Surface **surfaces;
    float3 world_origin;
    float world_scale;
    int nprimary_nodes;
};

#endif

