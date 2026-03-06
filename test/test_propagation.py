from .unittest_find import unittest
import tempfile
import numpy as np

from chroma.geometry import Mesh, Solid, Geometry, Surface, vacuum
from chroma.loader import create_geometry_from_obj
from chroma.make import box
from chroma.sim import Simulation
from chroma.event import NO_HIT, Photons
from chroma.tools import count_nonzero

class TestPropagation(unittest.TestCase):
    def setUp(self):
        cache_dir = tempfile.TemporaryDirectory()
        self.addCleanup(cache_dir.cleanup)
        self.cache_dir = cache_dir.name

    def testAbort(self):
        '''Photons that hit a triangle at normal incidence should not abort.

        Photons that hit a triangle at exactly normal incidence can sometimes
        produce a dot product that is outside the range allowed by acos().
        Trigger these with axis aligned photons in a box.
        '''

        # Setup geometry
        cube = Geometry(vacuum)
        cube.add_solid(Solid(box(100,100,100), vacuum, vacuum))
        geo = create_geometry_from_obj(cube, update_bvh_cache=False,
                                       read_bvh_cache=False,
                                       cache_dir=self.cache_dir)
        sim = Simulation(geo, geant4_processes=0)

        # Create initial photons
        nphotons = 10000
        pos = np.tile([0,0,0], (nphotons,1)).astype(np.float32)
        dir = np.tile([0,0,1], (nphotons,1)).astype(np.float32)
        pol = np.zeros_like(pos)
        phi = np.random.uniform(0, 2*np.pi, nphotons).astype(np.float32)
        pol[:,0] = np.cos(phi)
        pol[:,1] = np.sin(phi)
        t = np.zeros(nphotons, dtype=np.float32)
        wavelengths = np.empty(nphotons, np.float32)
        wavelengths.fill(400.0)

        photons = Photons(pos=pos, dir=dir, pol=pol, t=t,
                          wavelengths=wavelengths)

        # First make one step to check for strangeness
        photons_end = next(sim.simulate([photons], keep_photons_end=True,
                                   max_steps=1)).photons_end

        self.assertFalse(np.isnan(photons_end.pos).any())
        self.assertFalse(np.isnan(photons_end.dir).any())
        self.assertFalse(np.isnan(photons_end.pol).any())
        self.assertFalse(np.isnan(photons_end.t).any())
        self.assertFalse(np.isnan(photons_end.wavelengths).any())

        # Now let it run the usual ten steps
        photons_end = next(sim.simulate([photons], keep_photons_end=True,
                                   max_steps=10)).photons_end
        aborted = (photons_end.flags & (1 << 31)) > 0
        print('aborted photons: %1.1f' % \
            (float(count_nonzero(aborted)) / nphotons))
        self.assertFalse(aborted.any())

    def testJoinedReflectingPlanesReflectAtSharedEdge(self):
        '''Photons aimed exactly at a reflective seam should bounce back into the wedge.'''

        vertices = np.array([
            [0.0, 0.0, -50.0],
            [0.0, 0.0,  50.0],
            [0.0, 50.0, -50.0],
            [0.0, 50.0,  50.0],
            [50.0, 0.0, -50.0],
            [50.0, 0.0,  50.0],
        ], dtype=np.float32)
        triangles = np.array([
            [0, 1, 2], [1, 3, 2],
            [0, 4, 1], [1, 4, 5],
        ], dtype=np.int32)

        mirror = Surface('mirror')
        mirror.set('reflect_specular', 1.0)
        surfaces = np.array([mirror] * len(triangles), dtype=object)
        joined_planes = Geometry(vacuum)
        joined_planes.add_solid(Solid(Mesh(vertices, triangles,
                                           remove_null_triangles=False),
                                      vacuum, vacuum, surface=surfaces))

        geo = create_geometry_from_obj(joined_planes, update_bvh_cache=False,
                                       read_bvh_cache=False,
                                       cache_dir=self.cache_dir)
        sim = Simulation(geo, geant4_processes=0)

        nphotons = 4096
        positions = np.tile([10.0, 10.0, 0.0], (nphotons, 1)).astype(np.float32)
        direction = np.array([-1.0, -1.0, 0.0], dtype=np.float32)
        direction /= np.linalg.norm(direction)
        directions = np.tile(direction, (nphotons, 1)).astype(np.float32)
        polarizations = np.tile([0.0, 0.0, 1.0], (nphotons, 1)).astype(np.float32)
        times = np.zeros(nphotons, dtype=np.float32)
        wavelengths = np.full(nphotons, 400.0, dtype=np.float32)

        edge_hit_photons = Photons(pos=positions, dir=directions,
                                   pol=polarizations, t=times,
                                   wavelengths=wavelengths)
        photons_end = next(sim.simulate([edge_hit_photons], keep_photons_end=True,
                                        max_steps=1)).photons_end

        self.assertFalse(((photons_end.flags & NO_HIT) != 0).any())
        self.assertTrue((photons_end.dir[:, 0] > 0.0).all())
        self.assertTrue((photons_end.dir[:, 1] > 0.0).all())

        off_edge_positions = np.tile([10.0, 11.0, 0.0], (nphotons, 1)).astype(np.float32)
        off_edge_photons = Photons(pos=off_edge_positions, dir=directions,
                                   pol=polarizations, t=times,
                                   wavelengths=wavelengths)
        off_edge_end = next(sim.simulate([off_edge_photons], keep_photons_end=True,
                                         max_steps=1)).photons_end

        self.assertFalse(((off_edge_end.flags & NO_HIT) != 0).any())
        self.assertTrue((off_edge_end.dir[:, 0] > 0.0).all())
        self.assertTrue((off_edge_end.dir[:, 1] < 0.0).all())
