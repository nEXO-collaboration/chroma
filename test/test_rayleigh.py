from .unittest_find import unittest
import copy
import tempfile
import numpy as np
from scipy import stats

from chroma.geometry import Solid, Geometry
from chroma.loader import create_geometry_from_obj
from chroma.make import box
from chroma.sim import Simulation
from chroma.demo.optics import water
from chroma.event import Photons, RAYLEIGH_SCATTER

class TestRayleigh(unittest.TestCase):
    def setUp(self):
        cache_dir = tempfile.TemporaryDirectory()
        self.addCleanup(cache_dir.cleanup)
        self.cache_dir = cache_dir.name

        self.medium = copy.deepcopy(water)
        self.medium.name = 'rayleigh_test_water'
        self.medium.set('scattering_length', 10.0)

        self.cube = Geometry(self.medium)
        self.cube.add_solid(Solid(box(100,100,100), self.medium, self.medium))
        self.geo = create_geometry_from_obj(self.cube, update_bvh_cache=False,
                                            read_bvh_cache=False,
                                            cache_dir=self.cache_dir)
        self.sim = Simulation(self.geo, geant4_processes=0)

        nphotons = 100000
        pos = np.tile([0,0,0], (nphotons,1)).astype(np.float32)
        dir = np.tile([0,0,1], (nphotons,1)).astype(np.float32)
        pol = np.zeros_like(pos)
        phi = np.random.uniform(0, 2*np.pi, nphotons).astype(np.float32)
        pol[:,0] = np.cos(phi)
        pol[:,1] = np.sin(phi)
        t = np.zeros(nphotons, dtype=np.float32)
        wavelengths = np.empty(nphotons, np.float32)
        wavelengths.fill(400.0)

        self.photons = Photons(pos=pos, dir=dir, pol=pol, t=t, wavelengths=wavelengths)

    def testAngularDistributionPolarized(self):
        # Fully polarized photons
        self.photons.pol[:] = [1.0, 0.0, 0.0]

        photons_end = next(self.sim.simulate([self.photons], keep_photons_end=True, max_steps=1)).photons_end
        aborted = (photons_end.flags & (1 << 31)) > 0
        self.assertFalse(aborted.any())

        # Compute the dot product between initial and final dir
        rayleigh_scatters = (photons_end.flags & RAYLEIGH_SCATTER) > 0
        self.assertGreater(rayleigh_scatters.sum(), 1000)

        cos_scatter = (self.photons.dir[rayleigh_scatters] *
                       photons_end.dir[rayleigh_scatters]).sum(axis=1)
        cos_scatter = np.clip(cos_scatter, -1.0, 1.0)

        # For polarized light the scattering-angle PDF is proportional to
        # (1 + cos(theta)^2) * sin(theta), so the CDF in u = cos(theta) is
        # (u^3 + 3u + 4) / 8 over [-1, 1].
        ks_result = stats.kstest(
            cos_scatter,
            lambda u: np.clip((u**3 + 3.0*u + 4.0) / 8.0, 0.0, 1.0),
        )
        self.assertGreater(ks_result.pvalue, 1e-3)

        scattered_dirs = photons_end.dir[rayleigh_scatters]
        scattered_pols = photons_end.pol[rayleigh_scatters]
        self.assertLess(np.abs(np.sum(scattered_dirs * scattered_pols, axis=1)).max(), 1e-5)
        self.assertTrue(np.allclose(np.linalg.norm(scattered_dirs, axis=1), 1.0, atol=1e-5))
        self.assertTrue(np.allclose(np.linalg.norm(scattered_pols, axis=1), 1.0, atol=1e-5))
