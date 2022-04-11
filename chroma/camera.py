#!/usr/bin/env python
import numpy as np
import itertools
import threading
import multiprocessing
import os
from subprocess import call
import shutil
import tempfile
import sys

import pycuda.driver as cuda
from pycuda import gpuarray as ga

from chroma.geometry import Mesh, Solid, Geometry, vacuum
from chroma.transform import rotate, make_rotation_matrix
from chroma.sample import uniform_sphere
from chroma.tools import from_film
import chroma.event as event
from chroma import make
from chroma import gpu
from chroma.loader import create_geometry_from_obj
import pygame
from pygame.locals import *

from timeit import default_timer as timer

def bvh_mesh(geometry, layer):
    lower_bounds, upper_bounds = geometry.bvh.get_layer(layer).get_bounds()

    if len(lower_bounds) == 0 or len(upper_bounds) == 0:
        raise Exception('no nodes at layer %i' % layer)

    dx, dy, dz = upper_bounds[0] - lower_bounds[0]
    center = np.mean([upper_bounds[0],lower_bounds[0]], axis=0)

    geometry = Geometry()
    
    geometry.add_solid(Solid(make.box(dx,dy,dz,center), vacuum, vacuum, color=0x33ffffff))

    for center, dx, dy, dz in list(zip(np.mean([lower_bounds,upper_bounds],axis=0),
                                  *list(zip(*upper_bounds-lower_bounds))))[1:]:
        geometry.add_solid(Solid(make.box(dx,dy,dz,center), vacuum, vacuum, color=0x33ffffff))
    
    return create_geometry_from_obj(geometry)

def encode_movie(dir):
    root, ext = 'movie', 'avi'
    for i in itertools.count():
        path = '.'.join([root + str(i).zfill(5), ext])

        if not os.path.exists(path):
            break

    call(['mencoder', 'mf://' + dir + '/*.png', '-mf', 'fps=10', '-o',
          path, '-ovc', 'xvid', '-xvidencopts', 'bitrate=3000'])

    shutil.rmtree(dir)

    print('movie saved to %s.' % path)

class Camera(multiprocessing.Process):
    "The camera class is used to render a Geometry object."
    def __init__(self, geometry, size=(800,600), device_id=None, background=0x00000000):
        '''
        background is a 0xAARRGGBB 32bit color to use for points at infinity. alpha=255 is opaque
        '''
        super(Camera, self).__init__()
        self.geometry = geometry
        self.device_id = device_id
        self.size = size
        self.background = background

        self.bvh_layer_count = len(self.geometry.bvh.layer_offsets)
        self.currentlayer = None
        self.bvh_layers = {}

        self.display3d = False
        self.green_magenta = False
        self.max_alpha_depth = 50
        self.alpha_depth = 10

        try: 
            import spnav as spnav_module
            self.spnav_module = spnav_module
            self.spnav = True
        except:
            self.spnav = False

    def init_gpu(self):
        self.context = gpu.create_cuda_context(self.device_id)

        self.gpu_geometry = gpu.GPUGeometry(self.geometry)
        self.gpu_funcs = gpu.GPUFuncs(gpu.get_cu_module('mesh.h'))
        self.hybrid_funcs = gpu.GPUFuncs(gpu.get_cu_module('hybrid_render.cu'))

        self.gpu_geometries = [self.gpu_geometry]

        self.width, self.height = self.size

        self.npixels = self.width*self.height

        self.clock = pygame.time.Clock()

        self.doom_mode = False
        
        try:
            if self.width == 640: # SECRET DOOM MODE!
                print('shotgun activated!')
                self.doom_hud = pygame.image.load('images/doomhud.png').convert_alpha()
                rect = self.doom_hud.get_rect()
                self.doom_rect = rect.move(0, self.height - rect.height)
                self.doom_mode = True
        except:
            pass

        lower_bound, upper_bound = self.geometry.mesh.get_bounds()

        self.mesh_diagonal_norm = np.linalg.norm(upper_bound-lower_bound)

        self.scale = self.mesh_diagonal_norm

        self.motion = 'coarse'

        self.nblocks = 64

        self.point = np.array([(lower_bound[0]+upper_bound[0])/2, -self.mesh_diagonal_norm,
                               (lower_bound[2]+upper_bound[2])/2])

        self.axis1 = np.array([0,0,1], float)
        self.axis2 = np.array([1,0,0], float)

        self.film_width = 35.0 # mm

        pos, dir = from_film(self.point, axis1=self.axis1, axis2=self.axis2,
                             size=self.size, width=self.film_width)
                            
        self.rays = gpu.GPURays(pos, dir, max_alpha_depth=self.max_alpha_depth)
        
        self.pixels_gpu = ga.empty(self.npixels, dtype=np.uint32)

        self.movie = False
        self.movie_index = 0
        self.movie_dir = None
        self.hybrid_render = False

    def disable3d(self):
        pos, dir = from_film(self.point, axis1=self.axis1, axis2=self.axis2,
                             size=self.size, width=self.film_width)

        self.rays = gpu.GPURays(pos, dir, max_alpha_depth=self.max_alpha_depth)
        
        self.display3d = False

    def enable3d(self):
        self.point1 = self.point-(self.mesh_diagonal_norm/60)*self.axis2
        self.point2 = self.point+(self.mesh_diagonal_norm/60)*self.axis2

        self.viewing_angle = 0.0

        pos1, dir1 = from_film(self.point1, axis1=self.axis1, axis2=self.axis2,
                               size=self.size, width=self.film_width)
        pos2, dir2 = from_film(self.point2, axis1=self.axis1, axis2=self.axis2,
                               size=self.size, width=self.film_width)

        self.rays1 = gpu.GPURays(pos1, dir1,
                                 max_alpha_depth=self.max_alpha_depth)
        self.rays2 = gpu.GPURays(pos2, dir2,
                                 max_alpha_depth=self.max_alpha_depth)

        scope_size = (self.size[0]//4, self.size[0]//4)

        scope_pos, scope_dir = from_film(self.point, axis1=self.axis1,
                                         axis2=self.axis2, size=scope_size,
                                         width=self.film_width/4.0)

        self.scope_rays = gpu.GPURays(scope_pos, scope_dir)

        self.scope_pixels_gpu = ga.empty(self.scope_rays.pos.size, dtype=np.uint32)

        self.pixels1_gpu = ga.empty(self.width*self.height, dtype=np.uint32)
        self.pixels2_gpu = ga.empty(self.width*self.height, dtype=np.uint32)
        
        self.distances_gpu = ga.empty(self.scope_rays.pos.size,
                                      dtype=np.float32)
        self.display3d = True

    def initialize_render(self):
        self.rng_states_gpu = gpu.get_rng_states(self.npixels)
        self.xyz_lookup1_gpu = ga.zeros(len(self.geometry.mesh.triangles),
                                        dtype=ga.vec.float3)
        self.xyz_lookup2_gpu = ga.zeros(len(self.geometry.mesh.triangles),
                                        dtype=ga.vec.float3)

        if self.display3d:
            self.image1_gpu = ga.zeros(self.npixels, dtype=ga.vec.float3)
            self.image2_gpu = ga.zeros(self.npixels, dtype=ga.vec.float3)
        else:
            self.image_gpu = ga.zeros(self.npixels, dtype=ga.vec.float3)

        self.source_position = self.point

        self.nimages = 0
        self.nlookup_calls = 0
        self.max_steps = 10

    def clear_xyz_lookup(self):
        self.xyz_lookup1_gpu.fill(ga.vec.make_float3(0.0,0.0,0.0))
        self.xyz_lookup2_gpu.fill(ga.vec.make_float3(0.0,0.0,0.0))

        self.nlookup_calls = 0

    def update_xyz_lookup(self, source_position):
        for wavelength, rgb_tuple in \
                zip([685.0, 545.0, 445.0],[(1,0,0),(0,1,0),(0,0,1)]):
            for i in range(self.xyz_lookup1_gpu.size//(self.npixels)+1):
                self.hybrid_funcs.update_xyz_lookup(np.int32(self.npixels), np.int32(self.xyz_lookup1_gpu.size), np.int32(i*self.npixels), ga.vec.make_float3(*source_position), self.rng_states_gpu, np.float32(wavelength), ga.vec.make_float3(*rgb_tuple), self.xyz_lookup1_gpu, self.xyz_lookup2_gpu, np.int32(self.max_steps), self.gpu_geometry.gpudata, block=(self.nblocks,1,1), grid=(self.npixels//self.nblocks+1,1))

        self.nlookup_calls += 1

    def clear_image(self):
        if self.display3d:
            self.image1_gpu.fill(ga.vec.make_float3(0.0,0.0,0.0))
            self.image2_gpu.fill(ga.vec.make_float3(0.0,0.0,0.0))
        else:
            self.image_gpu.fill(ga.vec.make_float3(0.0,0.0,0.0))

        self.nimages = 0

    def update_image_from_rays(self, image, rays):
        for wavelength, rgb_tuple in \
                zip([685.0, 545.0, 445.0],[(1,0,0),(0,1,0),(0,0,1)]):
            self.hybrid_funcs.update_xyz_image(np.int32(rays.pos.size), self.rng_states_gpu, rays.pos, rays.dir, np.float32(wavelength), ga.vec.make_float3(*rgb_tuple), self.xyz_lookup1_gpu, self.xyz_lookup2_gpu, image, np.int32(self.nlookup_calls), np.int32(self.max_steps), self.gpu_geometry.gpudata, block=(self.nblocks,1,1), grid=(rays.pos.size//self.nblocks+1,1))

    def update_image(self):
        if self.display3d:
            self.update_image_from_rays(self.image1_gpu, self.rays1)
            self.update_image_from_rays(self.image2_gpu, self.rays2)
        else:
            self.update_image_from_rays(self.image_gpu, self.rays)

        self.nimages += 1

    def process_image(self):
        if self.display3d:
            self.hybrid_funcs.process_image(np.int32(self.pixels1_gpu.size), self.image1_gpu, self.pixels1_gpu, np.int32(self.nimages), block=(self.nblocks,1,1), grid=((self.pixels1_gpu.size)//self.nblocks+1,1))
            self.hybrid_funcs.process_image(np.int32(self.pixels2_gpu.size), self.image2_gpu, self.pixels2_gpu, np.int32(self.nimages), block=(self.nblocks,1,1), grid=((self.pixels2_gpu.size)//self.nblocks+1,1))
        else:
            self.hybrid_funcs.process_image(np.int32(self.pixels_gpu.size), self.image_gpu, self.pixels_gpu, np.int32(self.nimages), block=(self.nblocks,1,1), grid=((self.pixels_gpu.size)//self.nblocks+1,1))

    def screenshot(self, dir='', start=0):
        root, ext = 'screenshot', 'png'

        for i in itertools.count(start):
            path = os.path.join(dir, '.'.join([root + str(i).zfill(5), ext]))

            if not os.path.exists(path):
                break

        try:
            pygame.image.save(self.screen, path)
        except ImportError:
            import Image
            mode = 'RGBA'
            data = self.screen.get_buffer()
            im = Image.frombuffer(mode,self.size,data,'raw',mode,0,1)
            im.save(path)

        print('image saved to %s' % path)

    def rotate(self, phi, n):
        if self.display3d:
            self.rays1.rotate(phi, n)
            self.rays2.rotate(phi, n)
            self.scope_rays.rotate(phi, n)

            self.point1 = rotate(self.point1, phi, n)
            self.point2 = rotate(self.point2, phi, n)
        else:
            self.rays.rotate(phi, n)

        self.point = rotate(self.point, phi, n)
        self.axis1 = rotate(self.axis1, phi, n)
        self.axis2 = rotate(self.axis2, phi, n)

        if self.hybrid_render:
            self.clear_image()

        self.update()

    def rotate_around_point(self, phi, n, point, redraw=True):
        self.axis1 = rotate(self.axis1, phi, n)
        self.axis2 = rotate(self.axis2, phi, n)

        if self.display3d:
            self.rays1.rotate_around_point(phi, n, point)
            self.rays2.rotate_around_point(phi, n, point)
            self.scope_rays.rotate_around_point(phi, n, point)
        else:
            self.rays.rotate_around_point(phi, n, point)

        if redraw:
            if self.hybrid_render:
                self.clear_image()

            self.update()

    def translate(self, v, redraw=True):
        self.point += v

        if self.display3d:
            self.rays1.translate(v)
            self.rays2.translate(v)
            self.scope_rays.translate(v)

            self.point1 += v
            self.point2 += v
        else:
            self.rays.translate(v)

        if redraw:
            if self.hybrid_render:
                self.clear_image()

            self.update()

    def update_pixels(self, gpu_geometry=None, keep_last_render=False):
        if gpu_geometry is None:
            gpu_geometry = self.gpu_geometry

        if self.hybrid_render:
            while self.nlookup_calls < 10:
                self.update_xyz_lookup(self.source_position)
            self.update_image()
            self.process_image()
        else:
            if self.display3d:
                self.rays1.render(gpu_geometry, self.pixels1_gpu,
                                  self.alpha_depth, keep_last_render, 
                                  bg_color=self.background)
                self.rays2.render(gpu_geometry, self.pixels2_gpu,
                                  self.alpha_depth, keep_last_render, 
                                  bg_color=self.background)
            else:
                self.rays.render(gpu_geometry, self.pixels_gpu,
                                 self.alpha_depth, keep_last_render, 
                                  bg_color=self.background)

    def update_viewing_angle(self):
        if self.display3d:
            distance_gpu = ga.empty(self.scope_rays.pos.size, dtype=np.float32)
            distance_gpu.fill(1e9)

            for i, gpu_geometry in enumerate(self.gpu_geometries):
                self.gpu_funcs.distance_to_mesh(np.int32(self.scope_rays.pos.size), self.scope_rays.pos, self.scope_rays.dir, gpu_geometry.gpudata, distance_gpu, block=(self.nblocks,1,1), grid=(self.scope_rays.pos.size//self.nblocks,1))

                if i == 0:
                    distance = distance_gpu.get()
                else:
                    distance = np.minimum(distance, distance_gpu.get())

            baseline = distance.min()

            if baseline < 1e9:
                d1 = self.point1 - self.point
                v1 = d1/np.linalg.norm(d1)
                v1 *= baseline/60 - np.linalg.norm(d1)

                self.rays1.translate(v1)

                self.point1 += v1

                d2 = self.point2 - self.point
                v2 = d2/np.linalg.norm(d2)
                v2 *= baseline/60 - np.linalg.norm(d2)

                self.rays2.translate(v2)

                self.point2 += v2

            direction = np.cross(self.axis1,self.axis2)
            direction /= np.linalg.norm(direction)
            direction1 = self.point + direction*baseline - self.point1
            direction1 /= np.linalg.norm(direction1)

            new_viewing_angle = np.arccos(direction1.dot(direction))

            phi = new_viewing_angle - self.viewing_angle

            self.rays1.rotate_around_point(phi, self.axis1, self.point1)
            self.rays2.rotate_around_point(-phi, self.axis1, self.point2)

            self.viewing_angle = new_viewing_angle

    def update(self):
        if self.display3d:
            self.update_viewing_angle()
                
        start = timer()
        
        n = len(self.gpu_geometries)
        for i, gpu_geometry in enumerate(self.gpu_geometries):
            if i == 0:
                self.update_pixels(gpu_geometry)
            else:
                self.update_pixels(gpu_geometry, keep_last_render=True)
        
        end = timer()
        #print('render',end-start)

        start = timer()
        
        if self.display3d:
            pixels1 = self.pixels1_gpu.get(pagelocked=True)
            pixels2 = self.pixels2_gpu.get(pagelocked=True)

            if self.green_magenta:
                pixels = (pixels1 & 0x00ff00) | (pixels2 & 0xff00ff)
            else:
                pixels = (pixels1 & 0xff0000) | (pixels2 & 0x00ffff)

            alpha = ((0xff & (pixels1 >> 24)) + (0xff & (pixels2 >> 24)))/2

            pixels |= (alpha << 24)
        else:
            pixels = self.pixels_gpu.get(pagelocked=True)
            
        end = timer()
        #print('get',end-start)

        pygame.surfarray.blit_array(self.screen, pixels.reshape(self.size))
        if self.doom_mode:
            self.screen.blit(self.doom_hud, self.doom_rect)
        self.window.fill(0)
        self.window.blit(self.screen, (0,0))
        pygame.display.flip()

        if self.movie:
            self.screenshot(self.movie_dir, self.movie_index)
            self.movie_index += 1

    def loadlayer(self, layer):
        if layer is None:
            self.gpu_geometries = [self.gpu_geometry]
        else:
            try:
                gpu_geometry = self.bvh_layers[layer]
            except KeyError:
                geometry = bvh_mesh(self.geometry, layer)
                gpu_geometry = gpu.GPUGeometry(geometry, print_usage=False)
                self.bvh_layers[layer] = gpu_geometry

            self.gpu_geometries = [self.gpu_geometry, gpu_geometry]

        self.update()

    def process_event(self, event):
        if event.type == MOUSEBUTTONDOWN:
            if event.button == 4:
                v = self.scale*np.cross(self.axis1,self.axis2)/10.0
                self.translate(v)

            elif event.button == 5:
                v = -self.scale*np.cross(self.axis1,self.axis2)/10.0
                self.translate(v)

            elif event.button == 1:
                mouse_position = pygame.mouse.get_rel()
                self.clicked = True

        elif event.type == MOUSEBUTTONUP:
            if event.button == 1:
                self.clicked = False

        elif event.type == MOUSEMOTION and self.clicked:
            movement = np.array(pygame.mouse.get_rel())

            if (movement == 0).all():
                return

            length = np.linalg.norm(movement)

            mouse_direction = movement[0]*self.axis2 - movement[1]*self.axis1
            mouse_direction /= np.linalg.norm(mouse_direction)

            if pygame.key.get_mods() & (KMOD_LSHIFT | KMOD_RSHIFT):
                v = -mouse_direction*self.scale*length/float(self.width)
                self.translate(v)
            else:
                phi = np.float32(2*np.pi*length/float(self.width))
                n = rotate(mouse_direction, np.pi/2,
                           np.cross(self.axis1,self.axis2))

                if pygame.key.get_mods() & KMOD_LCTRL:
                    self.rotate_around_point(phi, n, self.point)
                else:
                    self.rotate(phi, n)

        elif event.type == KEYDOWN:
            if event.key == K_LALT or event.key == K_RALT:
                if self.motion == 'coarse':
                    self.scale = self.mesh_diagonal_norm/20.0
                    self.motion = 'fine'
                elif self.motion == 'fine':
                    self.scale = self.mesh_diagonal_norm/400.0
                    self.motion = 'superfine'
                elif self.motion == 'superfine':
                    self.scale = self.mesh_diagonal_norm
                    self.motion = 'coarse'

            elif event.key == K_F6:
                self.clear_xyz_lookup()
                self.clear_image()
                self.source_position = self.point

            elif event.key == K_F7:
                for i in range(100):
                    self.update_xyz_lookup(self.point)
                self.source_position = self.point

            elif event.key == K_F11:
                pygame.display.toggle_fullscreen()

            elif event.key == K_ESCAPE:
                self.done = True
                return

            elif event.key == K_EQUALS:
                if self.alpha_depth < self.max_alpha_depth:
                    self.alpha_depth += 1
                self.update()

            elif event.key == K_MINUS:
                if self.alpha_depth > 1:
                    self.alpha_depth -= 1
                    self.update()

            elif event.key == K_PAGEDOWN:
                if self.currentlayer is None:
                    self.currentlayer = None
                else:
                    if self.currentlayer > 0:
                        self.currentlayer -= 1
                    else:
                        self.currentlayer = None

                self.loadlayer(self.currentlayer)

            elif event.key == K_PAGEUP:
                if self.currentlayer is None:
                    self.currentlayer = 0
                else:
                    if self.currentlayer < self.bvh_layer_count:
                        self.currentlayer += 1
                    else:
                        self.currentlayer = None

                self.loadlayer(self.currentlayer)                    

            elif event.key == K_3:
                if self.display3d:
                    self.disable3d()
                else:
                    self.enable3d()
                self.update()

            elif event.key == K_g:
                self.green_magenta = not self.green_magenta
                self.update()

            elif event.key == K_F12:
                self.screenshot()

            elif event.key == K_F5:
                if not hasattr(self, 'rng_states_gpu'):
                    self.initialize_render()

                self.hybrid_render = not self.hybrid_render
                self.clear_image()
                self.update()

            elif event.key == K_m:
                if self.movie:
                    encode_movie(self.movie_dir)
                    self.movie_dir = None
                    self.movie = False
                else:
                    self.movie_index = 0
                    self.movie_dir = tempfile.mkdtemp()
                    self.movie = True

        elif event.type == pygame.SYSWMEVENT and self.spnav:
            # Space Navigator controls
            spnav_event = self.spnav_module.spnav_x11_event(event.event)
            if spnav_event is None:
                return

            if spnav_event.ev_type == self.spnav_module.SPNAV_EVENT_MOTION:
                if pygame.key.get_mods() & (KMOD_LSHIFT | KMOD_RSHIFT):
                    accelerate_factor = 2.0
                else:
                    accelerate_factor = 1.0

                v1 = self.axis1
                v2 = self.axis2
                v3 = np.cross(self.axis1,self.axis2)
                
                x, y, z = spnav_event.translation
                rx, ry, rz = spnav_event.rotation

                v = v2*x + v1*y + v3*z
                v *= self.scale / 5000.0 * accelerate_factor

                self.translate(v, redraw=False)                    

                axis = -v2*rx - v1*ry - v3*rz

                if (axis != 0).any():
                    axis = axis.astype(float)
                    length = np.linalg.norm(axis)
                    angle = length * 0.0001 * accelerate_factor
                    axis /= length
                    self.rotate_around_point(angle, axis, self.point,
                                             redraw=False)

                if self.hybrid_render:
                    self.clear_image()

                self.update()
                pygame.event.clear(pygame.SYSWMEVENT)

            elif spnav_event.ev_type == self.spnav_module.SPNAV_EVENT_BUTTON:
                if spnav_event.bnum == 0 and spnav_event.press:
                    if not hasattr(self, 'rng_states_gpu'):
                        self.initialize_render()

                    self.hybrid_render = not self.hybrid_render
                    self.clear_image()
                    self.update()
                    pygame.event.clear(pygame.SYSWMEVENT)

        elif event.type == pygame.QUIT:
            self.done = True
            return

    def run(self):
        pygame.init()
        self.window = pygame.display.set_mode(self.size)
        self.screen = pygame.Surface(self.size, pygame.SRCALPHA)
        pygame.display.set_caption('')
        
        self.init_gpu()
        #makes things significantly faster somehow
        self.rotate(0.001,[1/np.sqrt(2),0,1/np.sqrt(2)])
        
        if self.spnav:
            try:
                wm_info = pygame.display.get_wm_info()
                self.spnav_module.spnav_x11_open(wm_info['display'],
                                                 wm_info['window'])
                pygame.event.set_allowed(pygame.SYSWMEVENT)
                #print 'Space Navigator support enabled.'
            except:
                self.spnav = False

        self.update()

        self.done = False
        self.clicked = False

        while not self.done:
            self.clock.tick(20)

            if self.hybrid_render and not self.clicked and \
                    not pygame.event.peek(KEYDOWN):
                self.update()
            
            # Grab only last SYSWMEVENT (SpaceNav!) to avoid lagged controls
            for event in pygame.event.get(pygame.SYSWMEVENT)[-1:] + \
                    pygame.event.get():                
                self.process_event(event)

        if self.movie:
            encode_movie(self.movie_dir)

        pygame.display.quit()
        if self.spnav:
            self.spnav_module.spnav_close()

        self.context.pop()

def gen_rot(a,b):
    '''Construct a matrix to rotate vector a to vector b'''
    a = a/np.linalg.norm(a)
    b = b/np.linalg.norm(b)
    if (a==b).all():
        return np.diag([1.,1.,1.])
    if (a==-b).all():
        return np.diag([-1.,-1.,-1.])
    v = np.cross(a,b)
    c = np.arccos(-np.dot(a,b))
    return make_rotation_matrix(c,v)

class RevIter:
    def __init__(self,l):
        self.l = l
        self.i = 0
    def __next__(self):
        n = self.l[self.i]
        self.i += 1
        if self.i >= len(self.l):
            self.i = len(self.l-1)
        return n
    def __len__(self):
        return len(self.l)
    def __iter__(self):
        return iter(self.l)
    def prev(self):
        self.i -= 1
        if self.i < 0:
            self.i = 0
        return self.l[self.i]
        
class EventViewer(Camera):

    def __init__(self, geometry, filename, **kwargs):
        Camera.__init__(self, geometry, **kwargs)
        if type(filename) is str:
            # This is really slow, so we do it here in the constructor to 
            # avoid slowing down the import of this module
            from chroma.io.root import RootReader
            self.rr = RootReader(filename)
        else:
            self.rr = RevIter(filename)
        self.ev = next(self.rr)
        self.display_mode_iter = itertools.cycle(['geo','charge','time','hit','dichroicon'])
        self.display_mode = next(self.display_mode_iter)
        self.sum_mode = False
        self.photon_display_mode_iter = itertools.cycle(['none','beg','end'])
        self.photon_display_mode = next(self.photon_display_mode_iter)
        self.track_display_mode_iter = itertools.cycle(['none','geant4','chroma','both'])
        self.track_display_mode = next(self.track_display_mode_iter)
        
        
        ''' photons_max will randomly select at most that many photons
            photons_max_steps truncates all photon tracks to that number of steps
            photons_only_type can be set to 'cher', 'scint', or 'reemit' 
            photons_detected_only will show only detected photon tracks 
            photons_track_size controls the track size of the photons'''
        self.photons_max = 1000
        self.photons_max_steps = 20
        self.photons_only_type = None
        self.photons_detected_only = False
        self.photons_track_size = 0.1

    def render_photon_track(self,geometry,photon_track,sz=1.0,color='wavelength'):
        origin = photon_track.pos[:-1]
        extent = photon_track.pos[1:]-photon_track.pos[:-1]
        perp1 = np.cross(origin,extent)
        perp1 = np.inner(sz/2.0/np.linalg.norm(perp1,axis=1),perp1.T).T
        perp2 = np.cross(perp1,extent)
        perp2 = np.inner(sz/2.0/np.linalg.norm(perp2,axis=1),perp2.T).T
        verts = [perp1+perp2,-perp1+perp2,perp1-perp2,-perp1-perp2]
        bot = [vert+origin for vert in verts]
        top = [vert+origin+extent for vert in verts]
        vertices = [origin,origin+extent,bot[0],top[0],bot[1],top[1],bot[2],top[2],bot[3],top[3]]
        vertices = np.transpose(np.asarray(vertices,np.float32),(1,0,2))
        triangles = np.asarray([[1, 3, 5], [1, 5, 7], [1, 7, 9], [1, 9, 3], [3, 2, 4], [5, 4, 6], [7, 6, 8], [9, 8, 2], [2, 0, 0], [4, 0, 0], [6, 0, 0], [8, 0, 0],
                                [1, 5, 1], [1, 7, 1], [1, 9, 1], [1, 3, 1], [3, 4, 5], [5, 6, 7], [7, 8, 9], [9, 2, 3], [2, 0, 4], [4, 0, 6], [6, 0, 8], [8, 0, 2]],
                                dtype=np.int32)
        if color == 'wavelength':
            r = np.asarray(np.interp(photon_track.wavelengths[:-1],[300,550,800],[0,0,255]),dtype=np.uint32)
            g = np.asarray(np.interp(photon_track.wavelengths[:-1],[300,550,800],[0,255,0]),dtype=np.uint32)
            b = np.asarray(np.interp(photon_track.wavelengths[:-1],[300,550,800],[255,0,0]),dtype=np.uint32)
            colors = np.bitwise_or(b,np.bitwise_or(np.left_shift(g,8),np.left_shift(r,16)))
        else:
            r,g,b = color
            colors = np.full_like(photon_track.wavelengths[:-1],((r<<16)|(g<<8)|b),dtype=np.uint32)
        markers = [Solid(Mesh(v,triangles), vacuum, vacuum, color=c) for v,c in zip(vertices,colors)]
        [geometry.add_solid(marker) for marker in markers]

    def render_vertex(self,geometry,vertex,children=2,sz=5.0,colors={'mu+':0x50E0FF,'mu-':0xFFE050,'e+':0x0000FF,'e-':0xFF0000,'gamma':0x00FF00}):
        steps = np.vstack((vertex.steps.x,vertex.steps.y,vertex.steps.z)).T
        for index in range(len(steps)-1):
            vec = steps[index+1]-steps[index]
            mag = np.linalg.norm(vec)
            u = vec/mag
            axis = np.cross(u,[0,0,1])
            ang = np.arccos(np.dot(u,[0,0,1]))
            rotmat = make_rotation_matrix(ang,axis)
            x = sz/2
            y = x*np.sqrt(3)/2
            segment = make.linear_extrude([-x,0,x], [-y,y,-y], mag,[0,0,0],[0,0,0])
            segment.vertices[:,2] += mag/2.0
            marker = Solid(segment, vacuum, vacuum, color=colors[vertex.particle_name] if vertex.particle_name in colors else 0xAAAAAA)
            geometry.add_solid(marker, displacement=steps[index], rotation=rotmat)
            
        if children and vertex.children:
            for child in vertex.children:
                if isinstance(children,bool):
                    self.render_vertex(geometry,child,children)
                else:
                    self.render_vertex(geometry,child,children-1)

    def render_photons(self,geometry,photons):
        x = 10.0
        h = x*np.sqrt(3)/2
        pyramid = make.linear_extrude([-x/2,0,x/2], [-h/2,h/2,-h/2], h,
                                      [0]*3, [0]*3)
        marker = Solid(pyramid, vacuum, vacuum)
                

        sample_factor = 1
        subset = photons[::sample_factor]
        for p,d,w in zip(subset.pos,subset.dir,subset.wavelengths):
            geometry.add_solid(marker, displacement=p, rotation=gen_rot([0,1,0],d))


    def render_mc_info(self):
        #need to render photon tracking info if available
        
        self.gpu_geometries = [self.gpu_geometry]
        if self.sum_mode or self.ev is None:
            return
 
        if self.photon_display_mode == 'beg':
            photons = self.ev.photons_beg
        elif self.photon_display_mode == 'end':
            photons = self.ev.photons_end
        else:
            photons = None
        
        if photons is not None:
            geometry = Geometry()
            self.render_photons(geometry,photons)
            geometry = create_geometry_from_obj(geometry)
            gpu_geometry = gpu.GPUGeometry(geometry)
            self.gpu_geometries.append(gpu_geometry)
            
        if self.track_display_mode in ['geant4', 'both'] and self.ev.vertices is not None:
            geometry = Geometry()
            any = False
            for vertex in self.ev.vertices:
                if vertex.steps:
                    any = True
                    self.render_vertex(geometry,vertex,children=True)
            if any:
                geometry = create_geometry_from_obj(geometry)
                gpu_geometry = gpu.GPUGeometry(geometry)
                self.gpu_geometries.append(gpu_geometry)
                
        
        if self.track_display_mode in ['chroma', 'both'] and self.ev.photon_tracks is not None:
            geometry = Geometry()
            print('Total Photons',len(self.ev.photon_tracks))
            
            def has(flags,test):
                return flags & test == test
            
            tracks = self.ev.photon_tracks
            if self.photons_detected_only:
                detected = np.asarray([has(track.flags[-1],event.SURFACE_DETECT) for track in tracks])
                tracks = [t for t,m in zip(tracks,detected) if m]
            cherenkov = np.asarray([has(track.flags[0],event.CHERENKOV) and not has(track.flags[-1],event.BULK_REEMIT) for track in tracks])
            scintillation = np.asarray([has(track.flags[0],event.SCINTILLATION) and not has(track.flags[-1],event.BULK_REEMIT) for track in tracks])
            reemission = np.asarray([has(track.flags[-1],event.BULK_REEMIT) for track in tracks])
            if self.photons_only_type is not None:
                if self.photons_only_type == 'cher':
                    selector = cherenkov
                elif self.photons_only_type == 'scint':
                    selector = scintillation
                elif self.photons_only_type == 'reemit':
                    selector = reemission
                else:
                    raise Exception('Unknown only type: %s'%only)
                tracks = [t for t,m in zip(tracks,selector) if m]
                cherenkov = cherenkov[selector]
                scintillation = scintillation[selector]
                reemission = reemission[selector]
            nphotons = len(tracks)
            prob = self.photons_max/nphotons if self.photons_max is not None else 1.0
            selector = np.random.random(len(tracks)) < prob
            nphotons = np.count_nonzero(selector)
            for i,track in ((i,t) for i,(s,t) in enumerate(zip(selector,tracks)) if s):
                if cherenkov[i]:
                    color = [255,0,0]
                elif scintillation[i]:
                    color = [0,0,255]
                elif reemission[i]:
                    color = [0,255,0]
                else:
                    color = [255,255,255]
                steps = min(len(track),self.photons_max_steps) if self.photons_max_steps is not None else len(track)
                self.render_photon_track(geometry,track[:steps],sz=self.photons_track_size,color=color)
            if nphotons > 0:
                print('Rendered Photons',nphotons)
                geometry = create_geometry_from_obj(geometry)
                gpu_geometry = gpu.GPUGeometry(geometry)
                self.gpu_geometries.append(gpu_geometry)
            

    def sum_events(self):
        print('Summing events in file...')
        nchannels = self.geometry.num_channels()
        sum_hit = np.zeros(shape=nchannels, dtype=np.float)
        sum_t = np.zeros(shape=nchannels, dtype=np.float)
        sum_q = np.zeros(shape=nchannels, dtype=np.float)

        nevents = len(self.rr)

        for i, ev in enumerate(self.rr):
            sum_hit += ev.channels.hit
            sum_t[ev.channels.hit]   += ev.channels.t[ev.channels.hit]
            sum_q[ev.channels.hit]   += ev.channels.q[ev.channels.hit]
            
            if i % (nevents / 100 + 1) == 0:
                print('.', end=' ', file=sys.stderr)

        self.sum_hit = sum_hit
        self.sum_t   = sum_t / sum_hit
        self.sum_q   = sum_q / sum_hit
        print('Done.')

    def color_hit_pmts(self):
        from chroma.color import map_to_color
        self.gpu_geometry.reset_colors()

        if self.display_mode == 'geo' or self.ev is None or self.ev.channels is None:
            return

        if self.sum_mode:
            hit = self.sum_hit
            t = self.sum_t
            q = self.sum_q
            select = hit > 0
        else:
            hit = self.ev.channels.hit
            t = self.ev.channels.t
            q = self.ev.channels.q
            select = hit.copy()

        if np.count_nonzero(select) == 0:
            return

        # Important: Compute range only with HIT channels
        if self.display_mode == 'charge':
            channel_color = map_to_color(q, range=(q[select].min(),q[select].max()))
        elif self.display_mode == 'time':
            if self.sum_mode:
                crange = (t[select].min(), t[select].max())
            else:
                crange = (t[select].min(), t[select].mean())
            channel_color = map_to_color(t, range=crange)
        elif self.display_mode == 'hit':
            channel_color = map_to_color(hit, range=(hit.min(), hit.max()))
        elif self.display_mode == 'dichroicon':
            if len(select)%2 != 0:
                return
            channel_color = np.zeros_like(hit,dtype=np.uint32)
            channel_color[::2] |= (255*hit[::2]/np.max(hit[::2])).astype(np.uint32)
            channel_color[::2] |= (255*hit[1::2]/np.max(hit[1::2])).astype(np.uint32)<<16
            select[0::2] |= select[1::2]
            select[1::2] = 0
            
        solid_hit = np.zeros(len(self.geometry.mesh.triangles), dtype=np.bool)
        solid_color = np.zeros(len(self.geometry.mesh.triangles), dtype=np.uint32)

        #solid_hit[self.geometry.channel_index_to_solid_id] = select
        #all but hit PMTs transparent
        solid_hit[:] = True 
        solid_color[:] = 0xFF000000
        channel_color[np.logical_not(select)] = 0xFF000000
        solid_color[self.geometry.channel_index_to_solid_id] = channel_color

        self.gpu_geometry.color_solids(solid_hit, solid_color)

    def update(self):
        Camera.update(self)

    def process_event(self, event):
        if event.type == KEYDOWN:
            if event.key == K_p:
                self.photon_display_mode = next(self.photon_display_mode_iter)
                print(self.photon_display_mode)
                self.render_mc_info()
                self.update()
                return
                
            if event.key == K_t:
                self.track_display_mode = next(self.track_display_mode_iter)
                print(self.track_display_mode)
                self.render_mc_info()
                self.update()
                return

            if event.key == K_RIGHT and not self.sum_mode:
                try:
                    self.ev = next(self.rr)
                except StopIteration:
                    pass
                else:
                    self.color_hit_pmts()
                    self.render_mc_info()
                    self.update()
                return

            elif event.key == K_LEFT and not self.sum_mode:
                try:
                    self.ev = self.rr.prev()
                except StopIteration:
                    pass
                else:
                    self.color_hit_pmts()
                    self.render_mc_info()
                    self.update()
                return
            elif event.key == K_PERIOD:
                self.display_mode = next(self.display_mode_iter)
                self.color_hit_pmts()
                self.update()
                return
            elif event.key == K_s:
                self.sum_mode = not self.sum_mode
                if self.sum_mode and not hasattr(self, 'sum_hit'):
                    self.sum_events()
                elif not self.sum_mode and not hasattr(self, 'ev'):
                    return
                self.color_hit_pmts()
                self.render_mc_info()
                self.update()
                return

        Camera.process_event(self, event)

def view(obj, size=(800,600), mode='start', **camera_kwargs):
     geometry = create_geometry_from_obj(obj)
     camera = Camera(geometry, size, **camera_kwargs)
     if mode=='run':
         camera.run()
     elif mode=='start':
         camera.start()
         camera.join()
