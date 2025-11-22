# -*- coding: utf-8 -*-
"""
Project: 3D Maze
Author: Jihao Ye
Start Date: 11/21/2025

Brief Description:
    - Language: Python
    - Stack: pygame, PyOpenGL
"""

import sys
import math
import time

import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *


# -----------------------------
# Configuration
# -----------------------------
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 700

# Maze size
MAZE_ROWS = 10
MAZE_COLS = 10
MAZE_CELL_SIZE = 2.0

FOV_Y = 75.0
NEAR_PLANE = 0.1
FAR_PLANE = 200.0

TARGET_FPS = 60


# -----------------------------
# Player (movement + orientation)
# -----------------------------
class Player:
    """
    Player holds position and viewing direction
    - position: [x, y]
    - yaw: rotation around Y axis
    - pitch: rotation around x axis
    """
    
    def __init__(self, start_pos):
        self.position = [start_pos[0], start_pos[1], start_pos[2]]
        self.yaw = 0.0
        self.pitch = 0.0
        self.move_speed = 6.0
        self.mouse_sensitivity = 0.1
    
    # ---------- Orientation helpers ----------
    
    def handle_mouse(self, dx, dy):
        """
        Update yaw/pitch from mouse movement
        """
        self.yaw += dx * self.mouse_sensitivity
        self.pitch += dy * self.mouse_sensitivity
        
        # Clamp pitch to avoid flipping over
        self.pitch = max(-89.0, min(89,0, self.pitch))
        
    def _forward_vector(self):
        """
        Computer the forward direction vector from yaw/pitch
        - return [fx, fy, fz]
        """
        yaw_rad = math.radians(self.yaw)
        pitch_rad = math.radians(self.pitch)
        
        fx = math.cos(pitch_rad) * math.sin(yaw_rad)
        fy = math.sin(pitch_rad)
        fz = -math.cos(pitch_rad) * math.cos(yaw_rad)
        
        return [fx, fy, fz]
    
    def _right_vector(self):
        """
        Computer the right direction vector
        - internal used for WASD movement

        """
        fx, fy, fz = self._forward_vector()
        up = [0.0, 1.0, 0.0]
        
        # right = forward * up
        rx = fy * up[2] - fz * up[1]
        ry = fz * up[0] - fx * up[2]
        rz = fx * up[1] - fy * up[0]
        
        length = math.sqrt(rx * rx + ry * ry + rz * rz)
        if length > 0:
            rx /= length
            ry /= length
            rz /= length
        
        return [rx, ry, rz]
    
    def handle_keyboard(self, dt, keys):
        """
        Move in the XZ plane based on WASD keys
        """
        dir_x = 0.0
        dir_z = 0.0
        
        # Froward vector projected onto XZ plane
        fx, _, fz = self._forward_vector()
        length_f = math.sqrt(fx * fx + fz * fz)
        if length_f > 0:
            fx /= length_f
            fz /= length_f
        
        # Right vector projected onto XZ plane
        rx, _, rz = self._right_vector()
        length_r = math.sqrt(rx * rx + rz * rz)
        if length_r > 0:
            rx /= length_r
            rz /= length_r
            
        # WASD movement input:
        if keys[K_w]:
            dir_x += fx
            dir_z += fz
        if keys[K_a]:
            dir_x -= fx
            dir_z -= fz
        if keys[K_s]:
            dir_x += rx
            dir_z += rz
        if keys[K_d]:
            dir_x -= rx
            dir_z -= rz
            
        # Normalize direction
        length_dir = math.sqrt(dir_x * dir_x + dir_z * dir_z)
        if length_dir > 0:
            dir_x /= length_dir
            dir_z /= length_dir
        
        # Apply movement
        speed = self.move_speed * dt
        self.position[0] += dir_x * speed
        self.position[2] += dir_z * speed
        
    def set_position(self, x, y, z):
        """
        Teleport the player (used for restart/initalize)
        """
        self.position[0] = x
        self.position[1] = y
        self.position[2] = z
        
    # ---------- Camera application ----------
        
    def apply_camera_fps(self):
        """
        Set camera in fps mode (eye at player's head)
        """
        fx, fy, fz = self._forward_vector()
        px, py, pz = self.position
        
        cx = px + fx
        cy = py + fy
        cz = pz + fz
        
        glLoadIdentity()
        gluLookAt(px, py, pz,       # eye
                  cx, cy, cz,       # center
                  0.0, 1.0, 0.0)    # up
    
    def apply_camera_third_p(self):
        """
        Set camera in 3rd-person mode (behind and above the player)
        """
        yaw_rad = math.radians(self.yaw)
        dist = 6.0
        height = 6.0
        px, py, pz = self.position
        
        eye_x = px - math.sin(yaw_rad) * dist
        eye_y = py + height
        eye_z = pz + math.cos(yaw_rad) * dist
        
        center_x, center_y, center_z = px, py, pz
        
        glLoadIdentity()
        gluLookAt(eye_x, eye_y, eye_z,
                  center_x, center_y, center_z,
                  0.0, 1.0, 0.0)
    
    def apply_camera_top_down(self):
        """
        Set camera in top-down mode (look straight down at player)
        """
        height = 20.0
        px, py, pz = self.position
        eye_x, eye_y, eye_z = px, height, pz
        center_x, center_y, center_z = pz, 0.0, pz
        
        glLoadIdentity()
        gluLookAt(eye_x, eye_y, eye_z,
                  center_x, center_y, center_z,
                  0.0, 0.0, -1.0)
    
    def apply_camera_overview(self, maze=None):
        """
        Set camera in high view mode (center above the maze)
        """
        cx, cz = maze.get_center_world()
        height = max(30.0, maze.rows* maze.cell_size * 1.5)
        
        eye_x, eye_y, eye_z = cx, height, cz
        center_x, center_y, center_z = cx, 0.0, cz
        
        glLoadIdentity()
        gluLookAt(eye_x, eye_y, eye_z,
                  center_x, center_y, center_z,
                  0.0, 0.0, -1.0)
        


# -----------------------------
# Camera controller (view modes)
# -----------------------------
class CameraController:
    """
    CameraController manages different view modes:
        - fps: first-person view from the player's head
        - third_p: 3rd-person view
        - top_down: strict top-down view
        - overview: high view over the max
    """
    
    def __init__(self):
        self.mode = "fps"
        
    
    def apply(self, player):
        """
        Sets the view matrix based on current mode
        """
        if self.mode == "fps":
            player.apply_camera_fps()
            return
        elif self.mode == "third_p":
            player.apply_camera_third_p()
            return
        elif self.mode == "top_down":
            player.apply_camera_top_down()
            return
        elif self.mode == "overview":
            player.apply_camera_overview()
            return
        else:
            player.apply_camera_fps()


# -----------------------------
# Maze data structure
# -----------------------------
class Maze:
    """
    Maze holds the logical grid and drawing code
    - rows x cols grid of cells
    - each cell can later store:
        - which walls exist(N/E/S/W)
        - trap / power-up / special type
    """
    # TODO
    def __init__(self, rows, cols, cell_size=2.0):
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size
        
        # Grid of cells; each cell is a dict for flexibility
        self.grid = [
            [
                {
                    "walls": {"N": True, "E": True, "S": True, "W": True},
                    "type": "empty",
                }
                for _ in range(cols)
            ]
            for _ in range(rows)
        ]
        
        # Entrance and exit cells
        self.entrance = (0, 0)
        self.exit = (rows - 1, cols - 1)
        
    # ---------- Coordinate helpers ----------
    
    def cell_to_world(self, row, col):
        """
        Convert cell (row, col) to world (x, z) center coordinates
        """
        half_w = (self.cols * self.cell_size) / 2.0
        half_h = (self.rows * self.cell_size) / 2.0
        
        x = (col + 0.5) * self.cell_size - half_w
        z = (row + 0.5) * self.cell_size - half_h
        
        return x, z
    
    def get_entrance_world_position(self):
        """
        World-space starting position for thr player at the entrance
        """
        r, c = self.entrance
        x, z = self.cell_to_world(r, c)
        
        return [x, 1.0, z]
    
    def get_center_world(self):
        """
        World-space center of the maze
        """
        center_row = self.rows / 2.0
        center_col = self.cols / 2.0
        x, z = self.cell_to_world(center_row - 0.5, center_col - 0.5)
        
        return x, z
    
    # ---------- Generation (TODO) ----------

    def generate_random(self, seed=None):
        """
        Implement a random maze generator (use DFS backtracker ...)
        """
        # TODO
        if seed is not None:
            import random
            random.seed(seed)
        
        # Simple placeholder: remove all internal walls
        for r in range(self.rows):
            for c in range(self.cols):
                cell = self.grid[r][c]
                cell["walls"]["N"] = (r == 0)
                cell["walls"]["S"] = (r == self.rows - 1)
                cell["walls"]["W"] = (c == 0)
                cell["walls"]["E"] = (c == self.cols - 1)
        
        # TODO: Replace with real maze algorithm so there is a unique-ish path
    
    # ---------- Drawing ----------
    
    def draw(self):
        """
        Draw floor for each cell and walls around the maze
        - draw individual walls according to self.grid[r][c]["walls"]
        - apply textures and lighting
        - add visual markers for traps/power-ups
        """
        # Draw floor as quads per cell
        glColor3f(0.22, 0.24, 0.26)
        for r in range(self.rows):
            for c in range(self.cols):
                self._draw_floor_cell(r, c)
        
        # Draw a simple outer border wall as a placeholder
        glColor3f(0.7, 0.7, 0.7)
        self._draw_outer_walls()
        
    def _draw_floor_cell(self, row, col):
        """
        Draw a single floor quad for cell (row, col)
        """
        x_center, z_center = self.cell_to_world(row, col)
        s = self.cell_size / 2.0
        
        x0 = x_center - s
        x1 = x_center + s
        z0 = z_center - s
        z1 = z_center + s
        
        glBegin(GL_QUADS)
        glVertex3f(x0, 0.0, z0)
        glVertex3f(x1, 0.0, z0)
        glVertex3f(x1, 0.0, z1)
        glVertex3f(x0, 0.0, z1)
        glEnd()
        
    def _draw_outer_wall(self):
        """
        Draw a simple rectangular border around the whole maze
        """
        # TODO
        half_w = (self.cols * self.cell_size) / 2.0
        half_h = (self.rows * self.cell_size) / 2.0
        
        y0 = 0.0
        y1 = 2.0
        
        # Draw 4 walls as long quads
        glBegin(GL_QUADS)
        
        # North wall
        glVertex3f(-half_w, y0, -half_h)
        glVertex3f( half_w, y0, -half_h)
        glVertex3f( half_w, y1, -half_h)
        glVertex3f(-half_w, y1, -half_h)

        # South wall
        glVertex3f(-half_w, y0, half_h)
        glVertex3f( half_w, y0, half_h)
        glVertex3f( half_w, y1, half_h)
        glVertex3f(-half_w, y1, half_h)

        # West wall
        glVertex3f(-half_w, y0, -half_h)
        glVertex3f(-half_w, y0,  half_h)
        glVertex3f(-half_w, y1,  half_h)
        glVertex3f(-half_w, y1, -half_h)

        # East wall
        glVertex3f(half_w, y0, -half_h)
        glVertex3f(half_w, y0,  half_h)
        glVertex3f(half_w, y1,  half_h)
        glVertex3f(half_w, y1, -half_h)

        glEnd()


# -----------------------------
# Game (main loop + glue)
# -----------------------------
class Game:
    """
    Game ties together:
        - Window + OpenGL setup
        - Player, CameraController, Maze
        - Event handling, update, render loop
    """
    
    def __init__(self, width=WINDOW_WIDTH, height=WINDOW_HEIGHT):
        pygame.init()
        pygame.display.set_caption("3D Maze Game - Jihao Ye")
        
        pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
        self.width = width
        self.height = height
        self.running = True
        
        # OpenGL setup
        self.init_opengl()
        
        # Maze
        self.maze = Maze(MAZE_ROWS, MAZE_COLS, MAZE_CELL_SIZE)
        self.maze.generate_random(seed=None) # TODO: replace placeholder
        
        # Player
        start_pos = self.maze.get_entrance_world_position()
        self.player = Player(start_pos)
        
        # Camera
        self.camera = CameraController()
        
        # Mouse grab
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)
        
        # Timing
        self.clock = pygame.time.Clock()
        self.start_time = time.time()
        self.elapsed_time = 0.0

    def run(self):
        """
        Main game loops
        """
        while self.running:
            dt_ms = self.clock.tick(TARGET_FPS)
            dt = dt_ms / 1000.0

            mouse_dx, mouse_dy = self.handle_events()
            self.update(dt, mouse_dx, mouse_dy)
            self.draw_scene()

            pygame.display.flip()

        pygame.quit()
        sys.exit()

# -----------------------------
# Entry point
# -----------------------------

if __name__ == "__main__":
    Game().run()