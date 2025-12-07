# -*- coding: utf-8 -*-
"""
Project: 3D Maze
Author: Jihao Ye
Start Date: 11/21/2025

Brief Description:
    - Language: Python
    - Stack: pygame, PyOpenGL
"""

# Pylint notes:
# - We intentionally use wildcard imports from pygame.locals and PyOpenGL
#   for convenience in a real-time graphics script.
# - These modules are C extensions / dynamic, so pylint cannot reliably
#   see the symbols and reports them as undefined.
# For THIS file, we disable those specific checks.
# pylint: disable=fixme, wildcard-import, unused-wildcard-import, no-member, undefined-variable

import sys
import math
import time
import random

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
        self.pitch -= dy * self.mouse_sensitivity

        # Clamp pitch to avoid flipping over
        self.pitch = max(-89.0, min(89.0, self.pitch))

    def _forward_vector(self):
        """
        Computer the forward direction vector from yaw/pitch
        Return [fx, fy, fz]
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
        Internal used for WASD movement
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
        if keys[K_s]:
            dir_x -= fx
            dir_z -= fz
        if keys[K_d]:
            dir_x += rx
            dir_z += rz
        if keys[K_a]:
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
        center_x, center_y, center_z = px, 0.0, pz

        glLoadIdentity()
        gluLookAt(eye_x, eye_y, eye_z,
                  center_x, center_y, center_z,
                  0.0, 0.0, -1.0)

    def apply_camera_overview(self, maze=None):
        """
        Set camera in high view mode (center above the maze)
        """
        if maze is not None:
            cx, cz = maze.get_center_world()
            height = max(30.0, maze.rows * maze.cell_size * 1.5)
            eye_x, eye_y, eye_z = cx, height, cz
            center_x, center_y, center_z = cx, 0.0, cz
        else:
            # Fallback: simple top-down above player
            px, py, pz = self.position
            eye_x, eye_y, eye_z = px, 30.0, pz
            center_x, center_y, center_z = px, 0.0, pz

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


    def apply(self, player, maze=None):
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
            player.apply_camera_overview(maze)
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

    def get_entry_yaw(self):
        """
        Choose a reasonable initial yaw (in degrees) for the player
        at the entrance, based on which neighboring directions are open
        """
        r, c = self.entrance
        cell = self.grid[r][c]

        # South (toward +Z, deeper rows)
        if not cell["walls"]["S"] and r + 1 < self.rows:
            return 180.0
        # East (toward +X, deeper cols)
        if not cell["walls"]["E"] and c + 1 < self.cols:
            return 90.0
        # North (toward -Z)
        if not cell["walls"]["N"] and r - 1 >= 0:
            return 0.0
        # West (toward -X)
        if not cell["walls"]["W"] and c - 1 >= 0:
            return -90.0

        return 0.0

    # ---------- Collision helpers ----------

    def world_to_cell_indices(self, x, z):
        """
        Convert world (x, z) to integer (row, col) indices
        """
        half_w = (self.cols * self.cell_size) / 2.0
        half_h = (self.rows * self.cell_size) / 2.0

        # Shift so (0,0) in grid space is top_left corner of maze
        col = int((x + half_w) // self.cell_size)
        row = int((z + half_h) // self.cell_size)

        return row, col

    def _in_bounds(self, row, col):
        """
        Check if (row, col) is inside the maze grid
        """
        return 0 <= row < self.rows and 0 <= col < self.cols

    def apply_cell_collisions(self, old_x, old_z, new_x, new_z):
        """
        Given an old position and a proposed new position, decide if the move crosses a wall
        - if the move stays in the same cell:
            - aloow movement, but keep a small radius away from any walls in that cell
        - if the move goes to a neighboring cell:
            - check the corresponding wall in the old cell
            - if there is a wall, block (return old position)
            - if no wall, allow (return new position)
        - If the move goes outside the maze or jumps over more than one cell: block
        """
        row0, col0 = self.world_to_cell_indices(old_x, old_z)
        row1, col1 = self.world_to_cell_indices(new_x, new_z)

        if not (self._in_bounds(row0, col0) and self._in_bounds(row1, col1)):
            return old_x, old_z

        # Same cell => stay inside, but keep a small radius away from walls
        if row0 == row1 and col0 == col1:
            cell = self.grid[row0][col0]

            x_center, z_center = self.cell_to_world(row0, col0)
            s = self.cell_size / 2.0

            x_left  = x_center - s
            x_right = x_center + s
            z_top   = z_center - s
            z_bottom = z_center + s

            # Player radius: how close we allow to get to a wall
            radius = self.cell_size * 0.1

            x_clamped = new_x
            z_clamped = new_z

            # Keep away from north wall
            if cell["walls"]["N"]:
                min_z = z_top + radius
                if z_clamped < min_z:
                    z_clamped = min_z
            # Keep away from south wall
            if cell["walls"]["S"]:
                max_z = z_bottom - radius
                if z_clamped > max_z:
                    z_clamped = max_z
            # Keep away from west wall
            if cell["walls"]["W"]:
                min_x = x_left + radius
                if x_clamped < min_x:
                    x_clamped = min_x
            # Keep away from east wall
            if cell["walls"]["E"]:
                max_x = x_right - radius
                if x_clamped > max_x:
                    x_clamped = max_x

            return x_clamped, z_clamped

        dr = row1 - row0
        dc = col1 - col0

        if abs(dr) + abs(dc) > 1:
            return old_x, old_z

        cell = self.grid[row0][col0]

        # Moving north
        if dr == -1 and dc == 0:
            if cell["walls"]["N"]:
                return old_x, old_z
            else:
                return new_x, new_z
        # Moving south
        if dr == 1 and dc == 0:
            if cell["walls"]["S"]:
                return old_x, old_z
            else:
                return new_x, new_z
        # Moving east
        if dr == 0 and dc == 1:
            if cell["walls"]["E"]:
                return old_x, old_z
            else:
                return new_x, new_z
        # Moving west
        if dr == 0 and dc == -1:
            if cell["walls"]["W"]:
                return old_x, old_z
            else:
                return new_x, new_z

        return new_x, new_z

    # ---------- Generation (TODO) ----------

    def generate_random(self, seed=None):
        """
        Generate a random maze using a depth-first search (DFS) backtracker.
        - start from the entrance cell
        - removes walls between cells to form a spanning tree
        """
        # Local RNG so seeding does not affect global random state
        rng = random.Random(seed)


        # Reset all walls to present and types to empty
        for r in range(self.rows):
            for c in range(self.cols):
                cell = self.grid[r][c]
                cell["walls"] = {"N": True, "E": True, "S": True, "W": True}
                cell["type"] = "empty"
                # cell["walls"]["N"] = (r == 0)
                # cell["walls"]["S"] = (r == self.rows - 1)
                # cell["walls"]["W"] = (c == 0)
                # cell["walls"]["E"] = (c == self.cols - 1)

        # Visited grid for DFS
        visited = [[False for _ in range(self.cols)] for _ in range(self.rows)]

        # Start from entrance
        start_r, start_c = self.entrance
        self._carve_passages_from(start_r, start_c, visited, rng)

        # Exit is currently fixed at bottom-right; DFS guarantees it's reachable.
        # Later, we can optionally choose exit as farthest cell from entrance.

    def _carve_passages_from(self, r, c, visited, rng):
        """
        Recursive DFS maze carver
        - r, c: current cell indices
        - visited: 2D list of booleans
        - rng: random.Random instance
        """
        visited[r][c] = True

        # Directions: (dir_key, dr, dc, opposite_dir_key)
        directions = [
            ("N", -1, 0, "S"),
            ("S", 1, 0, "N"),
            ("W", 0, -1, "E"),
            ("E", 0, 1, "W"),
        ]
        rng.shuffle(directions)

        for dir_key, dr, dc, opposite_key in directions:
            nr = r + dr
            nc = c + dc

            # Check bounds
            if nr < 0 or nr >= self.rows or nc < 0 or nc >= self.cols:
                continue

            if not visited[nr][nc]:
                self.grid[r][c]["walls"][dir_key] = False
                self.grid[nr][nc]["walls"][opposite_key] = False

                # Recurse into neighbor
                self._carve_passages_from(nr, nc, visited, rng)

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

        # Draw walls per cell (N/E/S/W)
        glColor3f(0.75, 0.75, 0.75)
        self._draw_all_walls()

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

    def _draw_all_walls(self):
        """
        Draw walls for each cell according to its 'walls' dictionary
        To avoid drawing shared walls twice:
        - for every cell, draw N and W walls if present
        - for the last row, also draw S walls
        - for the last column, also draw E walls
        """
        wall_height = 2.0

        for r in range(self.rows):
            for c in range(self.cols):
                cell = self.grid[r][c]
                x_center, z_center = self.cell_to_world(r, c)
                s = self.cell_size / 2.0

                # Precompute cell edge coordinates
                x_left = x_center - s
                x_right = x_center + s
                z_top = z_center - s
                z_bottom = z_center + s

                # North wall
                if cell["walls"]["N"]:
                    self._draw_wall_segment(x_left, z_top, x_right, z_top, wall_height)
                # West wall
                if cell["walls"]["W"]:
                    self._draw_wall_segment(x_left, z_top, x_left, z_bottom, wall_height)
                # South wall
                if r == self.rows - 1 and cell["walls"]["S"]:
                    self._draw_wall_segment(x_left, z_bottom, x_right, z_bottom, wall_height)
                # East wall
                if c == self.cols - 1 and cell["walls"]["E"]:
                    self._draw_wall_segment(x_right, z_top, x_right, z_bottom, wall_height)

    def _draw_wall_segment(self, x0, z0, x1, z1, height):
        """
        Draw a vertical wall quad along the segment from (x0, z0) to (x1, z1)
        """
        glBegin(GL_QUADS)
        # Bottom edge
        glVertex3f(x0, 0.0, z0)
        glVertex3f(x1, 0.0, z1)
        # Top edge
        glVertex3f(x1, height, z1)
        glVertex3f(x0, height, z0)
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

        self.font = pygame.font.SysFont("consolas", 20)

        flags = DOUBLEBUF | OPENGL  # pylint: disable=unsupported-binary-operation
        pygame.display.set_mode((width, height), flags)
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
        self.player.yaw = self.maze.get_entry_yaw()
        self.player.pitch = 0.0

        # Camera
        self.camera = CameraController()

        # Mouse grab
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)

        # Timing
        self.clock = pygame.time.Clock()
        self.start_time = time.time()
        self.elapsed_time = 0.0

    def init_opengl(self):
        """
        Configure basic OpenGL state.
        """
        glViewport(0, 0, self.width, self.height)
        glEnable(GL_DEPTH_TEST)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(FOV_Y, self.width / float(self.height), NEAR_PLANE, FAR_PLANE)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # glEnable(GL_CULL_FACE)
        # glCullFace(GL_BACK)

    def handle_events(self):
        """
        Handle pygame events and return mouse deltas.
        """
        mouse_dx, mouse_dy = 0, 0

        for event in pygame.event.get():
            if event.type == QUIT:
                self.running = False

            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    self.running = False

                # Camera mode switching
                elif event.key == K_1:
                    self.camera.mode = "fps"
                    print("Camera mode: fps")
                elif event.key == K_2:
                    self.camera.mode = "third_p"
                    print("Camera mode: third-person")
                elif event.key == K_3:
                    self.camera.mode = "top_down"
                    print("Camera mode: top-down")
                elif event.key == K_4:
                    self.camera.mode = "overview"
                    print("Camera mode: overview")

                # Restart from entrance
                elif event.key == K_r:
                    self.restart_from_entrance()
                    print("Restarted from entrance")

                # Regenerate maze
                elif event.key == K_n:
                    self.regenerate_maze()
                    print("Regenerated maze")

            elif event.type == MOUSEMOTION:
                dx, dy = event.rel
                mouse_dx += dx
                mouse_dy += dy

        return mouse_dx, mouse_dy

    def restart_from_entrance(self):
        """
        Reset player to entrance and reset timer.
        """
        start_pos = self.maze.get_entrance_world_position()
        self.player.set_position(*start_pos)
        self.player.yaw = self.maze.get_entry_yaw()
        self.player.pitch = 0.0
        self.start_time = time.time()
        self.elapsed_time = 0.0

    def regenerate_maze(self):
        """
        Create a new maze, move player to entrance, reset timer.
        """
        self.maze.generate_random(seed=None)
        start_pos = self.maze.get_entrance_world_position()
        self.player.set_position(*start_pos)
        self.player.yaw = self.maze.get_entry_yaw()
        self.player.pitch = 0.0
        self.start_time = time.time()
        self.elapsed_time = 0.0

    def update(self, dt, mouse_dx, mouse_dy):
        """
        Update game state: orientation, movement, timer.
        """
        # Mouse -> orientation
        self.player.handle_mouse(mouse_dx, mouse_dy)

        # Keyboard -> movement
        keys = pygame.key.get_pressed()
        old_x = self.player.position[0]
        old_z = self.player.position[2]
        self.player.handle_keyboard(dt, keys)
        desired_x = self.player.position[0]
        desired_z = self.player.position[2]

        mid_x, mid_z = self.maze.apply_cell_collisions(old_x, old_z, desired_x, old_z)
        final_x, final_z = self.maze.apply_cell_collisions(mid_x, mid_z, mid_x, desired_z)

        self.player.position[0] = final_x
        self.player.position[2] = final_z

        # Timer
        self.elapsed_time = time.time() - self.start_time

        # TODO:
        # - collision with walls
        # - traps / power-ups
        # - detect reaching exit

    def draw_scene(self):
        """
        Render the 3D world and HUD.
        """
        glClearColor(0.08, 0.08, 0.12, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Set camera
        self.camera.apply(self.player, self.maze)

        # Draw maze (floor + border walls)
        self.maze.draw()

        # Simple reference cube at origin
        glColor3f(0.8, 0.3, 0.3)
        self._draw_unit_cube_at(0.0, 0.5, 0.0)

        # 2D HUD
        self.draw_hud()

        # TODO:
        # - draw player model for third-person/top-down if needed
        # - draw 2D HUD (timer, position) with pygame text

    def _draw_unit_cube_at(self, x, y, z):
        """
        Draw a 1x1x1 cube centered at (x, y, z).
        """
        s = 0.5
        vertices = [
            [-s, -s, -s],
            [ s, -s, -s],
            [ s,  s, -s],
            [-s,  s, -s],
            [-s, -s,  s],
            [ s, -s,  s],
            [ s,  s,  s],
            [-s,  s,  s],
        ]
        faces = [
            [0, 1, 2, 3],
            [3, 2, 6, 7],
            [7, 6, 5, 4],
            [4, 5, 1, 0],
            [1, 5, 6, 2],
            [4, 0, 3, 7],
        ]
        glPushMatrix()
        glTranslatef(x, y, z)
        glBegin(GL_QUADS)
        for face in faces:
            for idx in face:
                glVertex3fv(vertices[idx])
        glEnd()
        glPopMatrix()

    def _start_2d(self):
        """
        Switch to 2D orthographic projection for HUD drawing
        """
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, self.height, 0, -1, 1)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_DEPTH_TEST)

    def _end_2d(self):
        """
        Restore 3D projection/modelview after HUD drawing.
        """
        glEnable(GL_DEPTH_TEST)

        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()

        # Leave matrix mode in MODELVIEW for normal 3D rendering
        glMatrixMode(GL_MODELVIEW)


    def _draw_text_2d(self, x, y, text, color=(255, 255, 255, 255)):
        """
        Draw text at screen coordinates (x, y) using a temporary texture.
        (0,0) is top-left of the window.
        """
        if not text:
            return

        # Render text to a pygame surface
        surface = self.font.render(text, True, color[:3])
        text_data = pygame.image.tostring(surface, "RGBA", False)
        w, h = surface.get_size()

        # Create a temporary texture
        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, text_data)

        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Draw textured quad
        glColor4f(1.0, 1.0, 1.0, 1.0)

        # Draw textured quad
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 0.0)
        glVertex2f(x, y)
        glTexCoord2f(1.0, 0.0)
        glVertex2f(x + w, y)
        glTexCoord2f(1.0, 1.0)
        glVertex2f(x + w, y + h)
        glTexCoord2f(0.0, 1.0)
        glVertex2f(x, y + h)
        glEnd()

        glDisable(GL_BLEND)
        glDisable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, 0)
        glDeleteTextures([tex_id])

    def draw_hud(self):
        """
        Draw HUD with elapsed time and player position (cell indices).
        """
        # Format time as MM:SS
        total_seconds = int(self.elapsed_time)
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        time_text = f"Time: {minutes:02d}:{seconds:02d}"

        # Player cell position
        px = self.player.position[0]
        pz = self.player.position[2]
        row, col = self.maze.world_to_cell_indices(px, pz)
        pos_text = f"Cell: ({row}, {col})"

        # Optional: world coords (rounded)
        world_text = f"Pos: ({px:.1f}, {pz:.1f})"

        self._start_2d()
        # Top-left corner
        self._draw_text_2d(10, 10, time_text)
        self._draw_text_2d(10, 35, pos_text)
        self._draw_text_2d(10, 60, world_text)
        self._end_2d()

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
