# 3D Maze (pygame + PyOpenGL)

Assignment-oriented 3D maze game: random >=10x10 maze, FPS movement with extra camera modes, traps/power-ups, and HUD overlays. Single-file Python build (`maze.py`) using pygame for window/input/HUD and PyOpenGL for 3D.

## Quick start
- Requirements: Python 3.10+, `pygame`, `PyOpenGL`, `PyOpenGL-accelerate` (optional speedup).
- Install:
  - `python -m venv .venv`
  - `.venv\Scripts\activate`
  - `pip install pygame PyOpenGL PyOpenGL-accelerate`
- Run: `python maze.py`

## Controls (planned wiring)
- Move: `W/A/S/D`
- Look: mouse
- Restart run: `R`
- Regenerate maze: `N`
- Cameras: `1` fps (default), `2` third-person, `3` top-down, `4` overview/launch
- HUD: on-screen timer + position (pygame overlay)
(Current code defines player/camera classes; event wiring will hook these soon.)

## Architecture (single file)
- `Player`: position, yaw/pitch, FPS-style movement, camera helpers for fps/third/top-down/overview.
- `CameraController`: switches camera modes cleanly via `mode` + `apply`.
- `Maze`: grid with per-cell walls/types, coordinate helpers, entrance/exit, draw helpers; maze generation placeholder.
- `Game`: window + GL init, owns player/maze/camera, main loop scaffolding.

## Stage/progress tracker
- [ ] 1. Project skeleton (window, loop, input handling, camera switching) - window + classes exist; event/update/draw still TODO.
- [ ] 2. Maze generation (random >=10x10 via DFS) - placeholder clears inner walls.
- [ ] 3. Maze rendering (walls/floor geometry, consistent world coords) - floor + outer wall stub only.
- [ ] 4. Movement + collisions - movement maths exist; collision tests not wired.
- [ ] 5. Timer + HUD - fields defined; no overlay yet.
- [ ] 6. Traps / specials / power-ups - not started.
- [ ] 7. Visual polish (textures, basic lighting, special tile cues) - not started.
- [ ] 8. Replay controls (restart/regenerate) - keys reserved; logic pending.
- [ ] 9. Final QA vs spec - pending.

## Next focus
1) Implement `Game.handle_events/update/draw_scene` to enable movement + camera switching.
2) Replace maze placeholder with DFS backtracker and basic wall rendering.
3) Add collision checks + HUD overlay before layering traps/power-ups.
