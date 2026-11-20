# Plotting / rendering scripts

Run from the repo root (paths in arguments are relative to it); scripts find
`python/` modules themselves.

| script | what |
|---|---|
| `growth_movie.py TRAJ OUT_DIR [N_GIVEN_OXIDE [TITLE]]` | frames of a surface-growth trajectory (`window_generation*.py surface --traj`): accepted windows (green box, new atoms outlined), backtracks (red box), final rotation; `N_GIVEN_OXIDE` = oxide atoms of a `--continue_from` patch (drawn lighter) |
| `render_growth.py STRUCT OUT.png` | grown surface structure, two views, judged interior box, defects ringed |
| `render_defects.py` | generated window structures with defects marked |
| `mq_movie_render.py [OUT] [FPS]` | melt-quench MD movie from `data/surface/mq_movie_traj.extxyz` |
| `plot_fair_complete.py` | fair window-completion test bar chart |
| `export_vesta.py IN.extxyz OUT_PREFIX [INDEX]` | write POSCAR (`.vasp`) + CIF for VESTA (default: last frame) |
| `make_mp4.sh FRAME_DIR OUT.mp4 [FPS] [OUT.gif]` | assemble PNG frames with ffmpeg |

Example (2x2 growth movie):

    venv/bin/python python/plotting/growth_movie.py data/surface/growth_traj_2x2_new.extxyz \
        data/surface/growth_movie_2x2_new_frames 87 "2x2 continuation"
    sh python/plotting/make_mp4.sh data/surface/growth_movie_2x2_new_frames data/surface/growth_movie_2x2_new.mp4
