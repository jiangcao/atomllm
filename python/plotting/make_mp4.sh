#!/bin/sh
# Assemble PNG frames (f0000.png, f0001.png, ...) into an MP4 (and optionally a GIF).
# usage: sh python/plotting/make_mp4.sh FRAME_DIR OUT.mp4 [FPS=6] [OUT.gif]
d=$1; out=$2; fps=${3:-6}
ffmpeg -loglevel error -y -framerate "$fps" -i "$d/f%04d.png" -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
  -c:v libx264 -pix_fmt yuv420p "$out"
if [ -n "$4" ]; then
  ffmpeg -loglevel error -y -framerate "$fps" -i "$d/f%04d.png" \
    -vf "scale=600:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" "$4"
fi
ls -lh "$out" $4
