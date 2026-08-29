# Production FFmpeg

The video worker can use a versioned, repo-local FFmpeg build through
`VIDEO_MATTING_FFMPEG` and `VIDEO_MATTING_FFPROBE`. The checked-in systemd unit
points at `.local/ffmpeg/8.1.2`; `.local/` is intentionally ignored because
the binaries are host-local deployment artifacts.

The 8.1.2 build was configured with CUDA/NVENC/NVDEC, libnpp, VAAPI, QSV
(`libvpl`), OpenCL, DRM, libvpx, libwebp, libaom, dav1d, rav1e, x264, x265,
Opus, Vorbis, MP3, JPEG 2000, and zimg. Vulkan was left out because the host's
available Vulkan headers are older than this FFmpeg release requires; it can
be enabled after installing a current Vulkan-Headers package.

## Rebuild

Use a temporary build directory with the same FFmpeg release and the matching
NVIDIA codec headers. Verify the FFmpeg source signature before configuring it.
The important configure options are:

```sh
./configure \
  --prefix="$PWD/.local/ffmpeg/8.1.2" \
  --bindir="$PWD/.local/ffmpeg/8.1.2/bin" \
  --libdir="$PWD/.local/ffmpeg/8.1.2/lib" \
  --shlibdir="$PWD/.local/ffmpeg/8.1.2/lib" \
  --enable-gpl --enable-version3 --enable-nonfree \
  --enable-shared --disable-static \
  --enable-cuda-nvcc --enable-libnpp --enable-nvenc --enable-cuvid \
  --enable-vaapi --enable-opencl --enable-libdrm --enable-libvpl \
  --enable-libvpx --enable-libwebp --enable-libaom --enable-libdav1d \
  --enable-librav1e --enable-libx264 --enable-libx265 \
  --enable-libopus --enable-libvorbis --enable-libmp3lame \
  --enable-libopenjpeg --enable-libzimg
make -j"$(nproc)"
make install
```

The build must pass these checks before deployment:

```sh
.local/ffmpeg/8.1.2/bin/ffmpeg -hide_banner -encoders | \
  grep -E 'nvenc|qsv|vaapi|libvpx|libwebp'
.local/ffmpeg/8.1.2/bin/ffmpeg -hide_banner -hwaccels
```

Also encode a short VP9 `yuva420p` WebM and validate it with the explicit
`libvpx-vp9` decoder and `alphaextract`. A generic VP9 probe may display only
`yuv420p` because the alpha plane is carried separately in WebM.

## Deploy and rollback

The service unit is intentionally explicit about both executable paths. To
change only the media worker on a production host, install the checked-in
drop-in as root:

```sh
install -D -m 0644 systemd/omniserve-birefnet-worker-ffmpeg.conf \
  /etc/systemd/system/omniserve-birefnet-worker.service.d/ffmpeg.conf
systemctl daemon-reload
systemctl restart omniserve-birefnet-worker.service
systemctl status omniserve-birefnet-worker.service
```

Rollback is a service-only change: remove that specific drop-in, then run the
same `daemon-reload` and `restart` commands. Keep the versioned
`.local/ffmpeg/8.1.2` directory until the rollback window has passed.
