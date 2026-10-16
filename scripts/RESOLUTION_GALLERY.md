# Resolution gallery

Each of the 15 gallery originals has 57 resized versions (512–4096 long-edge pixels, step 64). These are image-resizing demonstrations, not new model outputs. Source PNG files are preserved.

- Full files: WebP quality 85.
- Previews: at most 768 pixels on the long edge, WebP quality 74.
- `gallery/resolutions/manifest.json` maps the existing gallery source paths to their size options.
- Open a gallery image to use the slider. Dragging loads previews only; stopping for 220 ms loads the selected full file. Keyboard changes use the same delay.
- Neighbor prefetching is limited to two steps in each direction, three simultaneous preview requests, and 12 decoded cached previews. Closing the modal aborts image requests and releases object URLs.
- “View at 100%” shows the selected file at its natural pixel size inside a scrollable viewport; the default fits the image in a fixed display area.

Generate/reuse assets (requires Pillow):

```sh
python3 scripts/build_resolution_gallery.py --workers 3
```

Existing generated WebP files are reused and checked. Remove a sample's generated folder before rebuilding with different encoder settings or replacing its original. The manifest is replaced only after all samples succeed.

Run logic checks:

```sh
node --test scripts/test_resolution_gallery.cjs
```

Serve this directory using an HTTP server, e.g.:

```sh
python3 -m http.server 8765 --bind 127.0.0.1
```

Open http://127.0.0.1:8765/ . Directly opening `index.html` also works: `manifest.js` supplies the size options without fetch, and local previews use image URLs directly. Deploy `index.html`, both JavaScript files, the stylesheet, and `gallery/resolutions/` together. The legacy `build.py` predates the gallery and should not be used to regenerate this edited page.
