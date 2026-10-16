# Resolution gallery

The gallery contains 15 samples, each with 57 resolution options from 512 to 4096 pixels in steps of 64.

`gallery/resolutions/manifest.json` and `manifest.js` map each sample to its full-size and preview assets. Keep both manifests synchronized.

Open a gallery image to use the slider. While dragging, the viewer loads lightweight previews. After a 220 ms pause, it loads the selected full-resolution file. Neighbor prefetching is limited to two steps in each direction, three simultaneous requests, and 12 cached previews. Closing the viewer cancels image requests and releases temporary URLs.

“View at 100%” displays the selected file at its natural pixel size in a scrollable viewport. The default view fits the image into a fixed area.

Run logic checks:

```sh
node --test scripts/test_resolution_gallery.cjs
```

The optional browser check requires Playwright and Chromium:

```sh
node scripts/test_resolution_gallery_browser.cjs
```

For HTTP preview:

```sh
python3 -m http.server 8765 --bind 127.0.0.1
```

Open http://127.0.0.1:8765/ . Directly opening `index.html` also works. Deploy the page, scripts, stylesheets, and `gallery/resolutions/` together. The legacy `build.py` predates the gallery and should not regenerate this edited page.
