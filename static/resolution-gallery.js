/* Resize-only previews: selected resolution is never presented as a model render. */
class ResolutionGallery {
  constructor(dialog, image) {
    this.dialog = dialog;
    this.image = image;
    this.controls = document.getElementById('resolution-controls');
    this.slider = document.getElementById('resolution-slider');
    this.output = document.getElementById('resolution-value');
    this.status = document.getElementById('resolution-status');
    this.stage = document.getElementById('gallery-stage');
    this.zoom = document.getElementById('resolution-zoom');
    this.download = document.getElementById('resolution-download');
    this.session = 0;
    this.selection = 0;
    this.cache = new Map();
    this.pending = new Map();
    this.queue = [];
    this.activeLoads = 0;
    this.fullURLs = [];
    this.slider.addEventListener('input', () => {
      if (this.raf) return;
      this.raf = requestAnimationFrame(() => {
        this.raf = 0;
        this.select(Number(this.slider.value));
      });
    });
    this.slider.addEventListener('pointerdown', () => {
      this.dragging = true;
      clearTimeout(this.timer);
      this.fullController?.abort();
    });
    const finish = () => {
      if (!this.dragging) return;
      this.dragging = false;
      if (this.sample) this.select(Number(this.slider.value));
    };
    window.addEventListener('pointerup', finish);
    window.addEventListener('pointercancel', finish);
    this.slider.addEventListener('change', () => {
      if (!this.dragging && this.sample) this.select(Number(this.slider.value));
    });
    this.zoom.addEventListener('click', () => {
      const enabled = this.zoom.getAttribute('aria-pressed') !== 'true';
      this.zoom.setAttribute('aria-pressed', String(enabled));
      this.zoom.textContent = enabled ? 'Fit image' : 'View at 100%';
      this.stage.classList.toggle('actual-size', enabled);
    });
  }

  async manifest() {
    if (window.RAPD_RESOLUTION_MANIFEST) return window.RAPD_RESOLUTION_MANIFEST;
    if (!this.manifestPromise) {
      this.manifestPromise = fetch('gallery/resolutions/manifest.json?v=1')
        .then(response => {
          if (!response.ok) throw new Error('Manifest unavailable');
          return response.json();
        }).catch(error => { this.manifestPromise = null; throw error; });
    }
    return this.manifestPromise;
  }

  async open(source) {
    this.close();
    const session = this.session;
    this.controller = new AbortController();
    this.controls.hidden = false;
    this.slider.disabled = true;
    this.output.textContent = '4096';
    this.status.textContent = 'Loading resolution options…';
    this.download.href = source;
    try {
      const manifest = await this.manifest();
      if (session !== this.session || !this.dialog.open) return;
      this.sample = manifest.samples[source];
      if (!this.sample) throw new Error('Sample unavailable');
      this.slider.min = manifest.min;
      this.slider.max = manifest.max;
      this.slider.step = manifest.step;
      this.slider.value = manifest.max;
      this.slider.disabled = false;
      this.select(manifest.max);
    } catch {
      if (session !== this.session || !this.dialog.open) return;
      this.status.textContent = 'Resolution previews unavailable. Showing the original image.';
      this.image.src = source;
    }
  }

  select(size) {
    if (!this.sample) return;
    const frame = this.sample.frames.find(frame => frame.size === size);
    if (!frame) return;
    const selection = ++this.selection;
    const session = this.session;
    this.fullReady = false;
    clearTimeout(this.timer);
    this.fullController?.abort();
    this.output.textContent = `${frame.width} × ${frame.height}`;
    this.slider.setAttribute('aria-valuetext', `${frame.width} by ${frame.height} pixels`);
    this.download.href = frame.full;
    this.download.download = `${this.sample.original.split('/').pop().replace(/\.[^.]+$/, '')}-${size}.webp`;
    this.status.textContent = 'Loading preview…';
    // Discard queued speculation so the latest requested frame has priority.
    this.queue.splice(0).forEach(job => { this.pending.delete(job.url); job.resolve(null); });
    this.preview(frame.preview, true).then(asset => {
      if (!asset || session !== this.session || selection !== this.selection || !this.dialog.open || this.fullReady) return;
      this.image.src = asset.url;
      this.status.textContent = 'Preview · full resolution loads when you stop';
    }).catch(() => {
      if (session === this.session && selection === this.selection && !this.fullReady) {
        this.status.textContent = 'Preview unavailable · loading full resolution';
      }
    });
    const index = this.sample.frames.indexOf(frame);
    for (const offset of [-1, 1, -2, 2]) {
      const neighbor = this.sample.frames[index + offset];
      if (neighbor) this.preview(neighbor.preview).catch(() => {});
    }
    if (!this.dragging) {
      this.timer = setTimeout(() => this.loadFull(frame, session, selection), 220);
    }
  }

  preview(url, urgent = false) {
    if (this.cache.has(url)) {
      const asset = this.cache.get(url);
      this.cache.delete(url);
      this.cache.set(url, asset);
      return Promise.resolve(asset);
    }
    if (this.pending.has(url)) return this.pending.get(url);
    const promise = new Promise((resolve, reject) => {
      const job = {url, resolve, reject, session: this.session, signal: this.controller.signal};
      urgent ? this.queue.unshift(job) : this.queue.push(job);
    });
    this.pending.set(url, promise);
    this.pump();
    return promise;
  }

  pump() {
    while (this.activeLoads < 3 && this.queue.length) {
      const job = this.queue.shift();
      this.activeLoads++;
      this.decode(job.url, job.signal).then(asset => {
        if (job.session !== this.session) {
          URL.revokeObjectURL(asset.url);
          job.resolve(null);
          return;
        }
        this.cache.set(job.url, asset);
        while (this.cache.size > 12) {
          const key = this.cache.keys().next().value;
          URL.revokeObjectURL(this.cache.get(key).url);
          this.cache.delete(key);
        }
        job.resolve(asset);
      }).catch(job.reject).finally(() => {
        if (job.session === this.session) this.pending.delete(job.url);
        this.activeLoads--;
        this.pump();
      });
    }
  }

  async decode(path, signal) {
    // Local file previews cannot use fetch; Image supports file URLs directly.
    if (window.location?.protocol === 'file:') {
      const image = new Image();
      const abort = () => { image.src = ''; };
      if (signal.aborted) throw new DOMException('Aborted', 'AbortError');
      signal.addEventListener('abort', abort, {once: true});
      try {
        image.src = path;
        await image.decode();
        if (signal.aborted) throw new DOMException('Aborted', 'AbortError');
        return {url: image.src, image};
      } catch (error) {
        if (signal.aborted) throw new DOMException('Aborted', 'AbortError');
        throw error;
      } finally {
        signal.removeEventListener('abort', abort);
      }
    }
    const response = await fetch(path, {signal});
    if (!response.ok) throw new Error(`Image request failed: ${response.status}`);
    const url = URL.createObjectURL(await response.blob());
    const image = new Image();
    image.src = url;
    try {
      await image.decode();
      if (signal.aborted) throw new DOMException('Aborted', 'AbortError');
      return {url, image};
    } catch (error) {
      URL.revokeObjectURL(url);
      throw error;
    }
  }

  async loadFull(frame, session, selection) {
    if (this.dragging || session !== this.session || selection !== this.selection) return;
    this.fullController = new AbortController();
    try {
      const asset = await this.decode(frame.full, this.fullController.signal);
      if (session !== this.session || selection !== this.selection || !this.dialog.open) {
        URL.revokeObjectURL(asset.url);
        return;
      }
      this.fullReady = true;
      this.image.src = asset.url;
      this.fullURLs.forEach(url => URL.revokeObjectURL(url));
      this.fullURLs = [asset.url];
      this.status.textContent = `Full resolution · ${frame.width} × ${frame.height}`;
    } catch (error) {
      if (error.name !== 'AbortError' && session === this.session && selection === this.selection) {
        this.status.textContent = 'Full resolution could not load. Move the slider to retry.';
      }
    }
  }

  close() {
    this.session++;
    this.selection++;
    this.controller?.abort();
    this.fullController?.abort();
    clearTimeout(this.timer);
    cancelAnimationFrame(this.raf);
    this.raf = 0;
    this.dragging = false;
    this.sample = null;
    this.queue.splice(0).forEach(job => job.resolve(null));
    this.pending.clear();
    this.cache.forEach(asset => URL.revokeObjectURL(asset.url));
    this.cache.clear();
    this.fullURLs.forEach(url => URL.revokeObjectURL(url));
    this.fullURLs = [];
    this.controls.hidden = true;
    this.stage.classList.remove('actual-size');
    this.zoom.setAttribute('aria-pressed', 'false');
    this.zoom.textContent = 'View at 100%';
  }
}
window.ResolutionGallery = ResolutionGallery;
