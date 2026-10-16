document.getElementById('copy').addEventListener('click', async () => {
  const code = document.getElementById('bibtex');
  const status = document.getElementById('copy-status');
  try {
    await navigator.clipboard.writeText(code.textContent);
    status.textContent = 'BibTeX copied.';
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(code);
    selection.removeAllRanges();
    selection.addRange(range);
    status.textContent = 'Citation selected. Press Ctrl+C or Command+C to copy.';
  }
});

const galleryTrack = document.querySelector('.gallery-track');
const galleryItems = [...document.querySelectorAll('.gallery-item')];
const galleryViewer = document.getElementById('gallery-viewer');
const galleryLarge = document.getElementById('gallery-large');
const galleryPrompt = document.getElementById('gallery-prompt');
const galleryPause = document.getElementById('gallery-pause');
const gallerySection = document.getElementById('gallery');
let galleryPaused = false;
let galleryHovered = false;
let galleryDragging = false;
let galleryOpener;
let galleryImageRequest = 0;
let galleryPosition = galleryTrack.scrollLeft;
let galleryLastWritten = galleryTrack.scrollLeft;
let galleryLastFrame;
const gallerySpeed = 30; // CSS pixels per second, independent of refresh rate.

// Repeat the DOM references for a seamless loop; no image data is inspected.
const galleryClones = galleryItems.map(button => {
  const clone = button.cloneNode(true);
  clone.tabIndex = -1;
  clone.setAttribute('aria-hidden', 'true');
  galleryTrack.appendChild(clone);
  return clone;
});

// Fetch small previews one viewport ahead of the scrolling track.
const galleryPreviewObserver = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (!entry.isIntersecting) return;
    entry.target.loading = 'eager';
    galleryPreviewObserver.unobserve(entry.target);
  });
}, {root: galleryTrack, rootMargin: '0px 100%'});
galleryTrack.querySelectorAll('img').forEach(img => galleryPreviewObserver.observe(img));

function animateGallery(timestamp) {
  const elapsed = galleryLastFrame === undefined ? 0 : Math.min(timestamp - galleryLastFrame, 100);
  galleryLastFrame = timestamp;
  if (galleryTrack.scrollLeft !== galleryLastWritten) galleryPosition = galleryTrack.scrollLeft;
  if (!galleryPaused && !galleryHovered && !galleryDragging && !galleryViewer.open && !document.hidden) {
    const loopWidth = galleryClones[0].offsetLeft - galleryItems[0].offsetLeft;
    if (loopWidth > 0) {
      galleryPosition = (galleryPosition + gallerySpeed * elapsed / 1000) % loopWidth;
      galleryTrack.scrollLeft = galleryPosition;
    }
  }
  galleryLastWritten = galleryTrack.scrollLeft;
  window.requestAnimationFrame(animateGallery);
}

[...galleryItems, ...galleryClones].forEach(button => button.addEventListener('click', () => {
  galleryOpener = button;
  const thumbnail = button.querySelector('img');
  galleryLarge.src = thumbnail.getAttribute('src');
  const request = ++galleryImageRequest;
  const fullImage = new Image();
  fullImage.onload = () => {
    if (request === galleryImageRequest && galleryViewer.open) galleryLarge.src = fullImage.src;
  };
  fullImage.src = thumbnail.dataset.full;
  galleryLarge.alt = thumbnail.alt;
  galleryPrompt.textContent = button.dataset.prompt;
  galleryViewer.showModal();
  document.body.classList.add('gallery-modal-open');
}));
document.getElementById('gallery-close').addEventListener('click', () => galleryViewer.close());
galleryViewer.addEventListener('click', event => {
  if (event.target !== galleryViewer) return;
  const bounds = galleryViewer.getBoundingClientRect();
  if (event.clientX < bounds.left || event.clientX > bounds.right ||
      event.clientY < bounds.top || event.clientY > bounds.bottom) galleryViewer.close();
});
galleryViewer.addEventListener('close', () => {
  galleryImageRequest++;
  document.body.classList.remove('gallery-modal-open');
  galleryLarge.removeAttribute('src');
  galleryPrompt.textContent = '';
  galleryOpener?.focus({preventScroll: true});
});
galleryPause.addEventListener('click', () => {
  galleryPaused = !galleryPaused;
  galleryPause.setAttribute('aria-pressed', String(galleryPaused));
  galleryPause.textContent = galleryPaused ? 'Resume scrolling' : 'Pause scrolling';
});
gallerySection.addEventListener('mouseenter', () => { galleryHovered = true; });
gallerySection.addEventListener('mouseleave', () => { galleryHovered = false; });
galleryTrack.addEventListener('pointerdown', () => { galleryDragging = true; });
window.addEventListener('pointerup', () => { galleryDragging = false; });
window.addEventListener('pointercancel', () => { galleryDragging = false; });
document.addEventListener('visibilitychange', () => { galleryLastFrame = undefined; });
window.requestAnimationFrame(animateGallery);

// Fit the equal-height pair to the full content width using DOM box geometry.
const teaserPair = document.querySelector('.teaser-pair');
const teaserPanels = [...teaserPair.querySelectorAll('.paper-figure')];
function fitTeaserPair() {
  if (!teaserPanels.every(panel => panel.complete)) return;
  const boxes = teaserPanels.map(panel => panel.getBoundingClientRect());
  if (boxes.some(box => box.width <= 0 || box.height <= 0)) return;
  const gap = parseFloat(window.getComputedStyle(teaserPair).columnGap) || 0;
  const available = teaserPair.clientWidth - gap * (boxes.length - 1);
  const ratioSum = boxes.reduce((sum, box) => sum + box.width / box.height, 0);
  if (available > 0) teaserPair.style.setProperty('--teaser-height', `${available / ratioSum}px`);
}
teaserPanels.forEach(panel => panel.addEventListener('load', fitTeaserPair));
new ResizeObserver(fitTeaserPair).observe(teaserPair);
fitTeaserPair();
