// Logic-only tests: fake network and decoding; no image files are opened.
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../static/resolution-gallery.js'), 'utf8');
function setup() {
  const nodes = new Map();
  function node() {
    const attrs = new Map();
    return {addEventListener() {}, classList: {toggle() {}, remove() {}},
      setAttribute(k,v) {attrs.set(k,v);}, getAttribute(k) {return attrs.get(k);}};
  }
  const window = {addEventListener() {}};
  vm.runInNewContext(source, {window, document: {getElementById(id) {
    if (!nodes.has(id)) nodes.set(id,node()); return nodes.get(id);
  }}, AbortController, DOMException, URL: {revokeObjectURL() {}},
    setTimeout, clearTimeout, requestAnimationFrame: fn => setTimeout(fn,0), cancelAnimationFrame: clearTimeout});
  const dialog = {open:true};
  const image = {};
  const gallery = new window.ResolutionGallery(dialog,image);
  const sample = {original:'gallery/test.png',frames:Array.from({length:57},(_,i) => {
    const size=512+i*64;
    return {size,width:size,height:size,preview:`p${size}`,full:`f${size}`};
  })};
  gallery.sample = sample;
  gallery.controller = new AbortController();
  return {gallery,image,dialog,sample};
}
const tick = () => new Promise(resolve => setImmediate(resolve));

test('57 size options and drag do not fetch full-size images', async () => {
  const {gallery,sample} = setup();
  gallery.dragging = true;
  let fullRequests = 0;
  gallery.preview = async () => ({url:'preview'});
  gallery.loadFull = () => {fullRequests++;};
  for (const frame of sample.frames) gallery.select(frame.size);
  await new Promise(resolve => setTimeout(resolve,250));
  assert.equal(fullRequests,0);
  assert.equal(gallery.output.textContent,'4096 × 4096');
  assert.equal(sample.frames.length,57);
  gallery.close();
});

test('late preview cannot replace a newer selection or a full-resolution result', async () => {
  const {gallery,image} = setup();
  gallery.dragging = true;
  const resolves = new Map();
  gallery.preview = (url,urgent) => urgent ? new Promise(resolve => resolves.set(url,resolve)) : Promise.resolve(null);
  gallery.select(512);
  gallery.select(576);
  resolves.get('p576')({url:'new'});
  await tick();
  resolves.get('p512')({url:'old'});
  await tick();
  assert.equal(image.src,'new');
  gallery.select(640);
  gallery.fullReady = true;
  image.src = 'full';
  resolves.get('p640')({url:'late'});
  await tick();
  assert.equal(image.src,'full');
  gallery.close();
});

test('full-resolution completion is rejected after selection changes or close', async () => {
  const {gallery,image,sample} = setup();
  let resolve;
  gallery.decode = () => new Promise(r => {resolve=r;});
  const old = gallery.loadFull(sample.frames[0],gallery.session,gallery.selection);
  gallery.selection++;
  resolve({url:'stale'});
  await old;
  assert.equal(image.src,undefined);
  const pending = gallery.loadFull(sample.frames[1],gallery.session,gallery.selection);
  gallery.close();
  resolve({url:'closed'});
  await pending;
  assert.equal(image.src,undefined);
});

test('preview loader limits concurrent loads and decoded cache', async () => {
  const {gallery} = setup();
  let active=0,max=0;
  gallery.decode = async url => {
    active++;max=Math.max(max,active);
    await new Promise(resolve => setTimeout(resolve,2));
    active--;return {url};
  };
  await Promise.all(Array.from({length:30},(_,i) => gallery.preview(`p${i}`)));
  assert.equal(max,3);
  assert.equal(gallery.cache.size,12);
  gallery.close();
  assert.equal(gallery.cache.size,0);
});

test('manifest failure retains original and disables unavailable slider', async () => {
  const {gallery,image} = setup();
  gallery.manifest = async () => {throw Error('offline');};
  await gallery.open('gallery/original.png');
  assert.equal(image.src,'gallery/original.png');
  assert.equal(gallery.slider.disabled,true);
  gallery.close();
});
