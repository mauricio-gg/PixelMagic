/* PixelMagic surfaces page: drawing and posting only.
   The person draws shapes over a surface and drags a small box on a plain patch of it. This script turns
   pointer positions into the picture's own pixels, keeps the shapes and the box, asks the server for the
   colours a box would give, and posts one surface in the typed form the command line uses
   ("name:kind; box x0,y0 x1,y1; shape x,y x,y x,y"). The kinds, how big a box must be, the colours, what a
   name may be and every refusal are the core's: nothing here decides them, and a refusal is shown as the
   sentence the server sent. */
(function () {
  'use strict';
  const byId = (id) => document.getElementById(id);
  const stage = byId('stage');
  const picture = byId('picture');
  const overlay = byId('overlay');
  const handlesLayer = byId('handles');
  const capsLayer = byId('caps');
  const hint = byId('hint');
  const undoButton = byId('undo');
  const nameInput = byId('name');
  const kindsGroup = byId('kinds');
  const kindExample = byId('kind-example');
  const shapesList = byId('shapes');
  const boxline = byId('boxline');
  const swatches = byId('swatches');
  const coloursNote = byId('colours-note');
  const coloursError = byId('colours-error');
  const refusal = byId('refusal');
  const refusalHead = refusal.querySelector('strong');
  const refusalText = byId('refusal-text');
  const drawButton = byId('draw');
  const continueLink = byId('continue');
  const width = Number(stage.dataset.width);
  const height = Number(stage.dataset.height);
  const minBoxPixels = Number(stage.dataset.minBox);
  const geometry = JSON.parse(byId('surfaces-geometry').textContent);
  const SVG = 'http://www.w3.org/2000/svg';

  /* ---------- what has been drawn ---------- */
  const editing = geometry.editing;
  let shapes = editing ? editing.shapes.map((corners) => corners.map((p) => [p[0], p[1]])) : []; // closed shapes
  let draft = null;     // the shape being drawn: corners so far, or null
  let current = shapes.length ? 0 : -1; // the closed shape shown strong
  let box = editing ? { x0: editing.box[0], y0: editing.box[1], x1: editing.box[2], y1: editing.box[3] } : null; // as dragged: corners may be in any order
  let tool = editing ? 'move' : 'shape';
  let kind = editing ? editing.kind : kindsGroup.querySelector('[role=radio]').dataset.value;
  let nameTouched = Boolean(editing);
  let colours = null;   // { shades, shares } or { error } from the server
  let reading = false; // a colours request is out
  let running = false; // the draw request is out
  let job = null;
  let lastCorner = null; // when and where the last corner was placed, to tell a double-click or double-tap from two corners

  /* ---------- undo: one small history of snapshots, taken before each change ---------- */
  const history = [];
  const HISTORY_LIMIT = 50;
  function remember() {
    history.push(JSON.stringify({ shapes: shapes, draft: draft, current: current, box: box }));
    if (history.length > HISTORY_LIMIT) history.shift();
  }
  function undo() {
    if (running) return;
    const last = history.pop();
    if (!last) return;
    const before = boxField();
    const snapshot = JSON.parse(last);
    shapes = snapshot.shapes;
    draft = snapshot.draft;
    current = snapshot.current;
    box = snapshot.box;
    lastCorner = null;
    refresh(true);
    if (boxField() !== before) loadColours();
  }

  /* ---------- the box ---------- */
  function boxSize() {
    return box ? { w: Math.abs(box.x1 - box.x0), h: Math.abs(box.y1 - box.y0) } : { w: 0, h: 0 };
  }
  function boxBigEnough() {
    const size = boxSize();
    return size.w * size.h >= minBoxPixels;
  }
  function boxField() {
    if (!box) return '';
    return Math.min(box.x0, box.x1) + ',' + Math.min(box.y0, box.y1) + ' ' + Math.max(box.x0, box.x1) + ',' + Math.max(box.y0, box.y1);
  }
  function typedSurface() {
    const parts = [nameInput.value.trim() + ':' + kind, 'box ' + boxField()];
    shapes.forEach((corners) => parts.push('shape ' + corners.map((p) => p.join(',')).join(' ')));
    return parts.join('; ');
  }

  /* ---------- pointer to the picture's own pixels ---------- */
  function toPicture(event) {
    const rect = picture.getBoundingClientRect();
    return [Math.round((event.clientX - rect.left) * width / rect.width), Math.round((event.clientY - rect.top) * height / rect.height)];
  }
  function percent(value, whole) { return (value / whole * 100).toFixed(3) + '%'; }
  function svg(tag, attributes) {
    const node = document.createElementNS(SVG, tag);
    Object.keys(attributes).forEach((key) => node.setAttribute(key, attributes[key]));
    return node;
  }
  function html(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function pointsOf(corners) { return corners.map((p) => p.join(',')).join(' '); }

  /* follow one pointer until it lifts: used for dragging a box on the picture and for dragging a handle */
  function track(element, event, onMove, onEnd) {
    element.setPointerCapture(event.pointerId);
    function end(last) {
      element.removeEventListener('pointermove', onMove);
      element.removeEventListener('pointerup', end);
      element.removeEventListener('pointercancel', end);
      onEnd(last);
    }
    element.addEventListener('pointermove', onMove);
    element.addEventListener('pointerup', end);
    element.addEventListener('pointercancel', end);
  }

  /* ---------- the canvas: shapes and box, handles, captions ---------- */
  function renderOverlay() {
    overlay.replaceChildren();
    shapes.forEach((corners, i) => {
      overlay.append(svg('polygon', { class: 'shp' + (i === current && !draft ? ' cur' : ''), points: pointsOf(corners) }));
    });
    if (draft) overlay.append(svg('polyline', { class: 'shp draft', points: pointsOf(draft) }));
    if (box) {
      const size = boxSize();
      overlay.append(svg('rect', {
        class: 'bx' + (boxBigEnough() ? '' : ' bad'),
        x: Math.min(box.x0, box.x1), y: Math.min(box.y0, box.y1), width: size.w, height: size.h,
      }));
    }
  }

  let handles = []; // { button, point } for each handle on the canvas
  function placeHandles() {
    handles.forEach((handle) => {
      const point = handle.point();
      handle.button.style.left = percent(point[0], width);
      handle.button.style.top = percent(point[1], height);
    });
  }
  /* a handle is a button, so it can be reached and moved from the keyboard (arrow keys move it 4 pixels) */
  function addHandle(label, className, point, move) {
    const button = html('button', 'hdl ' + className);
    button.type = 'button';
    button.setAttribute('aria-label', label);
    button.addEventListener('pointerdown', (event) => {
      if (running || tool !== 'move' || (event.pointerType === 'mouse' && event.button !== 0)) return;
      event.preventDefault();
      event.stopPropagation();
      remember();
      button.classList.add('drag');
      track(button, event, (moved) => { move(toPicture(moved)); updateGeometry(); }, () => {
        button.classList.remove('drag');
        refresh(false);
        if (box) loadColours();
      });
    });
    button.addEventListener('keydown', (event) => {
      const step = { ArrowLeft: [-4, 0], ArrowRight: [4, 0], ArrowUp: [0, -4], ArrowDown: [0, 4] }[event.key];
      if (!step || running) return;
      event.preventDefault();
      const now = point();
      remember();
      move([now[0] + step[0], now[1] + step[1]]);
      refresh(false);
      if (box) loadColours();
    });
    handles.push({ button: button, point: point });
    handlesLayer.append(button);
    return button;
  }
  function buildHandles() {
    handlesLayer.replaceChildren();
    handles = [];
    shapes.forEach((corners, si) => {
      corners.forEach((corner, ci) => {
        const button = addHandle('Shape ' + (si + 1) + ', corner ' + (ci + 1) + '. Drag to move.', si === current && !draft ? '' : 'dim',
          () => corner, (p) => { corner[0] = p[0]; corner[1] = p[1]; });
        button.dataset.shape = si;
      });
    });
    if (draft) {
      draft.forEach((corner, ci) => {
        const closes = ci === 0 && tool === 'shape' && draft.length >= 3;
        const button = addHandle('New shape, corner ' + (ci + 1) + (closes ? '. Press to close the shape.' : '. Drag to move.'), closes ? 'close' : '',
          () => corner, (p) => { corner[0] = p[0]; corner[1] = p[1]; });
        if (closes) button.addEventListener('click', closeShape);
      });
    }
    if (box) {
      [['x0', 'y0'], ['x1', 'y0'], ['x0', 'y1'], ['x1', 'y1']].forEach((keys, i) => {
        addHandle('Colour box, corner ' + (i + 1) + ' of 4. Drag to resize.', 'box',
          () => [box[keys[0]], box[keys[1]]], (p) => { box[keys[0]] = p[0]; box[keys[1]] = p[1]; });
      });
    }
    placeHandles();
  }

  function caption(text, point, className, offset) {
    const cap = html('span', 'cap' + (className ? ' ' + className : ''), text);
    cap.style.left = percent(point[0], width);
    cap.style.top = 'calc(' + percent(point[1], height) + ' + ' + offset + 'px)';
    capsLayer.append(cap);
  }
  function renderCaps() {
    capsLayer.replaceChildren();
    shapes.forEach((corners, i) => {
      const top = corners.reduce((a, b) => (b[1] < a[1] ? b : a));
      caption('Shape ' + (i + 1), [top[0] + 50, top[1]], '', -40);
    });
    if (box) {
      const size = boxSize();
      caption(size.w + ' x ' + size.h + ' px', [(box.x0 + box.x1) / 2, Math.max(box.y0, box.y1)], boxBigEnough() ? '' : 'bad', 22);
    }
  }

  /* ---------- the panel: shapes, box line, colours ---------- */
  function corners(count) { return count + (count === 1 ? ' corner' : ' corners'); }
  function renderShapes() {
    shapesList.replaceChildren();
    shapes.forEach((points, i) => {
      const row = html('li', 'row');
      const pick = html('button', 'btn quiet pick-shape');
      pick.type = 'button';
      pick.setAttribute('aria-pressed', i === current ? 'true' : 'false');
      pick.append(html('span', 'dotc' + (i === current ? ' on' : '')), document.createTextNode('Shape ' + (i + 1) + ' '), html('span', 'muted', '· ' + corners(points.length)));
      pick.addEventListener('click', () => { current = i; refresh(true); });
      const remove = html('button', 'btn quiet', 'Remove');
      remove.type = 'button';
      remove.setAttribute('aria-label', 'Remove shape ' + (i + 1));
      remove.addEventListener('click', () => {
        if (running) return;
        remember();
        shapes.splice(i, 1);
        current = Math.min(current, shapes.length - 1);
        refresh(true);
      });
      row.append(pick, remove);
      shapesList.append(row);
    });
    if (draft) {
      const row = html('li', 'foot-note', 'Drawing a new shape · ' + corners(draft.length) + (draft.length >= 3 ? '. Press Enter to close it.' : '. It needs at least 3 corners.'));
      shapesList.append(row);
    } else if (!shapes.length) {
      shapesList.append(html('li', 'foot-note', 'No shapes yet. Choose Draw shape and click corners around the surface.'));
    }
  }

  /* the box line is built once and its text changed in place, so a drag does not make a screen reader repeat itself */
  const boxTitle = html('span', null, 'Colour box');
  const boxSizeText = html('span', 'muted');
  const boxHead = html('div', 'head');
  boxHead.append(boxTitle, boxSizeText);
  const boxStrip = html('p', 'strip');
  boxStrip.setAttribute('role', 'status');
  const boxIcon = svg('svg', { class: 'i', 'aria-hidden': 'true' });
  const boxIconUse = svg('use', { href: '#i-check' });
  boxIcon.append(boxIconUse);
  const boxStripText = html('span');
  boxStrip.append(boxIcon, boxStripText);
  const boxEmpty = html('p', 'foot-note', 'Choose Colour box and drag across a plain patch of the surface.');
  boxline.append(boxHead, boxStrip, boxEmpty);
  function setText(node, text) { if (node.textContent !== text) node.textContent = text; }
  function renderBox() {
    const big = boxBigEnough();
    boxStrip.hidden = !box;
    boxEmpty.hidden = Boolean(box);
    if (!box) { setText(boxSizeText, 'not drawn yet'); return; }
    const size = boxSize();
    setText(boxSizeText, size.w + ' x ' + size.h + ' px');
    boxStrip.className = 'strip ' + (big ? 'ok' : 'warn');
    boxIconUse.setAttribute('href', big ? '#i-check' : '#i-warn');
    setText(boxStripText, big ? 'Big enough for true colours.' : 'A little bigger please: at least 1024 pixels, about 32 by 32, so the colours are true.');
  }

  const COUNT_WORDS = { 4: 'four', 5: 'five' };
  function renderColours() {
    const big = boxBigEnough();
    const failed = colours && colours.error;
    coloursError.hidden = !failed;
    if (failed) setText(coloursError.querySelector('span'), colours.error);
    swatches.hidden = !colours || failed;
    if (!swatches.hidden) {
      swatches.replaceChildren(...colours.shades.map((shade) => { const chip = html('i'); chip.style.background = shade; return chip; }));
      swatches.classList.toggle('dim', !big);
    }
    if (failed) setText(coloursNote, '');
    else if (reading) setText(coloursNote, 'Reading the colours from the box.');
    else if (big && colours) {
      const count = colours.shades.length;
      setText(coloursNote, 'The ' + (COUNT_WORDS[count] || count) + ' colours the ' + kind + ' will be drawn with.');
    } else setText(coloursNote, 'The colours show up once the box is big enough.');
  }

  /* the server reads the colours; a late answer for an older box or kind is ignored */
  let coloursRequest = 0;
  function loadColours() {
    const mine = ++coloursRequest;
    if (!box || !boxBigEnough()) {
      reading = false;
      if (colours && colours.error) colours = null;
      renderColours();
      return;
    }
    reading = true;
    renderColours();
    fetch(stage.dataset.coloursUrl, { method: 'POST', body: new URLSearchParams({ kind: kind, box: boxField(), name: nameInput.value.trim() }) })
      .then((reply) => reply.json())
      .then((body) => { if (mine === coloursRequest) colours = body.error ? { error: body.error } : { shades: body.shades, shares: body.shares }; })
      .catch(() => { if (mine === coloursRequest) colours = { error: 'Something went wrong reading the colours.' }; })
      .finally(() => { if (mine === coloursRequest) { reading = false; renderColours(); } });
  }

  /* ---------- the main button: Draw this surface, or Continue to Refine when there is nothing being drawn ---------- */
  function renderAction() {
    const empty = !shapes.length && !draft && !box;
    const way = empty && geometry.drawn_count > 0;
    continueLink.hidden = !way;
    if (running) return;
    drawButton.hidden = way;
    drawButton.disabled = !nameInput.value.trim() || !shapes.length || !boxBigEnough();
  }

  function showRefusal(sentence, fromCore) {
    refusalHead.hidden = !fromCore;
    refusalText.textContent = sentence;
    refusal.hidden = false;
    refusal.scrollIntoView({ block: 'nearest' });
  }
  function clearRefusal() { refusal.hidden = true; }

  /* everything that follows a change; `rebuild` also remakes the handles (after a shape is added, closed, removed or undone) */
  function refresh(rebuild) {
    clearRefusal();
    renderOverlay();
    if (rebuild) buildHandles(); else placeHandles();
    renderCaps();
    renderShapes();
    renderBox();
    renderColours();
    renderAction();
    undoButton.disabled = running || history.length === 0;
  }
  /* the lighter version for each step of a drag */
  function updateGeometry() {
    renderOverlay();
    placeHandles();
    renderCaps();
    renderBox();
    renderColours();
    renderAction();
  }

  /* ---------- the tools ---------- */
  const HINTS = {
    move: ['Drag a corner to move it. Pick a tool to draw more.', 'Drag a corner to move it. Pick a tool to draw more.'],
    shape: ['Click to place a corner. Enter closes the shape.', 'Tap to place a corner. Double-tap closes.'],
    box: ['Drag across a plain patch of the surface.', 'Drag across a plain patch of the surface.'],
  };
  const selectTool = PixelMagic.radiogroup(byId('tools'), setTool);
  function setTool(value) {
    tool = value;
    selectTool(value);
    stage.classList.toggle('drawing', value !== 'move');
    stage.classList.toggle('boxing', value === 'box');
    PixelMagic.setHint(hint, HINTS[value][0], HINTS[value][1]);
    buildHandles();
  }

  const DOUBLE_MS = 400;  // a second press this soon and this close is a double-click or double-tap
  const DOUBLE_PIXELS = 12;
  function closeShape() {
    if (running || !draft || draft.length < 3) return;
    remember();
    shapes.push(draft);
    current = shapes.length - 1;
    draft = null;
    lastCorner = null;
    refresh(true);
  }
  function placeCorner(event) {
    const double = draft && lastCorner && event.timeStamp - lastCorner.time < DOUBLE_MS
      && Math.hypot(event.clientX - lastCorner.x, event.clientY - lastCorner.y) < DOUBLE_PIXELS;
    if (double) { closeShape(); return; }
    remember();
    draft = draft || [];
    draft.push(toPicture(event));
    lastCorner = { time: event.timeStamp, x: event.clientX, y: event.clientY };
    refresh(true);
  }

  const MIN_DRAG_PIXELS = 4; // a press that hardly moved is a stray click, not a box
  function startBox(event) {
    const before = box ? Object.assign({}, box) : null;
    remember();
    const start = toPicture(event);
    box = { x0: start[0], y0: start[1], x1: start[0], y1: start[1] };
    refresh(true);
    track(stage, event, (moved) => {
      const point = toPicture(moved);
      box.x1 = point[0];
      box.y1 = point[1];
      updateGeometry();
    }, (last) => {
      if (Math.hypot(last.clientX - event.clientX, last.clientY - event.clientY) < MIN_DRAG_PIXELS) {
        history.pop();
        box = before;
      }
      refresh(true);
      loadColours();
    });
  }

  stage.addEventListener('pointerdown', (event) => {
    if (running || event.target.closest('.hdl') || (event.pointerType === 'mouse' && event.button !== 0)) return;
    if (tool === 'shape') placeCorner(event);
    else if (tool === 'box') startBox(event);
  });
  undoButton.addEventListener('click', undo);

  document.addEventListener('keydown', (event) => {
    if (running || !draft) return;
    if (event.key === 'Escape') {
      remember();
      draft = null;
      lastCorner = null;
      refresh(true);
      return;
    }
    /* Enter closes the shape, except where it already means something: in the panel, or on a button other than a handle */
    const target = event.target;
    const elsewhere = target.closest('.panel') || (target.closest('button') && !target.classList.contains('hdl'));
    if (event.key === 'Enter' && tool === 'shape' && !elsewhere) {
      event.preventDefault();
      closeShape();
    }
  });

  /* ---------- name and kind ---------- */
  function kindButton(value) { return kindsGroup.querySelector('[data-value="' + value + '"]'); }
  const selectKind = PixelMagic.radiogroup(kindsGroup, setKind);
  function setKind(value) {
    const previous = kindButton(kind).dataset.name;
    kind = value;
    selectKind(value);
    kindExample.textContent = kindButton(value).dataset.example;
    if (!nameTouched || nameInput.value === previous) {
      nameInput.value = kindButton(value).dataset.name;
      nameTouched = false;
    }
    clearRefusal();
    loadColours();
    renderAction();
  }
  nameInput.addEventListener('input', () => { nameTouched = true; clearRefusal(); renderAction(); });

  /* ---------- drawing: one request, an honest working strip, then the server's next page ---------- */
  /* editing a surface replaces it, so the old name goes along; the core decides whether a name is free */
  function drawFields() {
    const fields = { surface: typedSurface() };
    if (editing) fields.replaces = editing.name;
    return fields;
  }
  function failed(sentence, fromCore) {
    PixelMagic.stopJob(job);
    job = null;
    running = false;
    nameInput.disabled = false;
    delete stage.dataset.busy;
    refresh(true);
    showRefusal(sentence, fromCore);
  }
  drawButton.addEventListener('click', () => {
    if (running) return;
    running = true;
    nameInput.disabled = true;
    stage.dataset.busy = '1';
    clearRefusal();
    undoButton.disabled = true;
    job = PixelMagic.startJob([drawButton], 'Drawing ' + nameInput.value.trim(), 'Reading the colours, drawing the 32 tiles and checking that they join.');
    fetch(stage.dataset.drawUrl, { method: 'POST', body: new URLSearchParams(drawFields()) })
      .then((reply) => reply.json())
      .then((body) => {
        if (typeof body.next === 'string') location.assign(body.next);
        else if (typeof body.error === 'string') failed(body.error, true);
        else failed('Something went wrong drawing it.', false);
      })
      .catch(() => failed('Something went wrong drawing it.', false));
  });
  /* the page may come back from the back-forward cache while the request was out; the shared script has put the button back */
  window.addEventListener('pageshow', (event) => {
    if (!event.persisted || !running) return;
    running = false;
    job = null;
    nameInput.disabled = false;
    delete stage.dataset.busy;
    refresh(true);
  });

  /* removing a surface deletes its 32 tiles and their assets */
  document.querySelectorAll('form[data-confirm]').forEach((form) => {
    form.addEventListener('submit', (event) => { if (!confirm(form.dataset.confirm)) event.preventDefault(); });
  });

  /* ---------- start ---------- */
  const fresh = document.querySelector('.rows.drawn .row.fresh');
  if (fresh) fresh.scrollIntoView({ block: 'nearest' });
  selectKind(kind);
  kindExample.textContent = kindButton(kind).dataset.example;
  nameInput.value = editing ? editing.text : kindButton(kind).dataset.name;
  setTool(tool);
  refresh(true);
  loadColours();
})();
