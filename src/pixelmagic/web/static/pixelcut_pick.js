/* PixelMagic pixel-cut picking page: recording and posting only.
   A drag in Box mode is a box round a prop; a click in Add or Remove mode is a spot to add to or take from
   the selected pick. Holding Cmd or Ctrl switches to Add, and Option or Alt to Remove, only while the key is
   held. Everything is posted in picture pixels as a hidden form, so every answer is the server's redirect or
   sentence. What a box means, the 12 px guard, the cut and every refusal are the server's: nothing here
   decides them. */
(function () {
  'use strict';
  const stage = document.getElementById('stage');
  const picture = document.getElementById('picture');
  const ghost = document.getElementById('ghost');
  const hint = document.getElementById('hint');
  const modes = document.getElementById('modes');
  const undoForm = document.getElementById('undo-form');
  const base = stage.dataset.base;
  const width = Number(stage.dataset.width);
  const height = Number(stage.dataset.height);
  const selected = stage.dataset.selected;
  const CLICK_PIXELS = 6; // a press that moved less than this on screen is a click, not a box

  /* ---------- posting: a hidden form; one send at a time ---------- */
  let posting = false;
  function post(path, fields) {
    if (posting) return;
    posting = true;
    stage.setAttribute('aria-busy', 'true');
    const form = document.createElement('form');
    form.method = 'post';
    form.action = base + path;
    Object.keys(fields).forEach((key) => {
      const input = document.createElement('input');
      input.type = 'hidden';
      input.name = key;
      input.value = fields[key];
      form.appendChild(input);
    });
    document.body.appendChild(form);
    form.submit();
  }
  /* Undo shares the single-send guard: a double-click must undo one step, not two */
  undoForm.addEventListener('submit', (event) => {
    if (posting) { event.preventDefault(); return; }
    posting = true;
    stage.setAttribute('aria-busy', 'true');
  });
  /* the page may come back from the back-forward cache after a post */
  window.addEventListener('pageshow', (event) => {
    if (!event.persisted) return;
    posting = false;
    stage.removeAttribute('aria-busy');
  });

  /* ---------- picture pixels ---------- */
  function clamp(value, whole) { return Math.min(whole, Math.max(0, value)); }
  /* the picture pixel under the pointer (a cell), for a spot */
  function cellAt(event) {
    const rect = picture.getBoundingClientRect();
    return {
      x: Math.floor((event.clientX - rect.left) * width / rect.width),
      y: Math.floor((event.clientY - rect.top) * height / rect.height),
    };
  }
  /* the picture pixel line nearest the pointer, clamped to the picture, for a box corner */
  function cornerAt(event) {
    const rect = picture.getBoundingClientRect();
    return {
      x: clamp(Math.round((event.clientX - rect.left) * width / rect.width), width),
      y: clamp(Math.round((event.clientY - rect.top) * height / rect.height), height),
    };
  }
  function boxOf(a, b) {
    return { left: Math.min(a.x, b.x), top: Math.min(a.y, b.y), right: Math.max(a.x, b.x), bottom: Math.max(a.y, b.y) };
  }

  /* ---------- the mode and its hint ---------- */
  const HINTS = {
    box: ['Drag a box round a prop. One click alone does not pick it. Then use Add or Remove to fix a spot.',
          'Drag a box round a prop with your finger. One tap alone does not pick it.'],
    add: ['Click a part of the selected prop that was missed.', 'Tap a part of the selected prop that was missed.'],
    remove: ['Click a part of the selected prop that should not be there.',
             'Tap a part of the selected prop that should not be there.'],
  };
  let mode = 'box';
  function say(key) {
    hint.classList.remove('flash');
    PixelMagic.setHint(hint, HINTS[key][0], HINTS[key][1]);
  }
  const selectRadio = PixelMagic.radiogroup(modes, (value) => choose(value, true));
  let chosen = 'box'; // the tool the person picked in the tray: what a held key goes back to
  /* show a tool; with nothing selected Add and Remove fall back to Box */
  function setMode(value, held) {
    if (!selected && value !== 'box') value = 'box';
    mode = value;
    selectRadio(value, held);
    stage.classList.toggle('boxing', value === 'box');
    say(value);
  }
  /* the person's own choice: kept for the session, unlike a held key's tool */
  function choose(value, remember) {
    setMode(value);
    chosen = mode;
    if (remember) {
      try { sessionStorage.setItem('pm-pixelcut-mode', value); } catch (e) { /* storage blocked: the mode just does not persist */ }
    }
  }
  let remembered = 'box';
  try { remembered = sessionStorage.getItem('pm-pixelcut-mode') || 'box'; } catch (e) { /* storage blocked */ }
  choose(remembered, false);
  PixelMagic.holdKeys(stage, modes, { primary: 'add', alt: 'remove' }, (tool) => {
    if (!tool) setMode(chosen);
    else if (!selected) say('box'); // keep the current tool; Box's hint says to pick first
    else setMode(tool, true);
  });

  /* ---------- pointer events: drag a box, or click a spot ---------- */
  let down = null; // the press now under way: where it began, on screen and in picture pixels
  function hideGhost() { ghost.hidden = true; }
  function drawGhost(box) {
    ghost.style.left = (box.left / width * 100) + '%';
    ghost.style.top = (box.top / height * 100) + '%';
    ghost.style.width = ((box.right - box.left) / width * 100) + '%';
    ghost.style.height = ((box.bottom - box.top) / height * 100) + '%';
    ghost.hidden = false;
  }
  /* has the pointer moved far enough from where the given press began to be a drag? */
  function moved(press, event) { return Math.hypot(event.clientX - press.x, event.clientY - press.y) >= CLICK_PIXELS; }

  stage.addEventListener('pointerdown', (event) => {
    if (posting || !event.isPrimary || (event.pointerType === 'mouse' && event.button !== 0)) return;
    down = { x: event.clientX, y: event.clientY, corner: cornerAt(event), mode: mode, box: mode === 'box' };
    if (down.box) stage.setPointerCapture(event.pointerId);
  });
  stage.addEventListener('pointermove', (event) => {
    if (!down || !down.box || !moved(down, event)) return;
    drawGhost(boxOf(down.corner, cornerAt(event)));
  });
  stage.addEventListener('pointercancel', () => { down = null; hideGhost(); });
  stage.addEventListener('pointerup', (event) => {
    const start = down;
    down = null;
    hideGhost();
    if (posting || !start || !event.isPrimary) return;
    if (start.box && moved(start, event)) {
      const box = boxOf(start.corner, cornerAt(event));
      if (box.right > box.left && box.bottom > box.top) {
        post('/pick/box', box);
        return;
      }
    } else if (moved(start, event)) {
      return; // a drag in Add or Remove mode is not a click
    }
    const cell = cellAt(event);
    if (cell.x < 0 || cell.y < 0 || cell.x >= width || cell.y >= height) return;
    if (start.mode !== 'box') {
      /* a spot on the selected pick: the server's points route runs the 12 px guard and widens the box */
      post('/picks/' + selected + '/points', { x: cell.x, y: cell.y, label: start.mode === 'remove' ? 0 : 1 });
      return;
    }
    hint.classList.add('flash');
    PixelMagic.setHint(hint, 'That was a click. On pixel art a prop is picked with a box: drag one round it.',
      'That was a tap. On pixel art a prop is picked with a box: drag one round it.');
  });
})();
