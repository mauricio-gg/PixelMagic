/* PixelMagic pick-props page: lookup, drawing and posting only.
   The server renders a picture where each pixel says which suggestion lies there (red * 256 + green,
   0 for none) and one outline picture per suggestion. This script reads the pixel under the pointer,
   shows that outline and posts what the person did, in picture pixels. Scrolling and the Bigger and
   Smaller buttons ask the server for the suggestions under a spot (smallest first) and step through
   that list. A dragged Lasso line is recorded, drawn and posted once. Holding a key picks a tool only while it is
   held (Cmd or Ctrl: Add a spot; Option or Alt: Remove a spot; Shift: Lasso). A click means what the tool at that
   moment says. What it does, the 12 px guard and every refusal are the server's: nothing here decides them. */
(function () {
  'use strict';
  const stage = document.getElementById('stage');
  const picture = document.getElementById('picture');
  const hover = document.getElementById('hover');
  const trail = document.getElementById('trail');
  const nametag = document.getElementById('nametag');
  const hint = document.getElementById('hint');
  const live = document.getElementById('live');
  const modes = document.getElementById('modes');
  const bigger = document.getElementById('bigger');
  const smaller = document.getElementById('smaller');
  const base = stage.dataset.base;
  const width = Number(stage.dataset.width);
  const height = Number(stage.dataset.height);
  const selected = stage.dataset.selected;
  const selectedByLasso = stage.dataset.selectedMethod === 'lasso';

  /* ---------- posting: a hidden form, so every answer is the server's redirect or sentence ---------- */
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
  const undoForm = document.getElementById('undo-form');
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

  /* ---------- the lookup picture ---------- */
  let lookup = null;
  const map = new Image();
  map.onload = () => {
    const canvas = document.createElement('canvas');
    canvas.width = map.naturalWidth;
    canvas.height = map.naturalHeight;
    lookup = canvas.getContext('2d', { willReadFrequently: true });
    lookup.drawImage(map, 0, 0);
  };
  map.src = base + '/pick/map.png';

  function at(event) {
    const rect = picture.getBoundingClientRect();
    return {
      x: Math.floor((event.clientX - rect.left) * width / rect.width),
      y: Math.floor((event.clientY - rect.top) * height / rect.height),
    };
  }
  function inside(point) { return point.x >= 0 && point.y >= 0 && point.x < width && point.y < height; }
  function suggestionAt(point) {
    if (!lookup || !inside(point)) return 0;
    const pixel = lookup.getImageData(point.x, point.y, 1, 1).data;
    return pixel[0] * 256 + pixel[1];
  }
  function numberOfTag(target) {
    const tag = target.closest('button.otag[data-n]');
    return tag ? Number(tag.dataset.n) : 0;
  }

  /* ---------- the preview: an outline and a name tag for one suggestion ---------- */
  let mode = 'pick';
  let shown = 0;          // the suggestion outlined now, 0 for none
  let spot = null;        // where the pointer last was (or the last tap): picture pixels and client pixels
  let steps = null;       // the server's list of suggestions at the spot, and where we are in it
  let fetching = false;

  function percent(value, whole) { return (Math.min(Math.max(value / whole, 0.08), 0.92) * 100).toFixed(2) + '%'; }
  function tellName(html, left, top, text) {
    nametag.innerHTML = html ? '<div class="nametag" style="left:' + left + ';top:' + top + '">' + html + '</div>' : '';
    live.textContent = text || '';
  }
  function show(number) {
    if (number !== shown) {
      shown = number;
      if (number) {
        hover.src = base + '/suggestions/' + number + '/outline.png';
        hover.hidden = false;
      } else {
        hover.hidden = true;
      }
    }
    if (number && spot) {
      tellName('<b>Object ' + number + '</b><span class="h-mouse">click to pick</span><span class="h-touch">tap again to pick</span>',
        percent(spot.x, width), percent(spot.y, height), 'Object ' + number + ' outlined');
    } else {
      tellName('', '', '', '');
    }
  }
  function showPick(link) {
    show(0);
    const left = link.style.left, top = link.style.top;
    tellName('<b></b><span>picked</span>', left, top, '');
    nametag.querySelector('b').textContent = link.dataset.label;
  }
  function sizerState() {
    const on = mode === 'pick' && spot !== null;
    bigger.disabled = !on;
    smaller.disabled = !on;
  }
  function look(event) {
    const point = at(event);
    spot = { x: point.x, y: point.y, cx: event.clientX, cy: event.clientY };
    steps = null;
    sizerState();
    return point;
  }

  /* ---------- stepping to bigger and smaller objects ---------- */
  function move(direction) {
    if (!steps.list.length) return;
    steps.index = Math.min(steps.list.length - 1, Math.max(0, steps.index + direction));
    show(steps.list[steps.index]);
  }
  function step(direction) {
    if (!spot || fetching) return;
    if (steps) { move(direction); return; }
    fetching = true;
    fetch(base + '/pick/under?x=' + spot.x + '&y=' + spot.y)
      .then((reply) => {
        if (!reply.ok) throw new Error('pick/under answered ' + reply.status);
        return reply.json();
      })
      .then((list) => {
        steps = { list: list, index: Math.max(0, list.indexOf(shown)) };
        move(direction);
      })
      .catch(console.error)
      .finally(() => { fetching = false; });
  }
  bigger.addEventListener('click', () => step(1));
  smaller.addEventListener('click', () => step(-1));

  const STEP_MS = 250;   // at most one wheel step per 250 ms, so a trackpad flick moves a level or two
  const SPOT_PIXELS = 6; // moving the pointer further than this starts over at the smallest
  let lastStep = 0;
  stage.addEventListener('wheel', (event) => {
    if (mode !== 'pick') return;
    event.preventDefault();
    const now = Date.now();
    if (fetching || now - lastStep < STEP_MS || event.deltaY === 0) return;
    lastStep = now;
    if (!spot || Math.hypot(event.clientX - spot.cx, event.clientY - spot.cy) > SPOT_PIXELS) {
      show(suggestionAt(look(event)));
    }
    step(event.deltaY < 0 ? 1 : -1);
  }, { passive: false });

  /* ---------- the mode ---------- */
  const HINTS = {
    pick: ['Hover to outline. Scroll to resize. Click to pick.', 'Tap to preview. Tap again to pick.'],
    add: ['Click a missing part of an object to add that spot.', 'Tap a missing part to add that spot.'],
    remove: ['Click a part that is not the object to remove that spot.', 'Tap a part that is not the object to remove it.'],
    lasso: ['Drag a loop around an object made of several parts.', 'Draw a loop around an object made of several parts.'],
    none: ['Pick something first, then add or remove a spot.', 'Pick something first, then add or remove a spot.'],
  };
  function setHint(key) { PixelMagic.setHint(hint, HINTS[key][0], HINTS[key][1]); }
  const selectRadio = PixelMagic.radiogroup(modes, (value) => choose(value, true));
  let chosen = 'pick'; // the tool the person picked in the tray: what a held key goes back to
  /* show a tool; with nothing selected Add and Remove fall back to Pick */
  function setMode(value, held) {
    if (!selected && (value === 'add' || value === 'remove')) value = 'pick';
    mode = value;
    selectRadio(value, held);
    stage.classList.toggle('lasso', value === 'lasso');
    if (value !== 'pick') { spot = null; steps = null; show(0); }
    sizerState();
    setHint(value);
  }
  /* the person's own choice: kept for the session, unlike a held key's tool */
  function choose(value, remember) {
    setMode(value);
    chosen = mode;
    if (remember) {
      try { sessionStorage.setItem('pm-pick-mode', value); } catch (e) { /* storage blocked: the mode just does not persist */ }
    }
  }
  let remembered = 'pick';
  try { remembered = sessionStorage.getItem('pm-pick-mode') || 'pick'; } catch (e) { /* storage blocked */ }
  choose(remembered, false);
  PixelMagic.holdKeys(stage, modes, { primary: 'add', alt: 'remove', shift: 'lasso' }, (tool) => {
    if (!tool) {
      setMode(chosen);
      const item = document.activeElement.closest('.found-item'); // a focused list item gets its outline back
      if (item) show(Number(item.dataset.n));
    } else if (!selected && (tool === 'add' || tool === 'remove')) setHint('none'); // keep the current tool
    else setMode(tool, true);
  });

  /* ---------- the Lasso: record the dragged line in picture pixels, draw it, post it on release ---------- */
  const MIN_MOVE = 4; // display pixels between recorded points
  let loop = null;    // the points of the line being dragged, or null
  let lastRecorded = null;
  function record(event) {
    if (lastRecorded && Math.hypot(event.clientX - lastRecorded.x, event.clientY - lastRecorded.y) < MIN_MOVE) return;
    lastRecorded = { x: event.clientX, y: event.clientY };
    const point = at(event);
    loop.push([point.x, point.y]);
    trail.setAttribute('points', loop.map((p) => p[0] + ',' + p[1]).join(' '));
  }

  /* ---------- pointer events: hover, tap, click, drag ---------- */
  let down = null; // where the current press began, to tell a tap from a drag or a scroll

  stage.addEventListener('pointerdown', (event) => {
    if (posting || (event.pointerType === 'mouse' && event.button !== 0)) return;
    if (mode === 'lasso' && event.isPrimary) {
      event.preventDefault();
      stage.setPointerCapture(event.pointerId);
      loop = [];
      lastRecorded = null;
      trail.setAttribute('points', '');
      record(event);
      return;
    }
    down = { x: event.clientX, y: event.clientY, mode: mode }; // the tool when the press began, so a key let go mid-click changes nothing
  });

  stage.addEventListener('pointermove', (event) => {
    if (loop) { record(event); return; }
    if (event.pointerType === 'touch' || mode !== 'pick' || posting) return;
    if (steps && spot && Math.hypot(event.clientX - spot.cx, event.clientY - spot.cy) <= SPOT_PIXELS) return;
    const link = event.target.closest('a.otag');
    if (link) { look(event); showPick(link); return; }
    const point = look(event);
    show(numberOfTag(event.target) || suggestionAt(point));
  });
  /* the pointer leaving for the tray keeps the preview, so Bigger and Smaller have something to act on */

  stage.addEventListener('pointercancel', () => { loop = null; down = null; trail.setAttribute('points', ''); });

  stage.addEventListener('pointerup', (event) => {
    if (posting) return;
    if (loop) {
      record(event);
      const line = loop;
      loop = null;
      trail.setAttribute('points', '');
      if (line.length < 3) return;
      hint.textContent = 'Finding the object inside the line…';
      post('/pick/lasso', { loop: JSON.stringify(line) });
      return;
    }
    const start = down;
    down = null;
    if (!start || Math.hypot(event.clientX - start.x, event.clientY - start.y) > 10) return;
    if (event.target.closest('a.otag')) return; // a picked tag is a link to its row
    const point = at(event);
    if (!inside(point)) return;
    if (start.mode === 'add' || start.mode === 'remove') {
      fixSpot(point, start.mode === 'add' ? 1 : 0);
    } else if (start.mode === 'pick') {
      pickAt(event, point);
    }
  });

  /* a keyboard press on a number tag: the same as a click on its object */
  stage.addEventListener('click', (event) => {
    const number = event.detail === 0 && mode === 'pick' ? numberOfTag(event.target) : 0;
    if (number) post('/pick/suggestion/' + number, {});
  });

  /* Add or remove a spot on the selected pick. On a pick the lasso made, the server's lasso-fix
     adds or removes a whole piece; on any other it adds a click. */
  function fixSpot(point, label) {
    if (!selected) { setHint('none'); return; }
    post('/picks/' + selected + (selectedByLasso ? '/lasso-fix' : '/points'), { x: point.x, y: point.y, label: label });
  }

  function pickAt(event, point) {
    const tagged = numberOfTag(event.target);
    /* touch always previews first; a pen that never hovered did too */
    const twoSteps = event.pointerType === 'touch' || (event.pointerType === 'pen' && !shown);
    if (twoSteps) { tap(event, point, tagged); return; }
    const number = tagged || shown || suggestionAt(point);
    if (number) post('/pick/suggestion/' + number, {});
    else post('/pick/click', { x: point.x, y: point.y });
  }

  /* A tap previews; a second tap on the previewed object picks it. The server's list at the tap
     says whether the tap lies on it. A tap where nothing is outlined asks the server to pick whatever is there. */
  function tap(event, point, tagged) {
    const smallest = tagged || suggestionAt(point);
    if (shown && smallest === shown) { post('/pick/suggestion/' + shown, {}); return; }
    if (!shown) {
      if (smallest) { look(event); show(smallest); } else post('/pick/click', { x: point.x, y: point.y });
      return;
    }
    fetch(base + '/pick/under?x=' + point.x + '&y=' + point.y)
      .then((reply) => {
        if (!reply.ok) throw new Error('pick/under answered ' + reply.status);
        return reply.json();
      })
      .then((list) => {
        if (list.indexOf(shown) >= 0) { post('/pick/suggestion/' + shown, {}); return; }
        look(event);
        show(smallest);
      })
      .catch(console.error);
  }

  /* ---------- the found objects list: the keyboard's way in when the picture is too busy for number tags ----------
     One tab stop; arrow keys, Home and End move between the objects and outline each on the picture
     (the same preview a hover shows); Enter picks it like a click on a number tag would. */
  const FOUND_KEY = 'pm-found-index';
  const foundFold = document.getElementById('found-fold');
  let foundPlace = null;
  try {
    foundPlace = sessionStorage.getItem(FOUND_KEY);
    sessionStorage.removeItem(FOUND_KEY); // read on every load, so a stale place never carries to a later page
  } catch (e) { /* storage blocked */ }
  if (foundFold) {
    const items = Array.from(foundFold.querySelectorAll('.found-item'));
    function focusItem(index) {
      const item = items[Math.min(items.length - 1, Math.max(0, index))];
      items.forEach((other) => { other.tabIndex = other === item ? 0 : -1; });
      item.focus();
    }
    foundFold.addEventListener('focusin', (event) => {
      const item = event.target.closest('.found-item');
      if (item) show(Number(item.dataset.n));
    });
    foundFold.addEventListener('focusout', (event) => {
      if (event.target.closest('.found-item')) show(0);
    });
    foundFold.addEventListener('keydown', (event) => {
      const at = items.indexOf(event.target);
      if (at < 0) return;
      const to = { ArrowRight: at + 1, ArrowDown: at + 1, ArrowLeft: at - 1, ArrowUp: at - 1, Home: 0, End: items.length - 1 }[event.key];
      if (to === undefined) return;
      event.preventDefault();
      focusItem(to);
    });
    foundFold.addEventListener('click', (event) => {
      const item = event.target.closest('.found-item');
      if (!item) return;
      /* the page reloads after a pick: a keyboard pick (a click with detail 0) remembers its place so focus carries on from it */
      if (event.detail === 0) {
        try { sessionStorage.setItem(FOUND_KEY, String(items.indexOf(item))); } catch (e) { /* storage blocked: focus just starts over */ }
      }
      post('/pick/suggestion/' + item.dataset.n, {});
    });
    if (foundPlace !== null) { foundFold.open = true; focusItem(Number(foundPlace)); }
  }

  /* ---------- "Worked this time": a prop that had a problem note before a redraw and has a piece without one after.
     Nothing on disk remembers the earlier failure (the core clears it on success), so the page notes which rows
     carried a problem when a redraw is sent and checks them when it lands back with redrawn=1. ---------- */
  const FAILED_KEY = 'pm-pick-failed';
  document.querySelectorAll('form[action$="/redraw"]').forEach((form) => {
    form.addEventListener('submit', () => {
      const failed = Array.from(document.querySelectorAll('.row[data-failed]')).map((row) => row.dataset.number);
      try { sessionStorage.setItem(FAILED_KEY, JSON.stringify(failed)); } catch (e) { /* storage blocked: nothing is said */ }
    });
  });
  try {
    const before = JSON.parse(sessionStorage.getItem(FAILED_KEY) || '[]');
    sessionStorage.removeItem(FAILED_KEY);
    if (new URLSearchParams(location.search).get('redrawn') === '1') {
      before.forEach((number) => {
        const row = document.querySelector('.row[data-number="' + number + '"]');
        if (!row || !row.querySelector('.ready') || row.querySelector('.strip.bad')) return;
        const mark = document.createElement('span');
        mark.className = 'ready worked';
        mark.textContent = 'Worked this time';
        row.querySelector('.meta').append(' ', mark);
      });
    }
  } catch (e) { /* storage blocked or unreadable: nothing is said */ }

  /* the pick just made or selected can sit below the fold of the panel's own scroll */
  const current = document.querySelector('.row.selected');
  if (current) current.scrollIntoView({ block: 'nearest' });

  /* ---------- the picks list: always open beside the canvas, a fold under it on a tablet ---------- */
  const fold = document.getElementById('picks-fold');
  const narrow = matchMedia('(max-width: 900px)');
  function syncFold() {
    fold.classList.toggle('always-open', !narrow.matches);
    if (!narrow.matches) fold.open = true;
    else if (!fold.dataset.touched) fold.open = false;
  }
  fold.addEventListener('toggle', () => { if (narrow.matches) fold.dataset.touched = '1'; });
  narrow.addEventListener('change', syncFold);
  syncFold();
})();
