/* PixelMagic web page: the little script every page loads (deferred, no build step).
   The work is done at load. The one global is `PixelMagic`: the helpers that the picking scripts and surfaces.js share.
   1. Page fade where the browser has no cross-document View Transitions.
   2. Forms that start slow work (data-working): the button area turns into the job strip.
      Also the shared helpers: startJob/stopJob, radiogroup (tool trays), holdKeys (hold-down keys for a
      tray's tools) and setHint (mouse and touch hints).
   3. The step timeline: a tap on a finished step opens a "Reopen this step" popover,
      and a step that finished since the last visit plays the "Complete a step" animation. */
(function () {
  'use strict';
  const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)');
  const pause = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  if (!('onpagereveal' in window) && !reduceMotion.matches) document.body.classList.add('entering');

  /* ---------- slow work: be honest that it is running ----------
     Drawing, keeping and finding objects are one blocking request each, so the page cannot know progress.
     On submit the form's buttons give way to the job strip: the label, one status line, a travelling bar.
     No percentage, no countdown. The strip and the buttons come back if the page returns from the back-forward cache.
     startJob and stopJob are shared with the surfaces script, which posts its one request itself. */
  const working = [];
  function startJob(buttons, label, detail) {
    const strip = document.createElement('div');
    strip.className = 'job';
    strip.setAttribute('role', 'status');
    const words = document.createElement('div');
    words.className = 'jl';
    const title = document.createElement('b');
    title.textContent = label;
    const status = document.createElement('span');
    status.textContent = (detail ? detail + ' ' : '') + 'This page moves on by itself when it is done.';
    words.append(title, status);
    const bar = document.createElement('div');
    bar.className = 'bar indeterminate';
    bar.setAttribute('aria-hidden', 'true');
    strip.append(words, bar);
    buttons[0].before(strip);
    buttons.forEach((button) => { button.disabled = true; button.hidden = true; });
    const job = { strip, buttons };
    working.push(job);
    return job;
  }
  function stopJob(job) {
    const at = working.indexOf(job);
    if (at < 0) return;
    working.splice(at, 1);
    job.strip.remove();
    job.buttons.forEach((button) => { button.disabled = false; button.hidden = false; });
  }
  document.querySelectorAll('form[data-working]').forEach((form) => {
    form.addEventListener('submit', () => {
      const buttons = Array.from(form.elements).filter((el) => el.type === 'submit');
      if (!buttons.length || working.some((job) => job.form === form)) return;
      startJob(buttons, form.dataset.working, form.dataset.status).form = form;
    });
  });
  window.addEventListener('pageshow', (event) => {
    if (!event.persisted) return;
    working.slice().forEach(stopJob);
  });

  /* ---------- tool trays: a segmented control that behaves as a radio group ----------
     A click or an arrow key selects; the returned select(value, held) sets aria-checked and the roving tabindex (not the tabindex when held). */
  function radiogroup(group, onSelect) {
    const radios = Array.from(group.querySelectorAll('[role=radio]'));
    function select(value, held) {
      radios.forEach((radio) => {
        const on = radio.dataset.value === value;
        radio.setAttribute('aria-checked', on ? 'true' : 'false');
        if (!held) radio.tabIndex = on ? 0 : -1; // a held key's tool never moves the tab stop
      });
    }
    radios.forEach((radio) => {
      radio.addEventListener('click', () => { if (!radio.disabled) onSelect(radio.dataset.value); });
      radio.addEventListener('keydown', (event) => {
        const keys = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 };
        if (!(event.key in keys)) return;
        event.preventDefault();
        const enabled = radios.filter((r) => !r.disabled);
        const next = enabled[(enabled.indexOf(radio) + keys[event.key] + enabled.length) % enabled.length];
        next.focus();
        onSelect(next.dataset.value);
      });
    });
    return select;
  }

  /* ---------- hold-down keys for a tray's tools ----------
     Holding a key switches the tray to a tool only while it is held; letting go hands back to the tool the person
     chose (the caller keeps that, and never remembers a held tool). `tools` maps a key to a tool's data-value:
     'primary' (Cmd on a Mac, Ctrl elsewhere), 'alt' (Option, Alt) or 'shift'. onChange(tool) gets the tool of the
     key pressed last that is still held, or null when none is. The same keys held on the stage never reach the
     browser as shift-, alt- or primary-clicks (links, new windows, downloads) or drags (text, images).
     The tray's buttons show their key (.keyhint) and say it in their title. */
  const mac = /^(mac|iphone|ipad)/i.test((navigator.userAgentData && navigator.userAgentData.platform) || navigator.platform || '');
  const KEY_NAMES = { primary: mac ? 'Meta' : 'Control', alt: 'Alt', shift: 'Shift' };
  const KEY_LABELS = mac ? { primary: '\u2318', alt: '\u2325', shift: '\u21E7' } : { primary: 'Ctrl', alt: 'Alt', shift: 'Shift' };
  const KEY_FLAGS = { primary: mac ? 'metaKey' : 'ctrlKey', alt: 'altKey', shift: 'shiftKey' };
  function holdKeys(stage, group, tools, onChange) {
    Object.keys(tools).forEach((key) => {
      const radio = group.querySelector('[data-value="' + tools[key] + '"]');
      radio.querySelector('.keyhint').textContent = KEY_LABELS[key];
      radio.title += ' (hold ' + KEY_LABELS[key] + ')';
    });
    let held = []; // the held keys that have a tool, oldest first
    let shown = null;
    function report() {
      const tool = held.length ? tools[held[held.length - 1]] : null;
      if (tool === shown) return;
      shown = tool;
      onChange(tool);
    }
    /* a key event says which modifiers are down now, so a release we never heard (Cmd-Tab away) is caught at the next one */
    function release(event) { held = held.filter((key) => event[KEY_FLAGS[key]]); }
    document.addEventListener('keydown', (event) => {
      if (event.repeat || event.target.closest('input, textarea, select, [contenteditable]')) return;
      release(event);
      const key = Object.keys(tools).find((name) => KEY_NAMES[name] === event.key);
      if (key && !held.includes(key)) held.push(key);
      report();
    });
    /* a release is heard even in a text field, so a key let go there never sticks */
    document.addEventListener('keyup', (event) => { release(event); report(); });
    const dropAll = () => { held = []; report(); };
    window.addEventListener('blur', dropAll);
    document.addEventListener('visibilitychange', () => { if (document.hidden) dropAll(); });

    const modified = (event) => event.shiftKey || event.altKey || event[KEY_FLAGS.primary];
    ['mousedown', 'click', 'auxclick'].forEach((type) => {
      stage.addEventListener(type, (event) => { if (modified(event)) event.preventDefault(); });
    });
    stage.addEventListener('dragstart', (event) => event.preventDefault());
  }

  /* ---------- the hint line: one text for a mouse, one for touch (CSS shows the one that fits the device) ---------- */
  function setHint(hint, mouseText, touchText) {
    const mouse = document.createElement('span');
    mouse.className = 'h-mouse';
    mouse.textContent = mouseText;
    const touch = document.createElement('span');
    touch.className = 'h-touch';
    touch.textContent = touchText;
    hint.replaceChildren(mouse, touch);
  }

  /* ---------- whole-number zoom for pixel art ----------
     CSS fits a picture to its panel at any zoom, so some art pixels come out wider than others and the edges look
     jagged. wholeZoom keeps the space the layout gives `art` today, but sizes it to `source`'s natural size times the
     largest whole number that fits. `source` is the pixel-art picture (it may be `art` itself, or a smaller one whose
     box `art` fills). `stage`, when given, holds layers stretched over `art`; on a narrow screen CSS makes it wider
     than the picture, so it is narrowed to match or the layers would be stretched to a non-whole zoom. When even 1x
     does not fit, the CSS fit is left alone. Re-run on load and when the panel resizes. */
  function wholeZoom(art, source, stage) {
    function apply() {
      /* clear our own size first, or the space could only ever shrink */
      art.style.width = '';
      art.style.height = '';
      if (stage) stage.style.width = '';
      const space = art.getBoundingClientRect();
      const zoom = Math.floor(Math.min(space.width / source.naturalWidth, space.height / source.naturalHeight));
      if (zoom < 1) return;
      art.style.width = source.naturalWidth * zoom + 'px';
      art.style.height = source.naturalHeight * zoom + 'px';
      if (stage) stage.style.width = art.style.width;
    }
    const pictures = art === source ? [art] : [art, source];
    let waiting = pictures.length;
    pictures.forEach((picture) => {
      if (picture.complete && picture.naturalWidth) { waiting -= 1; return; }
      picture.addEventListener('load', () => { waiting -= 1; if (waiting === 0) apply(); }, { once: true });
    });
    if (waiting === 0) apply();
    new ResizeObserver(() => { if (waiting === 0) apply(); }).observe(art.closest('.work'));
  }

  window.PixelMagic = { startJob, stopJob, radiogroup, holdKeys, setHint, wholeZoom };

  const timeline = document.querySelector('[data-tl]');
  if (!timeline) return;
  const items = Array.from(timeline.querySelectorAll('.tl-item'));

  /* ---------- touch: a tap on a done step opens a popover instead of leaving the page ---------- */
  const popover = document.getElementById('step-pop');
  const coarse = matchMedia('(pointer: coarse)');
  items.forEach((item) => {
    const link = item.querySelector('a.tl-hit');
    if (!popover || !link || item.dataset.state !== 'done') return;
    link.addEventListener('click', (event) => {
      /* the event's own pointer type decides; a keyboard click has none and navigates */
      const touched = 'pointerType' in event ? event.pointerType === 'touch' : coarse.matches;
      if (!touched) return;
      event.preventDefault();
      popover.querySelector('[data-pop-name]').textContent = link.querySelector('.tl-name').textContent;
      popover.querySelector('[data-pop-go]').href = link.href;
      const box = link.getBoundingClientRect();
      popover.style.left = Math.max(12, Math.min(box.left, innerWidth - 300)) + 'px';
      popover.style.top = box.bottom + 6 + 'px';
      popover.showPopover();
    });
  });

  /* ---------- completing a step ---------- */
  /* The page arrives in its final state. If a step is done now that was not done the last time this
     scene's timeline was shown in this tab, wind that step back and play: the marker squashes (90 ms),
     the check draws (180 ms), the line grows (360 ms), the next marker takes the ring (240 ms). */
  const key = timeline.dataset.scene ? 'pm-timeline:' + timeline.dataset.scene : '';
  const now = items.map((item) => item.dataset.state);
  let before = null;
  try {
    if (key) {
      before = JSON.parse(sessionStorage.getItem(key));
      sessionStorage.setItem(key, JSON.stringify(now));
    }
  } catch (e) { /* storage blocked: the page works, it just does not animate */ }

  const finished = Array.isArray(before) && before.length === now.length
    ? now.findIndex((state, i) => state === 'done' && before[i] !== 'done') : -1;
  if (finished < 0 || reduceMotion.matches || timeline.hasAttribute('data-reopen')) return;

  const final = items.map((item) => [item.dataset.state, item.dataset.fill]);
  const show = (item, state) => {
    item.dataset.state = state;
    if (state === 'done') item.dataset.fill = '1'; else delete item.dataset.fill;
  };
  items.forEach((item, i) => show(item, before[i]));

  (async () => {
    const item = items[finished];
    await pause(60);
    item.classList.add('squash');
    await pause(90);
    item.classList.remove('squash');
    item.dataset.state = 'done';
    const check = item.querySelector('.check');
    check.classList.add('draw');
    await pause(60);
    item.dataset.fill = '1';
    await pause(150);
    items.forEach((other, i) => {
      if (i === finished) return;
      other.dataset.state = final[i][0];
      if (final[i][1]) other.dataset.fill = final[i][1]; else delete other.dataset.fill;
    });
    await pause(300);
    check.classList.remove('draw');
  })();
})();
