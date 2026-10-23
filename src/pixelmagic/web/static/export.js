/* PixelMagic export page: one small job that decides nothing.
   The three layer chips hide and show the room's three pictures (tiles, structure, props). The room itself, what
   goes in the bundle and the export are the server's. */
(function () {
  'use strict';
  /* the room is small pixel art: show it at a whole-number zoom; the other two layers fill the tiles' box */
  const tiles = document.querySelector('.stage.room img.art');
  if (tiles) PixelMagic.wholeZoom(tiles, tiles, tiles.parentElement);
  document.querySelectorAll('.layers button').forEach((chip) => {
    const layer = document.querySelector('.stage img[data-layer="' + chip.dataset.layer + '"]');
    chip.addEventListener('click', () => {
      const on = chip.getAttribute('aria-pressed') !== 'true';
      chip.setAttribute('aria-pressed', on ? 'true' : 'false');
      layer.classList.toggle('off', !on);
    });
  });
})();
