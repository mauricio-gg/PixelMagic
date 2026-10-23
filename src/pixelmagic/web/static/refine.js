/* PixelMagic refine page: three small jobs, none of which decides anything.
   1. Before and After: a tab swaps which picture of the scene is on show.
   2. A click on a prop's refined picture fills in its anchor's x and y, scaled to the prop's target size
      (the art pixels the manifest counts in) and clamped to it. "Set the anchor" still posts the form.
   3. The scene is shown at a whole-number zoom of the refined picture, so every art pixel is the same width. */
(function () {
  'use strict';
  const refined = document.getElementById('refined');
  const views = document.getElementById('views');
  /* the refined picture is the pixel art; Before (a raw drawing) fills the same box */
  if (refined) PixelMagic.wholeZoom(document.getElementById('picture'), refined, document.getElementById('stage'));
  if (views && refined) {
    const select = PixelMagic.radiogroup(views, (value) => {
      refined.hidden = value !== 'after';
      select(value);
    });
    select('after');
  }

  document.querySelectorAll('img[data-anchor-for]').forEach((picture) => {
    picture.addEventListener('click', (event) => {
      const form = document.getElementById(picture.dataset.anchorFor);
      /* getBoundingClientRect() includes the picture's border, so the click is measured against the content box
         (clientLeft/clientTop/clientWidth/clientHeight) or the far edge of the target is never reachable. */
      const bounds = picture.getBoundingClientRect();
      const targetW = Number(picture.dataset.targetW);
      const targetH = Number(picture.dataset.targetH);
      const x = (event.clientX - bounds.left - picture.clientLeft) * targetW / picture.clientWidth;
      const y = (event.clientY - bounds.top - picture.clientTop) * targetH / picture.clientHeight;
      form.elements.x.value = Math.min(targetW, Math.max(0, Math.round(x)));
      form.elements.y.value = Math.min(targetH, Math.max(0, Math.round(y)));
    });
  });
})();
