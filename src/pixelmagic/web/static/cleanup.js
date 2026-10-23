/* Clean up: the palette choice shows the pane of the option picked and tells the form which one. */
(function () {
  /* the cleaned picture is native-size pixel art: show it at a whole-number zoom (the Original is a raw drawing) */
  const cleaned = document.querySelector('.stage.pair figure.clean img.art');
  if (cleaned) window.PixelMagic.wholeZoom(cleaned, cleaned);
  const group = document.getElementById('palette-choice');
  if (!group) return;
  const field = document.getElementById('palette');
  const panes = Array.from(document.querySelectorAll('[data-pane]'));
  const select = window.PixelMagic.radiogroup(group, choose);
  function choose(value) {
    select(value);
    field.value = value;
    panes.forEach((pane) => { pane.hidden = pane.dataset.pane !== value; });
  }
})();
