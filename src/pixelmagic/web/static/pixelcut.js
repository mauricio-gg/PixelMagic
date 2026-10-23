/* Cut props, size step: the choice shows the pane of the option picked. The number box is only sent (and only required) when "I will type it" is chosen. */
(function () {
  const group = document.getElementById('size-mode');
  if (!group) return;
  const field = document.getElementById('size');
  const panes = Array.from(document.querySelectorAll('[data-pane]'));
  const select = window.PixelMagic.radiogroup(group, choose);
  function choose(value) {
    select(value);
    panes.forEach((pane) => { pane.hidden = pane.dataset.pane !== value; });
    field.disabled = value !== 'type';
    field.required = value === 'type';
    if (value === 'type') field.focus();
  }
  document.getElementById('native').addEventListener('click', () => { field.value = 1; });
})();
