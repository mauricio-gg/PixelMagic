/* PixelMagic "New art direction": the button names the image service only while the name box is empty,
   because an empty name is the one case where the service is asked for something. */
(function () {
  'use strict';
  const name = document.getElementById('direction-name');
  const button = document.getElementById('create');
  const sync = () => { button.textContent = name.value.trim() ? button.dataset.named : button.dataset.unnamed; };
  name.addEventListener('input', sync);
  sync();
})();
