/* PixelMagic Services page: the only two scripts it needs, both optional.
   1. Picking a radio saves the choice at once (without this script the "Use the selected service" button does it).
   2. "Keep it" closes the inline "Remove for good" question. */
(function () {
  'use strict';
  const chooser = document.getElementById('choose');
  document.querySelectorAll('input[name=service]').forEach((radio) => {
    radio.addEventListener('change', () => chooser.requestSubmit());
  });
  document.querySelectorAll('[data-choose-submit]').forEach((button) => { button.hidden = true; });
  document.querySelectorAll('details.confirm [data-close]').forEach((button) => {
    button.addEventListener('click', () => {
      const details = button.closest('details');
      details.open = false;
      details.querySelector('summary').focus();
    });
  });
})();
