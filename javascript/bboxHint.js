const USE_ORINAL_CANVAS = false;
const BBOX_MAX_NUM = 8;
const DEFAULT_X = 0.4;
const DEFAULT_Y = 0.4;
const DEFAULT_H = 0.2;
const DEFAULT_W = 0.2;

const TIMER_FADE = -1;
const COLOR_MAP  = [
  ['#ff0000', 'rgba(255, 0, 0, 0.3)'],      // red
  ['#ff9900', 'rgba(255, 153, 0, 0.3)'],    // orange
  ['#ffff00', 'rgba(255, 255, 0, 0.3)'],    // yellow
  ['#33cc33', 'rgba(51, 204, 51, 0.3)'],    // green
  ['#33cccc', 'rgba(51, 204, 204, 0.3)'],   // indogo
  ['#0066ff', 'rgba(0, 102, 255, 0.3)'],    // blue
  ['#6600ff', 'rgba(102, 0, 255, 0.3)'],    // purple
  ['#cc00cc', 'rgba(204, 0, 204, 0.3)'],    // dark pink
];

const timers = new Array(BBOX_MAX_NUM).fill(null);
const bboxes = new Array(BBOX_MAX_NUM).fill(null);
var bbox_ip = 0;

function enable_bbox_control_change(status) {
  if (status) {
    bbox_ip = 0;
  } else {
    for (let i = 0; i < bboxes.length; i++) {
      const obj = bboxes[i];
      if (!obj) continue;
      const [div, _] = obj;
      div.style.display = 'none';
    }
  }
  return args_to_array(arguments);
}

function btn_bbox_del_click() {
  if (bbox_ip > 0) {
    const [div, bbox] = bboxes[bbox_ip];
    div.style.display = 'none';
    bbox_ip--;
  }
  return args_to_array(arguments);
}

function onBoxChange(e, what, idx) {
  if (!gradioApp().querySelector("#MD-bbox-control")) { return; }

  // find canvas
  let canvas = null;
  if (USE_ORINAL_CANVAS) {
    let tabIdx = get_tab_index('mode_img2img');
    switch (tabIdx) {
      case 0: canvas = gradioApp().querySelector('div[data-testid=image] img');                 break; // img2img
      case 1: canvas = gradioApp().querySelector('#img2img_sketch div[data-testid=image] img'); break; // Sketch
      case 2: canvas = gradioApp().querySelector('#img2maskimg div[data-testid=image] img');    break; // Inpaint
      case 3: canvas = gradioApp().querySelector('#inpaint_sketch div[data-testid=image] img'); break; // Inpaint sketch
    }
  } else {
    canvas = gradioApp().querySelector('#MD-bbox-ref img');
  }

  if (!canvas) { return; }

  // parse trigger
  const v = e.target.value;

  // init bbox
  if (!bboxes[idx]) {
    const x = DEFAULT_X;  // left
    const y = DEFAULT_Y;  // top
    const w = DEFAULT_W;  // width
    const h = DEFAULT_H;  // height
    const bbox = [x, y, w, h];

    const colorMap = COLOR_MAP[idx % COLOR_MAP.length];
    const div = document.createElement('div');
    div.id               = 'MD-bbox-' + idx;
    div.style.left       = '0px';
    div.style.top        = '0px';
    div.style.width      = '0px';
    div.style.height     = '0px';
    div.style.position   = 'absolute';
    div.style.border     = '2px solid ' + colorMap[0];
    div.style.background = colorMap[1];
    div.style.zIndex     = '900';
    div.style.display    = 'none';
    div.addEventListener('mousedown', function(e) {
      if (e.button == 2) {  // right click
        div.style.display = 'none';
      }
    });
    gradioApp().getRootNode().appendChild(div);

    bboxes[idx] = [div, bbox];
  }

  // load
  let [div, bbox] = bboxes[idx];

  // update bbox
  let [x, y, w, h] = bbox;
  switch (what) {
    case 'x': x = v; break;
    case 'y': y = v; break;
    case 'w': w = v; break;
    case 'h': h = v; break;
  }
  bbox = [x, y, w, h];

  // client: canvas widget display size
  // natural: content image real size
  let vpOffset      = canvas.getBoundingClientRect();
  let vpScale       = Math.min(canvas.clientWidth / canvas.naturalWidth, canvas.clientHeight / canvas.naturalHeight);
  let canvasCenterX = (vpOffset.left + window.scrollX) + canvas.clientWidth  / 2;
  let canvasCenterY = (vpOffset.top  + window.scrollY) + canvas.clientHeight / 2;
  let scaledX       = canvas.naturalWidth  * vpScale;
  let scaledY       = canvas.naturalHeight * vpScale;
  let viewRectLeft  = canvasCenterX - scaledX / 2;
  let viewRectRight = canvasCenterX + scaledX / 2;
  let viewRectTop   = canvasCenterY - scaledY / 2;
  let viewRectDown  = canvasCenterY + scaledY / 2;

  let xDiv = viewRectLeft + scaledX * x;
  let yDiv = viewRectTop  + scaledY * y;
  let wDiv = Math.min(scaledX * w, viewRectRight - xDiv);
  let hDiv = Math.min(scaledY * h, viewRectDown  - yDiv);

  // update <div>
  div.style.left   = xDiv + 'px';
  div.style.top    = yDiv + 'px';
  div.style.width  = wDiv + 'px';
  div.style.height = hDiv + 'px';

  // save
  bboxes[idx] = [div, bbox];

  // draw <div>
  if (TIMER_FADE > 0) {
    if (timers[idx]) clearTimeout(timers[idx]);
    timers[idx] = setTimeout(function() {
      div.style.display = 'none';
    }, TIMER_FADE);
  }
  div.style.display = 'block';
}

onUiUpdate(function() {
  if (!gradioApp().querySelector("#MD-bbox-control")) { return; }

  gradioApp().querySelectorAll('input[type=range]').forEach(e => {
    if (e.parentElement.id.startsWith('MD') && !e.classList.contains('onBoxChange')) {
      const [_, what, idx] = e.parentElement.id.split('-');      // 'MD-x-2'
      e.addEventListener('input', function(e) { onBoxChange(e, what, idx); });
      e.classList.add('onBoxChange');
    }
  })
});
