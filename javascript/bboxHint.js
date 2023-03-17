const BBOX_MAX_NUM = 16;
const DEFAULT_X = 0.4;
const DEFAULT_Y = 0.4;
const DEFAULT_H = 0.2;
const DEFAULT_W = 0.2;

const TIMER_FADE = -1;
const COLOR_MAP = [
    ['#ff0000', 'rgba(255, 0, 0, 0.3)'],          // red
    ['#ff9900', 'rgba(255, 153, 0, 0.3)'],        // orange
    ['#ffff00', 'rgba(255, 255, 0, 0.3)'],        // yellow
    ['#33cc33', 'rgba(51, 204, 51, 0.3)'],        // green
    ['#33cccc', 'rgba(51, 204, 204, 0.3)'],       // indigo
    ['#0066ff', 'rgba(0, 102, 255, 0.3)'],        // blue
    ['#6600ff', 'rgba(102, 0, 255, 0.3)'],        // purple
    ['#cc00cc', 'rgba(204, 0, 204, 0.3)'],        // dark pink
    ['#ff6666', 'rgba(255, 102, 102, 0.3)'],      // light red
    ['#ffcc66', 'rgba(255, 204, 102, 0.3)'],      // light orange
    ['#99cc00', 'rgba(153, 204, 0, 0.3)'],        // lime green
    ['#00cc99', 'rgba(0, 204, 153, 0.3)'],        // teal
    ['#0099cc', 'rgba(0, 153, 204, 0.3)'],        // steel blue
    ['#9933cc', 'rgba(153, 51, 204, 0.3)'],       // lavender
    ['#ff3399', 'rgba(255, 51, 153, 0.3)'],       // hot pink
    ['#996633', 'rgba(153, 102, 51, 0.3)'],       // brown
];

const RESIZE_BORDER = 5;
const MOVE_BORDER = 5;

const timers = new Array(BBOX_MAX_NUM).fill(null);
const bboxes = new Array(BBOX_MAX_NUM).fill(null);


function onBoxEnableClick(idx, enable) {
    let canvas = gradioApp().querySelector('#MD-bbox-ref img');
    if (!canvas) { return false; }
    if (enable) {
        // Check if the bounding box already exists
        if (!bboxes[idx]) {
            // Initialize bounding box
            const x = DEFAULT_X;
            const y = DEFAULT_Y;
            const w = DEFAULT_W;
            const h = DEFAULT_H;
            const bbox = [x, y, w, h];
            const colorMap = COLOR_MAP[idx % COLOR_MAP.length];
            const div = document.createElement('div');
            div.id = 'MD-bbox-' + idx;
            div.style.left = '0px';
            div.style.top = '0px';
            div.style.width = '0px';
            div.style.height = '0px';
            div.style.position = 'absolute';
            div.style.border = '2px solid ' + colorMap[0];
            div.style.background = colorMap[1];
            div.style.zIndex = '900';
            div.style.display = 'none';
            div.addEventListener('mousedown', function (e) {
                if (e.button === 0) {
                    onBoxMouseDown(e, idx);
                }
            });
            div.addEventListener('mousemove', function (e) {
                updateCursorStyle(e, idx);
            });
            document.body.appendChild(div);
            bboxes[idx] = [div, bbox];
        }
        // Show the bounding box
        let [div, bbox] = bboxes[idx];
        let [x, y, w, h] = bbox;
        updateBox(canvas, div, idx, x, y, w, h);
        return true;
    } else {
        if (!bboxes[idx]) { return false; }
        const [div, _] = bboxes[idx];
        div.style.display = 'none';
    }
    return false;
}

function updateBox(canvas, div, idx, x, y, w, h) {
    // update bbox
    bbox = [x, y, w, h];

    // client: canvas widget display size
    // natural: content image real size
    let vpOffset = canvas.getBoundingClientRect();
    let vpScale = Math.min(canvas.clientWidth / canvas.naturalWidth, canvas.clientHeight / canvas.naturalHeight);
    let canvasCenterX = (vpOffset.left + window.scrollX) + canvas.clientWidth / 2;
    let canvasCenterY = (vpOffset.top + window.scrollY) + canvas.clientHeight / 2;
    let scaledX = canvas.naturalWidth * vpScale;
    let scaledY = canvas.naturalHeight * vpScale;
    let viewRectLeft = canvasCenterX - scaledX / 2;
    let viewRectRight = canvasCenterX + scaledX / 2;
    let viewRectTop = canvasCenterY - scaledY / 2;
    let viewRectDown = canvasCenterY + scaledY / 2;

    let xDiv = viewRectLeft + scaledX * x;
    let yDiv = viewRectTop + scaledY * y;
    let wDiv = Math.min(scaledX * w, viewRectRight - xDiv);
    let hDiv = Math.min(scaledY * h, viewRectDown - yDiv);

    // update <div>
    div.style.left = xDiv + 'px';
    div.style.top = yDiv + 'px';
    div.style.width = wDiv + 'px';
    div.style.height = hDiv + 'px';

    // save
    bboxes[idx] = [div, bbox];

    // draw <div>
    if (TIMER_FADE > 0) {
        if (timers[idx]) clearTimeout(timers[idx]);
        timers[idx] = setTimeout(function () {
            div.style.display = 'none';
        }, TIMER_FADE);
    }
    div.style.display = 'block';
}

function onBoxChange(idx, what, v) {
    if (!bboxes[idx]) {
        switch (what) {
            case 'x': return DEFAULT_X;
            case 'y': return DEFAULT_Y;
            case 'w': return DEFAULT_W;
            case 'h': return DEFAULT_H;
        }
    }
    let [div, bbox] = bboxes[idx];
    if (div.style.display == 'none') { return v; }
    let [x, y, w, h] = bbox;
    let canvas = gradioApp().querySelector('#MD-bbox-ref img');
    if (!canvas) { return; }
    // parse trigger
    switch (what) {
        case 'x': x = v; break;
        case 'y': y = v; break;
        case 'w': w = v; break;
        case 'h': h = v; break;
    }
    bbox = [x, y, w, h];
    updateBox(canvas, div, idx, x, y, w, h);
    return v
}

function updateCallback(idx) {
    if (!bboxes[idx]) { return [DEFAULT_X, DEFAULT_Y, DEFAULT_W, DEFAULT_H]; }
    let [div, bbox] = bboxes[idx];
    return bbox;
}

function onBoxMouseDown(e, idx) {
    if (!bboxes[idx]) return;
    let [div, bbox] = bboxes[idx];

    // Check if the click is inside the bounding box
    let boxRect = div.getBoundingClientRect();
    let mouseX = e.clientX;
    let mouseY = e.clientY;

    let resizeLeft = mouseX >= boxRect.left && mouseX <= boxRect.left + RESIZE_BORDER;
    let resizeRight = mouseX >= boxRect.right - RESIZE_BORDER && mouseX <= boxRect.right;
    let resizeTop = mouseY >= boxRect.top && mouseY <= boxRect.top + RESIZE_BORDER;
    let resizeBottom = mouseY >= boxRect.bottom - RESIZE_BORDER && mouseY <= boxRect.bottom;

    let moveHorizontal = mouseX >= boxRect.left + MOVE_BORDER && mouseX <= boxRect.right - MOVE_BORDER;
    let moveVertical = mouseY >= boxRect.top + MOVE_BORDER && mouseY <= boxRect.bottom - MOVE_BORDER;

    if (!resizeLeft && !resizeRight && !resizeTop && !resizeBottom && !moveHorizontal && !moveVertical) {
        return;
    }

    const horizontalPivot = resizeLeft ? bbox[0] + bbox[2] : bbox[0];
    const verticalPivot = resizeTop ? bbox[1] + bbox[3] : bbox[1];

    // Move or resize the bounding box on mousemove
    function onMouseMove(e) {
        let [div, bbox] = bboxes[idx];
        let canvas = gradioApp().querySelector('#MD-bbox-ref img');
        if (!canvas) { return; }
        // prevent selection anything irrelevant
        e.preventDefault();
        let newMouseX = e.clientX;
        let newMouseY = e.clientY;

        let vpScale = Math.min(canvas.clientWidth / canvas.naturalWidth, canvas.clientHeight / canvas.naturalHeight);
        let vpOffset = canvas.getBoundingClientRect();

        let scaledX = canvas.naturalWidth * vpScale;
        let scaledY = canvas.naturalHeight * vpScale;

        let canvasCenterX = (vpOffset.left + window.scrollX) + canvas.clientWidth / 2;
        let canvasCenterY = (vpOffset.top + window.scrollY) + canvas.clientHeight / 2;
        let viewRectLeft = canvasCenterX - scaledX / 2 - window.scrollX;
        let viewRectRight = canvasCenterX + scaledX / 2 - window.scrollX;
        let viewRectTop = canvasCenterY - scaledY / 2 - window.scrollY;
        let viewRectDown = canvasCenterY + scaledY / 2 - window.scrollY;

        let [x, y, w, h] = bbox;

        let dx = (newMouseX - mouseX) / scaledX;
        let dy = (newMouseY - mouseY) / scaledY;

        if (newMouseX < viewRectLeft && dx > 0) dx = 0;
        if (newMouseX > viewRectRight && dx < 0) dx = 0;
        if (newMouseY < viewRectTop && dy > 0) dy = 0;
        if (newMouseY > viewRectDown && dy < 0) dy = 0;

        if (moveHorizontal && moveVertical) {
            x = Math.min(Math.max(x + dx, 0), 1 - w);
            y = Math.min(Math.max(y + dy, 0), 1 - h);
        } else {
            if (resizeLeft || resizeRight) {
                if (x < horizontalPivot){
                    if (dx <= w){
                        x = x + dx;
                        w = w - dx;
                    } else {
                        w = dx - w;
                        x = horizontalPivot;
                    }
                } else {
                    if(w + dx < 0){
                        x = horizontalPivot + w + dx;
                        w = - dx - w;
                    } else {
                        x = horizontalPivot;
                        w = w + dx;
                    }
                }
                if (x < 0) {
                    w = w + x;
                    x = 0;
                } else if (x + w > 1) {
                    w = 1 - x;
                }
            }
            if (resizeTop || resizeBottom) {
                if (y < verticalPivot){
                    if (dy <= h){
                        y = y + dy;
                        h = h - dy;
                    } else {
                        h = dy - h;
                        y = verticalPivot;
                    }
                } else {
                    if(h + dy < 0){
                        y = verticalPivot + h + dy;
                        h = - dy - h;
                    } else {
                        y = verticalPivot;
                        h = h + dy;
                    }
                }
                if (y < 0) {
                    h = h + y;
                    y = 0;
                } else if (y + h > 1) {
                    h = 1 - y;
                }
            }
        }

        updateBox(canvas, div, idx, x, y, w, h);
        gradioApp().querySelector('#md-update-' + idx).click();
        mouseX = e.clientX;
        mouseY = e.clientY;
    }

    // Remove the mousemove and mouseup event listeners
    function onMouseUp() {
        document.removeEventListener('mousemove', onMouseMove);
        document.removeEventListener('mouseup', onMouseUp);
    }

    // Add the event listeners
    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
}

function updateCursorStyle(e, idx) {
    if (!bboxes[idx]) return;
    let [div, _] = bboxes[idx];
    let boxRect = div.getBoundingClientRect();
    let mouseX = e.clientX;
    let mouseY = e.clientY;

    let resizeLeft = mouseX >= boxRect.left && mouseX <= boxRect.left + RESIZE_BORDER;
    let resizeRight = mouseX >= boxRect.right - RESIZE_BORDER && mouseX <= boxRect.right;
    let resizeTop = mouseY >= boxRect.top && mouseY <= boxRect.top + RESIZE_BORDER;
    let resizeBottom = mouseY >= boxRect.bottom - RESIZE_BORDER && mouseY <= boxRect.bottom;

    if ((resizeLeft && resizeTop) || (resizeRight && resizeBottom)) {
        div.style.cursor = 'nwse-resize';
    } else if ((resizeLeft && resizeBottom) || (resizeRight && resizeTop)) {
        div.style.cursor = 'nesw-resize';
    } else if (resizeLeft || resizeRight) {
        div.style.cursor = 'ew-resize';
    } else if (resizeTop || resizeBottom) {
        div.style.cursor = 'ns-resize';
    } else {
        div.style.cursor = 'move';
    }
}

function updateAllBoxes() {
    let canvas = gradioApp().querySelector('#MD-bbox-ref img');
    if (!canvas) {
        return;
    }
    for (let idx = 0; idx < bboxes.length; idx++) {
        if (!bboxes[idx]) continue;
        let [div, bbox] = bboxes[idx];
        let [x, y, w, h] = bbox;
        updateBox(canvas, div, idx, x, y, w, h);
    }
}

window.addEventListener('resize', updateAllBoxes);