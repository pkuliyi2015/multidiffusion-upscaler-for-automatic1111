const BBOX_MAX_NUM = 16;
const DEFAULT_X = 0.4;
const DEFAULT_Y = 0.4;
const DEFAULT_H = 0.2;
const DEFAULT_W = 0.2;

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

const t2i_bboxes = new Array(BBOX_MAX_NUM).fill(null);
const i2i_bboxes = new Array(BBOX_MAX_NUM).fill(null);



function onBoxEnableClick(is_t2i, idx, enable) {
    let canvas = null;
    let bboxes = null;
    if (is_t2i) {
        ref_div = gradioApp().querySelector('#MD-bbox-ref-t2i');
        canvas = gradioApp().querySelector('#MD-bbox-ref-t2i img');
        bboxes = t2i_bboxes;
    } else {
        ref_div = gradioApp().querySelector('#MD-bbox-ref-i2i');
        canvas = gradioApp().querySelector('#MD-bbox-ref-i2i img');
        bboxes = i2i_bboxes;
    }
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
            div.id = 'MD-bbox-' + (is_t2i ? 't2i-' : 'i2i-') + idx;
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
                    onBoxMouseDown(e, is_t2i, idx);
                }
            });
            div.addEventListener('mousemove', function (e) {
                updateCursorStyle(e, is_t2i, idx);
            });
            ref_div.appendChild(div);
            bboxes[idx] = [div, bbox];
        }
        // Show the bounding box
        let [div, bbox] = bboxes[idx];
        let [x, y, w, h] = bbox;
        displayBox(canvas, div, x, y, w, h);
        return true;
    } else {
        if (!bboxes[idx]) { return false; }
        const [div, _] = bboxes[idx];
        div.style.display = 'none';
    }
    return false;
}

function displayBox(canvas, div, x, y, w, h) {
    // check null input
    if (!canvas || !div || x == null || y == null || w == null || h == null) { return; }
    // client: canvas widget display size
    // natural: content image real size
    let vpScale = Math.min(canvas.clientWidth / canvas.naturalWidth, canvas.clientHeight / canvas.naturalHeight);
    let canvasCenterX = canvas.clientWidth / 2;
    let canvasCenterY = canvas.clientHeight / 2;
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
    div.style.display = 'block';
}

function onBoxChange(is_t2i, idx, what, v) {
    let bboxes = null;
    let canvas = null;
    if (is_t2i) {
        bboxes = t2i_bboxes;
        canvas = gradioApp().querySelector('#MD-bbox-ref-t2i img');
    } else {
        bboxes = i2i_bboxes;
        canvas = gradioApp().querySelector('#MD-bbox-ref-i2i img');
    }
    if (!bboxes[idx] || !canvas) {
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
    // parse trigger
    switch (what) {
        case 'x': x = v; break;
        case 'y': y = v; break;
        case 'w': w = v; break;
        case 'h': h = v; break;
    }
    bbox = [x, y, w, h];
    displayBox(canvas, div, x, y, w, h);
    bboxes[idx] = [div, bbox];
    return v
}

function onBoxMouseDown(e, is_t2i, idx) {
    let bboxes = null;
    let canvas = null;
    if (is_t2i) {
        bboxes = t2i_bboxes;
        canvas = gradioApp().querySelector('#MD-bbox-ref-t2i img');
    } else {
        bboxes = i2i_bboxes;
        canvas = gradioApp().querySelector('#MD-bbox-ref-i2i img');
    }
    // Get the bounding box
    if (!canvas || !bboxes[idx]) return;
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

    // Canvas can be regarded as invariant during the drag operation
    // Calculate in advance to reduce overhead

    // Calculate viewport scale based on the current canvas size and the natural image size
    let vpScale = Math.min(canvas.clientWidth / canvas.naturalWidth, canvas.clientHeight / canvas.naturalHeight);
    let vpOffset = canvas.getBoundingClientRect();

    // Calculate scaled dimensions of the canvas
    let scaledX = canvas.naturalWidth * vpScale;
    let scaledY = canvas.naturalHeight * vpScale;

    // Calculate the canvas center and view rectangle coordinates
    let canvasCenterX = (vpOffset.left + window.scrollX) + canvas.clientWidth / 2;
    let canvasCenterY = (vpOffset.top + window.scrollY) + canvas.clientHeight / 2;
    let viewRectLeft = canvasCenterX - scaledX / 2 - window.scrollX;
    let viewRectRight = canvasCenterX + scaledX / 2 - window.scrollX;
    let viewRectTop = canvasCenterY - scaledY / 2 - window.scrollY;
    let viewRectDown = canvasCenterY + scaledY / 2 - window.scrollY;

    mouseX = Math.min(Math.max(mouseX, viewRectLeft), viewRectRight);
    mouseY = Math.min(Math.max(mouseY, viewRectTop), viewRectDown);

    // Move or resize the bounding box on mousemove
    function onMouseMove(e) {

        // Prevent selecting anything irrelevant
        e.preventDefault();

        // Get the new mouse position
        let newMouseX = e.clientX;
        let newMouseY = e.clientY;

        // clamp the mouse position to the view rectangle
        newMouseX = Math.min(Math.max(newMouseX, viewRectLeft), viewRectRight);
        newMouseY = Math.min(Math.max(newMouseY, viewRectTop), viewRectDown);

        // Calculate the mouse movement delta
        let dx = (newMouseX - mouseX) / scaledX;
        let dy = (newMouseY - mouseY) / scaledY;

        // Update the mouse position
        mouseX = newMouseX;
        mouseY = newMouseY;

        // if no move just return
        if (dx == 0 && dy == 0) return;

        // Update the mouse position
        let [x, y, w, h] = bbox;
        if (moveHorizontal && moveVertical) {
            // If moving the bounding box
            x = Math.min(Math.max(x + dx, 0), 1 - w);
            y = Math.min(Math.max(y + dy, 0), 1 - h);
        } else {
            // If resizing the bounding box
            if (resizeLeft || resizeRight) {
                if (x < horizontalPivot) {
                    if (dx <= w) {
                        // If still within the left side of the pivot
                        x = x + dx;
                        w = w - dx;
                    } else {
                        // If crossing the pivot
                        w = dx - w;
                        x = horizontalPivot;
                    }
                } else {
                    if (w + dx < 0) {
                        // If still within the right side of the pivot
                        x = horizontalPivot + w + dx;
                        w = - dx - w;
                    } else {
                        // If crossing the pivot
                        x = horizontalPivot;
                        w = w + dx;
                    }
                }
                // Clamp the bounding box to the image
                if (x < 0) {
                    w = w + x;
                    x = 0;
                } else if (x + w > 1) {
                    w = 1 - x;
                }
            }
            // Same as above, but for the vertical axis
            if (resizeTop || resizeBottom) {
                if (y < verticalPivot) {
                    if (dy <= h) {
                        y = y + dy;
                        h = h - dy;
                    } else {
                        h = dy - h;
                        y = verticalPivot;
                    }
                } else {
                    if (h + dy < 0) {
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
        old_bbox = bboxes[idx][1];
        if (old_bbox[0] == x && old_bbox[1] == y && old_bbox[2] == w && old_bbox[3] == h) {
            return;
        }
        // Update the bounding box value
        bbox = [x, y, w, h];
        bboxes[idx] = [div, bbox];
        // Click the invisible update button to update the value in Python script.
        // This will also trigger the bbox rendering.
        gradioApp().querySelector('#MD-update-' + (is_t2i ? 't2i-' : 'i2i-') + idx).click();
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

function updateCursorStyle(e, is_t2i, idx) {
    // This function changes the cursor style when hovering over the bounding box
    let bboxes = is_t2i ? t2i_bboxes : i2i_bboxes;
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

function updateAllBoxes(is_t2i) {
    // This function redraw all bounding boxes
    let bboxes = null;
    let canvas = null;
    if (is_t2i) {
        bboxes = t2i_bboxes;
        canvas = gradioApp().querySelector('#MD-bbox-ref-t2i img');
    } else {
        bboxes = i2i_bboxes;
        canvas = gradioApp().querySelector('#MD-bbox-ref-i2i img');
    }
    if (!canvas) return;
    for (let idx = 0; idx < bboxes.length; idx++) {
        if (!bboxes[idx]) continue;
        let [div, bbox] = bboxes[idx];
        if (div.style.display == 'none') continue;
        let [x, y, w, h] = bbox;
        displayBox(canvas, div, x, y, w, h);
    }
}

window.addEventListener('resize', updateAllBoxes);

function updateCallback(is_t2i, idx) {
    let bboxes = is_t2i ? t2i_bboxes : i2i_bboxes;
    if (!bboxes[idx]) return [DEFAULT_X, DEFAULT_Y, DEFAULT_W, DEFAULT_H];
    let [_, bbox] = bboxes[idx];
    return bbox;
}

// function observeCanvasVisibility() {
//     // This function observes the visibility of the canvas
//     let canvas = gradioApp().querySelector('#MD-bbox-ref img');
//     if (!canvas) {
//         return;
//     }
//     let observer = new IntersectionObserver((entries) => {
//         if (entries[0].isIntersecting) {
//             updateAllBoxes();
//         }
//     });
//     observer.observe(canvas);
// }