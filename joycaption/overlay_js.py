from __future__ import annotations


OVERLAY_EDIT_JS = r"""
if (!element.dataset.jcQwenOverlayBound) {
  element.dataset.jcQwenOverlayBound = "1";

  const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

  const rowsForFrame = (frame) => {
    try {
      const rows = JSON.parse(frame.dataset.rows || "[]");
      return Array.isArray(rows) ? rows : [];
    } catch {
      return [];
    }
  };

  const frameRect = (frame) => {
    const img = frame.querySelector(".jc-overlay-image");
    const rect = (img && img.naturalWidth > 0) ? img.getBoundingClientRect() : frame.getBoundingClientRect();
    return { left: rect.left, top: rect.top, width: rect.width, height: rect.height };
  };

  const selectBox = (box) => {
    const frame = box.closest(".jc-overlay-frame");
    if (!frame) return;
    frame.querySelectorAll(".jc-box.is-selected").forEach((item) => item.classList.remove("is-selected"));
    box.classList.add("is-selected");
  };

  const applyBoxRect = (box, rect, surface) => {
    box.style.left = `${(rect.left / surface.width) * 100}%`;
    box.style.top = `${(rect.top / surface.height) * 100}%`;
    box.style.width = `${(rect.width / surface.width) * 100}%`;
    box.style.height = `${(rect.height / surface.height) * 100}%`;
  };

  const boxRect = (box, surface) => {
    const rect = box.getBoundingClientRect();
    return {
      left: rect.left - surface.left,
      top: rect.top - surface.top,
      width: rect.width,
      height: rect.height,
    };
  };

  const commitFrame = (frame, activeIndex) => {
    const surface = frameRect(frame);
    if (!surface.width || !surface.height) return;
    const rows = rowsForFrame(frame);
    const activeBox = frame.querySelector(`.jc-box[data-row-index="${activeIndex}"]`);
    if (!activeBox || !rows[activeIndex]) return;
    const rect = boxRect(activeBox, surface);
    const xMin = clamp(Math.round((rect.left / surface.width) * 1000), 0, 999);
    const yMin = clamp(Math.round((rect.top / surface.height) * 1000), 0, 999);
    const xMax = clamp(Math.round(((rect.left + rect.width) / surface.width) * 1000), xMin + 1, 1000);
    const yMax = clamp(Math.round(((rect.top + rect.height) / surface.height) * 1000), yMin + 1, 1000);
    const bboxOrder = (frame.dataset.bboxOrder || "yxyx").toLowerCase();
    if (bboxOrder === "xyxy") {
      rows[activeIndex][1] = xMin;
      rows[activeIndex][2] = yMin;
      rows[activeIndex][3] = xMax;
      rows[activeIndex][4] = yMax;
    } else {
      rows[activeIndex][1] = yMin;
      rows[activeIndex][2] = xMin;
      rows[activeIndex][3] = yMax;
      rows[activeIndex][4] = xMax;
    }
    trigger("click", { action: "box-edit", rows, index: activeIndex });
  };

  const bindFrame = (frame) => {
    if (frame.dataset.jcQwenFrameBound || !frame.closest(".jc-overlay-interactive")) return;
    frame.dataset.jcQwenFrameBound = "1";

    frame.addEventListener("pointerdown", (event) => {
      const box = event.target.closest(".jc-box");
      if (!box || !frame.contains(box)) return;
      event.preventDefault();
      event.stopPropagation();
      selectBox(box);

      const surface = frameRect(frame);
      if (!surface.width || !surface.height) return;
      const start = boxRect(box, surface);
      const handle = event.target.dataset.handle || "move";
      const startPointer = { x: event.clientX, y: event.clientY };
      const minSize = Math.max(12, Math.min(surface.width, surface.height) * 0.015);
      const activeIndex = Number(box.dataset.rowIndex);
      let changed = false;

      box.setPointerCapture?.(event.pointerId);
      box.classList.add("is-editing");

      const move = (moveEvent) => {
        const dx = moveEvent.clientX - startPointer.x;
        const dy = moveEvent.clientY - startPointer.y;
        if (Math.abs(dx) > 0.5 || Math.abs(dy) > 0.5) changed = true;

        let left = start.left;
        let top = start.top;
        let width = start.width;
        let height = start.height;

        if (handle === "move") {
          left = clamp(start.left + dx, 0, surface.width - width);
          top = clamp(start.top + dy, 0, surface.height - height);
        } else {
          if (handle.includes("e")) width = clamp(start.width + dx, minSize, surface.width - start.left);
          if (handle.includes("s")) height = clamp(start.height + dy, minSize, surface.height - start.top);
          if (handle.includes("w")) {
            const nextLeft = clamp(start.left + dx, 0, start.left + start.width - minSize);
            width = start.left + start.width - nextLeft;
            left = nextLeft;
          }
          if (handle.includes("n")) {
            const nextTop = clamp(start.top + dy, 0, start.top + start.height - minSize);
            height = start.top + start.height - nextTop;
            top = nextTop;
          }
        }

        applyBoxRect(box, { left, top, width, height }, surface);
      };

      const up = () => {
        box.classList.remove("is-editing");
        box.releasePointerCapture?.(event.pointerId);
        window.removeEventListener("pointermove", move);
        window.removeEventListener("mousemove", move);
        window.removeEventListener("pointerup", up);
        window.removeEventListener("mouseup", up);
        if (changed) {
          commitFrame(frame, activeIndex);
        }
      };

      window.addEventListener("pointermove", move);
      window.addEventListener("mousemove", move);
      window.addEventListener("pointerup", up, { once: true });
      window.addEventListener("mouseup", up, { once: true });
    });
  };

  const install = () => {
    element.querySelectorAll(".jc-overlay-frame").forEach(bindFrame);
  };

  watch("value", () => queueMicrotask(install));
  queueMicrotask(install);
}
"""
