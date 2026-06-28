CUSTOM_HEAD = ""


CUSTOM_CSS = """
.btn-save-preset button,
button.btn-save-preset { background: #2563eb !important; color: #ffffff !important; border-color: #2563eb !important; }
.btn-load-preset button,
button.btn-load-preset { background: #059669 !important; color: #ffffff !important; border-color: #059669 !important; }
.btn-reset-preset button,
button.btn-reset-preset { background: #ca8a04 !important; color: #ffffff !important; border-color: #ca8a04 !important; }
.btn-delete-preset button,
button.btn-delete-preset { background: #dc2626 !important; color: #ffffff !important; border-color: #dc2626 !important; }
.btn-refresh button,
button.btn-refresh { background: #0d9488 !important; color: #ffffff !important; border-color: #0d9488 !important; }

.btn-pre-caption button,
button.btn-pre-caption { background: #0d9488 !important; color: #ffffff !important; border-color: #0d9488 !important; }
.btn-pre-batch button,
button.btn-pre-batch { background: #2563eb !important; color: #ffffff !important; border-color: #2563eb !important; }
.btn-pre-stop button,
button.btn-pre-stop { background: #dc2626 !important; color: #ffffff !important; border-color: #dc2626 !important; }

.btn-alpha1-caption button,
button.btn-alpha1-caption { background: #7c3aed !important; color: #ffffff !important; border-color: #7c3aed !important; }
.btn-alpha1-batch button,
button.btn-alpha1-batch { background: #ea580c !important; color: #ffffff !important; border-color: #ea580c !important; }
.btn-alpha1-stop button,
button.btn-alpha1-stop { background: #b91c1c !important; color: #ffffff !important; border-color: #b91c1c !important; }

.btn-alpha2-caption button,
button.btn-alpha2-caption { background: #db2777 !important; color: #ffffff !important; border-color: #db2777 !important; }
.btn-alpha2-batch button,
button.btn-alpha2-batch { background: #0f766e !important; color: #ffffff !important; border-color: #0f766e !important; }
.btn-alpha2-stop button,
button.btn-alpha2-stop { background: #be123c !important; color: #ffffff !important; border-color: #be123c !important; }

.btn-beta-caption button,
button.btn-beta-caption { background: #4f46e5 !important; color: #ffffff !important; border-color: #4f46e5 !important; }
.btn-beta-zip button,
button.btn-beta-zip { background: #0891b2 !important; color: #ffffff !important; border-color: #0891b2 !important; }
.btn-beta-folder button,
button.btn-beta-folder { background: #16a34a !important; color: #ffffff !important; border-color: #16a34a !important; }
.btn-beta-option button,
button.btn-beta-option { background: #c2410c !important; color: #ffffff !important; border-color: #c2410c !important; }
.btn-qwen-caption button,
button.btn-qwen-caption { background: #6d28d9 !important; color: #ffffff !important; border-color: #6d28d9 !important; }
.btn-qwen-render button,
button.btn-qwen-render { background: #0f766e !important; color: #ffffff !important; border-color: #0f766e !important; }
.btn-qwen-apply button,
button.btn-qwen-apply { background: #2563eb !important; color: #ffffff !important; border-color: #2563eb !important; }
.btn-qwen-zip button,
button.btn-qwen-zip { background: #0891b2 !important; color: #ffffff !important; border-color: #0891b2 !important; }
.btn-qwen-folder button,
button.btn-qwen-folder { background: #16a34a !important; color: #ffffff !important; border-color: #16a34a !important; }
.btn-json-build button,
button.btn-json-build { background: #7c3aed !important; color: #ffffff !important; border-color: #7c3aed !important; }
.btn-json-add button,
button.btn-json-add { background: #0d9488 !important; color: #ffffff !important; border-color: #0d9488 !important; }
.btn-open-folder button,
button.btn-open-folder { background: #475569 !important; color: #ffffff !important; border-color: #475569 !important; }
.btn-cancel button,
button.btn-cancel { background: #dc2626 !important; color: #ffffff !important; border-color: #dc2626 !important; }

.btn-save-preset button:hover,
.btn-load-preset button:hover,
.btn-reset-preset button:hover,
.btn-delete-preset button:hover,
.btn-refresh button:hover,
.btn-pre-caption button:hover,
.btn-pre-batch button:hover,
.btn-pre-stop button:hover,
.btn-alpha1-caption button:hover,
.btn-alpha1-batch button:hover,
.btn-alpha1-stop button:hover,
.btn-alpha2-caption button:hover,
.btn-alpha2-batch button:hover,
.btn-alpha2-stop button:hover,
.btn-beta-caption button:hover,
.btn-beta-zip button:hover,
.btn-beta-folder button:hover,
.btn-beta-option button:hover,
.btn-qwen-caption button:hover,
.btn-qwen-render button:hover,
.btn-qwen-apply button:hover,
.btn-qwen-zip button:hover,
.btn-qwen-folder button:hover,
.btn-json-build button:hover,
.btn-json-add button:hover,
.btn-open-folder button:hover,
.btn-cancel button:hover {
  filter: brightness(1.06);
}

.jc-topbar {
  align-items: flex-start !important;
  gap: 24px !important;
  margin-bottom: 10px;
}

.jc-brand {
  min-width: min(520px, 100%);
}

.jc-brand h1 {
  margin: 0 0 8px;
  font-size: 28px;
  line-height: 1.15;
}

.jc-brand p {
  margin: 0;
}

.jc-header-status {
  gap: 4px !important;
  margin-top: 18px;
  max-width: 780px;
}

.jc-header-status .block {
  min-height: 0 !important;
  padding: 0 !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}

.jc-header-status .html-container,
.jc-header-status .prose {
  padding: 0 !important;
  max-width: none !important;
}

.jc-header-status .jc-info,
.jc-header-status .jc-success,
.jc-header-status .jc-error {
  margin: 0;
}

.jc-header-status .jc-info pre {
  margin-top: 4px;
}

.jc-preset-panel {
  min-width: min(560px, 100%);
}

.jc-codeish textarea,
.jc-codeish input {
  font-family: "JetBrains Mono", "Consolas", monospace !important;
  font-size: 12px !important;
  line-height: 1.45 !important;
}

.jc-output textarea {
  max-height: 560px !important;
  overflow: auto !important;
  resize: vertical !important;
  white-space: pre !important;
}

.jc-qwen-workspace {
  min-width: 0;
}

.jc-qwen-settings-rail {
  min-width: 320px;
}

.jc-qwen-elements-wide {
  margin-top: 10px;
}

.jc-hidden-sync {
  display: none !important;
}

.block.jc-qwen-status-scroll {
  max-height: 250px !important;
  overflow: auto !important;
  overflow-y: auto !important;
  border: 1px solid rgba(148, 163, 184, 0.22);
  border-radius: 6px;
  background: rgba(2, 6, 23, 0.28);
  padding: 8px;
}

.block.jc-qwen-status-scroll:has(.prose.jc-qwen-status-scroll:empty) {
  display: none !important;
}

.block.jc-qwen-status-scroll .html-container,
.block.jc-qwen-status-scroll .prose.jc-qwen-status-scroll {
  max-height: 232px !important;
  overflow-y: auto !important;
}

.prose.jc-qwen-status-scroll {
  border: 0 !important;
  background: transparent !important;
  padding: 0 !important;
  max-width: none !important;
}

.block.jc-qwen-status-scroll .jc-info,
.block.jc-qwen-status-scroll .jc-success,
.block.jc-qwen-status-scroll .jc-error {
  margin: 0;
}

.jc-qwen-status-bottom {
  margin-top: 10px;
}

.jc-qwen-preview-panel {
  margin-top: 10px;
}

.jc-qwen-box-toolbar {
  align-items: end;
  gap: 8px;
  margin-bottom: 6px;
}

.jc-qwen-box-filter {
  margin-bottom: 8px;
}

.jc-qwen-box-filter fieldset {
  max-height: 118px;
  overflow: auto;
}

.jc-json-table-editor {
  max-height: 320px;
  overflow: auto;
  border: 1px solid rgba(148, 163, 184, 0.28);
  border-radius: 6px;
  background: #0b0d12;
}

.jc-json-builder-boxes-large .jc-json-table-editor {
  min-height: 360px;
  max-height: 560px;
}

.jc-json-table-editor table {
  width: 100%;
  border-collapse: collapse;
  table-layout: fixed;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
}

.jc-json-table-editor th,
.jc-json-table-editor td {
  border: 1px solid rgba(148, 163, 184, 0.24);
  padding: 0;
}

.jc-json-table-editor th {
  position: sticky;
  top: 0;
  z-index: 1;
  padding: 10px 8px;
  background: #111318;
  color: #f8fafc;
  text-align: left;
  font-weight: 800;
}

.jc-json-table-editor input {
  width: 100%;
  min-width: 0;
  height: 38px;
  border: 0;
  border-radius: 0;
  background: transparent;
  color: #f8fafc;
  padding: 6px 8px;
  font: inherit;
}

.jc-json-table-editor input:focus {
  outline: 2px solid #f97316;
  outline-offset: -2px;
  background: rgba(249, 115, 22, 0.1);
}

.jc-json-table-empty {
  padding: 12px;
  color: #94a3b8;
  text-align: center;
}

.jc-qwen-preview-panel .jc-overlay-shell {
  min-height: 360px;
  max-height: 768px;
}

.jc-overlay-shell {
  width: 100%;
  overflow: auto;
  padding: 28px 12px 12px;
  border: 1px solid rgba(148, 163, 184, 0.24);
  border-radius: 8px;
  background: #0f172a;
  text-align: center;
}

.jc-overlay-frame {
  position: relative;
  display: inline-block;
  max-width: 100%;
  margin: 0 auto;
  background: #111827;
  border: 1px solid rgba(148, 163, 184, 0.35);
  box-shadow: inset 0 0 0 1px rgba(255,255,255,0.03);
  line-height: 0;
  vertical-align: top;
}

.jc-overlay-image {
  display: block;
  width: auto;
  max-width: 100%;
  max-height: 768px;
  height: auto;
  object-fit: contain;
}

.jc-overlay-blank {
  width: min(100%, 960px);
  min-height: 320px;
  background-image:
    linear-gradient(rgba(148, 163, 184, 0.12) 1px, transparent 1px),
    linear-gradient(90deg, rgba(148, 163, 184, 0.12) 1px, transparent 1px);
  background-size: 32px 32px;
}

.jc-box {
  position: absolute;
  box-sizing: border-box;
  border: 2px solid;
  border-radius: 4px;
  min-width: 18px;
  min-height: 18px;
  line-height: 1.25;
  text-align: left;
}

.jc-overlay-interactive .jc-box {
  cursor: move;
  touch-action: none;
}

.jc-overlay-interactive .jc-box:hover,
.jc-overlay-interactive .jc-box.is-selected,
.jc-overlay-interactive .jc-box.is-editing {
  box-shadow:
    0 0 0 2px rgba(255, 255, 255, 0.85),
    0 10px 24px rgba(0, 0, 0, 0.35);
  z-index: 20;
}

.jc-box span {
  position: absolute;
  left: -2px;
  top: -24px;
  max-width: min(420px, 80vw);
  padding: 3px 7px;
  border-radius: 4px;
  color: #ffffff;
  font-size: 11px;
  line-height: 1.25;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  font-family: "JetBrains Mono", "Consolas", monospace;
}

.jc-box-handle {
  display: none;
  position: absolute;
  width: 10px;
  height: 10px;
  border: 2px solid #ffffff;
  border-radius: 999px;
  background: #111827;
  box-shadow: 0 0 0 1px rgba(15, 23, 42, 0.85);
  pointer-events: auto;
  z-index: 22;
}

.jc-overlay-interactive .jc-box:hover .jc-box-handle,
.jc-overlay-interactive .jc-box.is-selected .jc-box-handle,
.jc-overlay-interactive .jc-box.is-editing .jc-box-handle {
  display: block;
}

.jc-box-handle-nw {
  left: -7px;
  top: -7px;
  cursor: nwse-resize;
}

.jc-box-handle-n {
  left: 50%;
  top: -7px;
  transform: translateX(-50%);
  cursor: ns-resize;
}

.jc-box-handle-ne {
  right: -7px;
  top: -7px;
  cursor: nesw-resize;
}

.jc-box-handle-e {
  right: -7px;
  top: 50%;
  transform: translateY(-50%);
  cursor: ew-resize;
}

.jc-box-handle-se {
  right: -7px;
  bottom: -7px;
  cursor: nwse-resize;
}

.jc-box-handle-s {
  left: 50%;
  bottom: -7px;
  transform: translateX(-50%);
  cursor: ns-resize;
}

.jc-box-handle-sw {
  left: -7px;
  bottom: -7px;
  cursor: nesw-resize;
}

.jc-box-handle-w {
  left: -7px;
  top: 50%;
  transform: translateY(-50%);
  cursor: ew-resize;
}

.jc-overlay-empty {
  position: absolute;
  inset: 0;
  display: grid;
  place-items: center;
  color: #94a3b8;
  font-size: 13px;
}

.jc-info pre,
.jc-success pre,
.jc-error pre {
  white-space: pre-wrap;
  margin: 6px 0 0;
}
"""
