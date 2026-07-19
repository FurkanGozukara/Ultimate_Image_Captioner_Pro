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
.btn-add-replace-pair button,
button.btn-add-replace-pair { background: #0f766e !important; color: #ffffff !important; border-color: #0f766e !important; }

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
.btn-add-replace-pair button:hover,
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

.jc-caption-output textarea {
  white-space: pre-wrap !important;
  overflow-wrap: anywhere !important;
  word-break: break-word !important;
}

.jc-replace-pairs .block,
.jc-replace-pairs .html-container,
.jc-replace-pairs .prose {
  min-height: 0 !important;
  padding: 0 !important;
}

.jc-replace-empty {
  padding: 8px 10px;
  border: 1px dashed rgba(148, 163, 184, 0.45);
  border-radius: 6px;
  color: #94a3b8;
  font-size: 12px;
}

.jc-replace-list {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 4px;
}

.jc-replace-chip {
  display: inline-flex;
  align-items: center;
  max-width: 100%;
  gap: 6px;
  padding: 5px 6px 5px 9px;
  border: 1px solid rgba(148, 163, 184, 0.35);
  border-radius: 6px;
  background: rgba(15, 23, 42, 0.32);
  font-size: 12px;
  line-height: 1.2;
}

.jc-replace-find,
.jc-replace-to {
  overflow-wrap: anywhere;
}

.jc-replace-find {
  font-weight: 700;
}

.jc-replace-arrow {
  color: #94a3b8;
}

.jc-replace-remove {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 22px;
  height: 22px;
  min-width: 22px;
  border: 0;
  border-radius: 4px;
  background: #dc2626;
  color: #ffffff;
  cursor: pointer;
  font-size: 12px;
  font-weight: 800;
  line-height: 1;
}

.jc-replace-remove:hover {
  background: #b91c1c;
}

.jc-generated-json-header {
  align-items: center !important;
  gap: 8px !important;
  flex-wrap: nowrap !important;
  margin-bottom: 6px;
}

.jc-generated-json-title {
  flex: 1 1 auto;
  min-width: 0;
}

.jc-generated-json-title .prose {
  margin: 0 !important;
}

.jc-copy-json-control {
  flex: 0 0 auto !important;
  width: auto !important;
  min-width: 0 !important;
  margin-left: auto;
}

.jc-copy-json-control .block,
.jc-copy-json-control .html-container,
.jc-copy-json-control .prose {
  min-height: 0 !important;
  padding: 0 !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}

.jc-copy-json-button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 7px;
  height: 36px;
  min-width: 0;
  padding: 0 12px;
  border: 1px solid #0891b2;
  border-radius: 6px;
  background: #0891b2;
  color: #ffffff;
  cursor: pointer;
  font-family: inherit;
  font-size: 13px;
  font-weight: 700;
  line-height: 1;
  white-space: nowrap;
  transition: background-color 120ms ease, border-color 120ms ease, color 120ms ease, transform 120ms ease;
}

.jc-copy-json-button:hover {
  border-color: #0e7490;
  background: #0e7490;
  color: #ffffff;
}

.jc-copy-json-button:active {
  transform: translateY(1px);
}

.jc-copy-json-button.is-copied {
  border-color: #16a34a;
  background: #16a34a;
  color: #ffffff;
}

.jc-copy-json-button.is-failed {
  border-color: #dc2626;
  background: #dc2626;
  color: #ffffff;
}

.jc-copy-json-button svg {
  width: 15px;
  height: 15px;
  fill: none;
  stroke: currentColor;
  stroke-width: 2;
  stroke-linecap: round;
  stroke-linejoin: round;
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

.jc-qwen-elements-wide [data-col="5"] .wrap,
.jc-qwen-elements-wide [data-col="7"] .wrap {
  white-space: normal !important;
  overflow-wrap: anywhere !important;
  word-break: break-word !important;
}

.jc-qwen-elements-wide [data-col="5"] .cell-wrap,
.jc-qwen-elements-wide [data-col="7"] .cell-wrap {
  align-items: flex-start !important;
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
  --jc-json-col-type: 68px;
  --jc-json-col-bbox: 70px;
  --jc-json-col-caption: 480px;
  --jc-json-col-box-title: 108px;
  --jc-json-col-text: 300px;
  --jc-json-table-width: 1236px;
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
  width: var(--jc-json-table-width);
  min-width: var(--jc-json-table-width);
  border-collapse: collapse;
  table-layout: fixed;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
}

.jc-json-table-editor .jc-json-col-type {
  width: var(--jc-json-col-type);
}

.jc-json-table-editor .jc-json-col-bbox {
  width: var(--jc-json-col-bbox);
}

.jc-json-table-editor .jc-json-col-caption {
  width: var(--jc-json-col-caption);
}

.jc-json-table-editor .jc-json-col-box-title {
  width: var(--jc-json-col-box-title);
}

.jc-json-table-editor .jc-json-col-text {
  width: var(--jc-json-col-text);
}

.jc-json-table-editor th,
.jc-json-table-editor td {
  border: 1px solid rgba(148, 163, 184, 0.24);
  padding: 0;
  vertical-align: top;
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
  white-space: normal;
  overflow-wrap: anywhere;
}

.jc-json-table-editor input,
.jc-json-table-editor textarea {
  width: 100%;
  min-width: 0;
  min-height: 38px;
  border: 0;
  border-radius: 0;
  background: transparent;
  color: #f8fafc;
  padding: 6px 8px;
  font: inherit;
  box-sizing: border-box;
}

.jc-json-table-editor input {
  height: 38px;
}

.jc-json-table-editor textarea {
  display: block;
  height: auto;
  max-height: 160px;
  line-height: 1.4;
  resize: vertical;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  word-break: break-word;
}

.jc-json-table-editor input:focus,
.jc-json-table-editor textarea:focus {
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

.jc-json-boxed-preview:has(.jc-boxed-preview:empty),
.jc-json-boxed-preview .html-container:empty,
.jc-json-boxed-preview .prose:empty {
  display: none !important;
}

.jc-boxed-preview {
  margin-top: 10px;
  padding: 10px;
  border: 1px solid rgba(148, 163, 184, 0.24);
  border-radius: 8px;
  background: #0f172a;
}

.jc-boxed-preview-title {
  margin-bottom: 8px;
  color: #cbd5e1;
  font-size: 12px;
  font-weight: 700;
}

.jc-loaded-output textarea,
.jc-loaded-output input {
  font-family: "JetBrains Mono", "Consolas", monospace;
  font-size: 12px !important;
}

.jc-boxed-preview-source {
  margin-bottom: 8px;
  color: #94a3b8;
  font-family: "JetBrains Mono", "Consolas", monospace;
  font-size: 11px;
  line-height: 1.35;
  overflow-wrap: anywhere;
}

.jc-boxed-preview img {
  display: block;
  width: auto;
  max-width: 100%;
  max-height: 520px;
  height: auto;
  margin: 0 auto;
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
