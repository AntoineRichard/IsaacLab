// Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
// All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

// Click-to-copy handler for the host-IP cell in Tab A's jobs table.
// Pure client-side — no Dash callback round-trip. Reads the IP from
// the `.tab-a-host-text` span that immediately precedes the button so
// the host string lives in exactly one place in the DOM (the cell
// itself), with no extra data-* attribute to keep in sync.
document.addEventListener("click", (event) => {
    const button = event.target;
    if (!button || !button.classList || !button.classList.contains("tab-a-host-copy")) {
        return;
    }
    const span = button.previousElementSibling;
    if (!span) {
        return;
    }
    const value = span.textContent ? span.textContent.trim() : "";
    if (!value || value === "—") {
        return;
    }
    if (!navigator.clipboard || !navigator.clipboard.writeText) {
        // Older browsers — let the click bubble; we won't crash silently.
        return;
    }
    navigator.clipboard.writeText(value).then(() => {
        // Brief visual ack: swap the icon to a checkmark for ~0.6s so the
        // operator knows the copy landed without us needing a toast.
        const original = button.textContent;
        button.textContent = "✓";
        button.classList.add("tab-a-host-copy-flash");
        setTimeout(() => {
            button.textContent = original;
            button.classList.remove("tab-a-host-copy-flash");
        }, 600);
    }).catch(() => {
        // Permissions denied / non-secure context — silently ignore.
    });
});
