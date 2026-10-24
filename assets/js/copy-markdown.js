(() => {
    const button = document.querySelector(".post-copy-link");
    const source = document.getElementById("post-markdown");
    if (!button || !source || !source.value.trim()) return;

    const label = button.textContent;
    let resetTimer;
    button.hidden = false;

    function copyWithSelection() {
        const selection = document.createElement("textarea");
        selection.value = source.value;
        selection.readOnly = true;
        selection.style.position = "fixed";
        selection.style.opacity = "0";
        document.body.append(selection);
        selection.select();
        try {
            if (!document.execCommand("copy")) throw new Error("Copy failed");
        } finally {
            selection.remove();
            button.focus({ preventScroll: true });
        }
    }

    button.addEventListener("click", async () => {
        clearTimeout(resetTimer);
        button.disabled = true;
        try {
            if (navigator.clipboard && window.isSecureContext) {
                try {
                    await navigator.clipboard.writeText(source.value);
                } catch {
                    copyWithSelection();
                }
            } else {
                copyWithSelection();
            }
            button.textContent = "copied!";
        } catch {
            button.textContent = "copy failed — try again";
        } finally {
            button.disabled = false;
            resetTimer = setTimeout(() => { button.textContent = label; }, 2500);
        }
    });
})();
