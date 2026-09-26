(() => {
    const main = document.querySelector(".main");
    const list = main?.querySelector(".footnotes > ol");
    if (!list) return;

    const references = [...main.querySelectorAll("a.footnote")];
    const notes = [...list.children].map((note) => ({
        note,
        reference: references.find((link) =>
            decodeURIComponent(link.hash.slice(1)) === note.id
        ),
    }));
    if (!notes.length || notes.some(({ reference }) => !reference)) return;

    const wideScreen = window.matchMedia("screen and (min-width: 1025px)");
    function positionNotes() {
        main.classList.toggle("has-margin-footnotes", wideScreen.matches);
        if (!wideScreen.matches) return;

        const mainTop = main.getBoundingClientRect().top;
        let nextTop = 0;
        for (const { note, reference } of notes) {
            const top = Math.max(reference.getBoundingClientRect().top - mainTop, nextTop);
            note.style.setProperty("--footnote-top", `${top}px`);
            nextTop = top + note.getBoundingClientRect().height + 16;
        }
        // Reserve room if a long final note extends beyond the article.
        const listTop = list.getBoundingClientRect().top - mainTop;
        list.style.setProperty("--footnotes-overflow", `${Math.max(0, nextTop - listTop)}px`);
    }

    let frame;
    function schedulePosition() {
        cancelAnimationFrame(frame);
        frame = requestAnimationFrame(positionNotes);
    }

    positionNotes();
    window.addEventListener("resize", schedulePosition);
    window.addEventListener("load", schedulePosition);
    document.fonts.ready.then(schedulePosition);
    // Media and MathJax can change paragraph positions after the initial render.
    if ("ResizeObserver" in window) {
        const observer = new ResizeObserver(schedulePosition);
        [...main.children, ...list.children].forEach((element) => observer.observe(element));
    }
})();
