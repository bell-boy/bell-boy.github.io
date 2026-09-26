(() => {
    document.querySelectorAll(".main img[alt]").forEach((image) => {
        const text = image.alt.trim();
        if (!text || image.closest("figure")) return;

        // Keep a linked image clickable and replace its standalone paragraph.
        const media = image.parentElement.tagName === "A" ? image.parentElement : image;
        const paragraph = media.parentElement;
        if (paragraph.tagName !== "P" || [...paragraph.childNodes].some((node) =>
            node !== media && (node.nodeType !== Node.TEXT_NODE || node.textContent.trim())
        )) return;

        const figure = document.createElement("figure");
        const caption = document.createElement("figcaption");
        caption.className = "image-caption";
        caption.textContent = text;
        paragraph.replaceWith(figure);
        figure.append(media, caption);
    });
})();
