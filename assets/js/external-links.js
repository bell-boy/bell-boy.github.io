(() => {
    const siteUrl = document.currentScript.dataset.siteUrl;
    const internalOrigins = new Set([window.location.origin]);
    if (siteUrl) internalOrigins.add(new URL(siteUrl, document.baseURI).origin);

    document.querySelectorAll("a[href]").forEach((link) => {
        let url;
        try {
            url = new URL(link.getAttribute("href"), document.baseURI);
        } catch {
            return;
        }
        if (!["http:", "https:"].includes(url.protocol) || internalOrigins.has(url.origin)) return;

        // Explicit targets override the default.
        if (!link.hasAttribute("target")) link.target = "_blank";
        if (link.target === "_blank") link.relList.add("noopener");
    });
})();
