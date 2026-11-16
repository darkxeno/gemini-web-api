(() => {
  if (window.__geminiNetHookInstalled) return;
  window.__geminiNetHookInstalled = true;

  let activeStreams = 0;
  const emit = () => {
    window.dispatchEvent(
      new CustomEvent("gemini-stream-state", {
        detail: { active: activeStreams > 0, count: activeStreams }
      })
    );
  };

  const isGenUrl = (url) =>
    typeof url === "string" &&
    (url.includes("StreamGenerate") || url.includes("assistant.lamda"));

  // 1. Hook window.fetch (usado por Gemini Web para StreamGenerate)
  const origFetch = window.fetch;
  window.fetch = async function (...args) {
    const req = args[0];
    const url = typeof req === "string" ? req : (req && req.url) || "";
    const tracked = isGenUrl(url);

    if (tracked) {
      activeStreams += 1;
      emit();
    }

    try {
      const res = await origFetch.apply(this, args);
      if (tracked && res && res.body && typeof res.body.getReader === "function") {
        const reader = res.body.getReader();
        const stream = new ReadableStream({
          async pull(controller) {
            try {
              const { done, value } = await reader.read();
              if (done) {
                activeStreams = Math.max(0, activeStreams - 1);
                emit();
                controller.close();
                return;
              }
              controller.enqueue(value);
            } catch (err) {
              activeStreams = Math.max(0, activeStreams - 1);
              emit();
              controller.error(err);
            }
          },
          cancel(reason) {
            activeStreams = Math.max(0, activeStreams - 1);
            emit();
            return reader.cancel(reason);
          }
        });
        return new Response(stream, {
          status: res.status,
          statusText: res.statusText,
          headers: res.headers
        });
      }
      if (tracked) {
        activeStreams = Math.max(0, activeStreams - 1);
        emit();
      }
      return res;
    } catch (err) {
      if (tracked) {
        activeStreams = Math.max(0, activeStreams - 1);
        emit();
      }
      throw err;
    }
  };

  // 2. Hook XMLHttpRequest como respaldo
  const origOpen = XMLHttpRequest.prototype.open;
  const origSend = XMLHttpRequest.prototype.send;
  XMLHttpRequest.prototype.open = function (method, url, ...rest) {
    this.__isGeminiGen = isGenUrl(url);
    return origOpen.apply(this, [method, url, ...rest]);
  };
  XMLHttpRequest.prototype.send = function (...args) {
    if (this.__isGeminiGen) {
      activeStreams += 1;
      emit();
      const done = () => {
        if (!this.__geminiDone) {
          this.__geminiDone = true;
          activeStreams = Math.max(0, activeStreams - 1);
          emit();
        }
      };
      this.addEventListener("loadend", done);
      this.addEventListener("error", done);
      this.addEventListener("abort", done);
    }
    return origSend.apply(this, args);
  };
})();
