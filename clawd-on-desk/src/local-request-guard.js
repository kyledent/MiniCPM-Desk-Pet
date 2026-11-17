"use strict";

// The agent server listens on 127.0.0.1, but any web page open in a browser
// on the same PC can still send it a "simple" cross-origin POST (text/plain,
// mode: "no-cors"): no preflight, and the request is processed even though
// the page cannot read the answer. That would let a web page put text in the
// pet's bubble (POST /state with `say`) or answer a coding agent's permission
// prompt (POST /permission).
//
// Browsers always send `Origin` on cross-origin POSTs, and modern ones also
// send `Sec-Fetch-*` headers. The real clients (agent hooks in Node, Python
// and PowerShell, curl) send neither, so a POST carrying them is refused.

function isBrowserRequest(req) {
  const h = (req && req.headers) || {};
  return Boolean(h.origin || h["sec-fetch-site"] || h["sec-fetch-mode"]);
}

function rejectBrowserRequest(res, headers = {}) {
  res.writeHead(403, { "Content-Type": "text/plain", ...headers });
  res.end("browser requests are not accepted");
}

module.exports = { isBrowserRequest, rejectBrowserRequest };
