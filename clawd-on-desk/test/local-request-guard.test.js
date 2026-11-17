"use strict";

// Browser requests (Origin / Sec-Fetch-*) are refused on POST /state and
// /permission: a web page must not be able to drive the pet or answer a
// permission prompt. Agent hooks send neither header and are unaffected.

const { describe, it } = require("node:test");
const assert = require("node:assert");
const { EventEmitter } = require("node:events");

const initServer = require("../src/server");
const { isBrowserRequest } = require("../src/local-request-guard");

function makeFakeHttp() {
  let capturedHandler = null;
  function createHttpServer(handler) {
    capturedHandler = handler;
    const server = new EventEmitter();
    server.listen = function () { this.emit("listening"); };
    server.close = function () {};
    return server;
  }
  return { createHttpServer, getHandler: () => capturedHandler };
}

function makeReq(method, url, body, headers) {
  const req = new EventEmitter();
  req.method = method;
  req.url = url;
  req.headers = headers || { "content-type": "application/json" };
  // Emit data/end asynchronously — mirrors real http.IncomingMessage behavior
  setImmediate(() => {
    if (body != null) req.emit("data", Buffer.from(body));
    req.emit("end");
  });
  return req;
}

function callHandler(handler, req) {
  return new Promise((resolve) => {
    const res = {
      statusCode: null,
      headers: {},
      body: "",
      writeHead(code, headers) {
        this.statusCode = code;
        if (headers) this.headers = headers;
      },
      end(data) {
        if (data) this.body += String(data);
        resolve(this);
      },
    };
    handler(req, res);
  });
}

function makeCtx(overrides = {}) {
  const updateSessionCalls = [];
  const setStateCalls = [];

  const ctx = {
    // Required HTTP/server deps — injected to avoid real file/network I/O.
    createHttpServer: null, // caller fills in
    setImmediate: () => {}, // no-op: don't fire sync routines during /state tests
    getPortCandidates: () => [23333],
    writeRuntimeConfig: () => true,
    clearRuntimeConfig: () => true,
    readRuntimePort: () => null,

    // Hook sync impls — keep them all as no-ops
    syncClawdHooksImpl: () => {},
    syncGeminiHooksImpl: () => {},
    syncCursorHooksImpl: () => {},
    syncCodeBuddyHooksImpl: () => {},
    syncKiroHooksImpl: () => {},
    syncQwenHooksImpl: () => {},
    syncOpencodePluginImpl: () => {},

    // /state handler deps
    STATE_SVGS: {
      idle: "x.svg",
      working: "x.svg",
      thinking: "x.svg",
      attention: "x.svg",
      sweeping: "x.svg",
    },
    pendingPermissions: [],
    isAgentEnabled: () => true,
    isAgentPermissionsEnabled: () => true,
    setState: (...args) => setStateCalls.push(args),
    updateSession: (...args) => updateSessionCalls.push(args),
    resolvePermissionEntry: () => {},

    // Telemetry hooks — optional, default to no-ops
    permLog: () => {},
    updateLog: () => {},

    ...overrides,
  };

  return { ctx, updateSessionCalls, setStateCalls };
}

function startServer(overrides) {
  const http = makeFakeHttp();
  const { ctx, updateSessionCalls, setStateCalls } = makeCtx(overrides);
  ctx.createHttpServer = http.createHttpServer;
  const api = initServer(ctx);
  api.startHttpServer();
  return {
    handler: http.getHandler(),
    updateSessionCalls,
    setStateCalls,
    api,
  };
}

describe("local-request-guard", () => {
  it("recognises browser requests by Origin or Sec-Fetch headers", () => {
    assert.strictEqual(isBrowserRequest({ headers: { origin: "http://evil.example" } }), true);
    assert.strictEqual(isBrowserRequest({ headers: { origin: "null" } }), true);
    assert.strictEqual(isBrowserRequest({ headers: { "sec-fetch-site": "cross-site" } }), true);
    assert.strictEqual(isBrowserRequest({ headers: { "sec-fetch-mode": "no-cors" } }), true);
    assert.strictEqual(isBrowserRequest({ headers: { "content-type": "application/json" } }), false);
    assert.strictEqual(isBrowserRequest({}), false);
  });

  const body = JSON.stringify({ state: "attention", session_id: "s1", event: "Stop", say: "Session expired - re-enter your password" });

  it("refuses a cross-site no-cors POST /state and never reaches the narrator", async () => {
    const events = [];
    const { handler, updateSessionCalls } = startServer({ onStateEvent: (d) => events.push(d) });
    const res = await callHandler(handler, makeReq("POST", "/state", body, { origin: "http://evil.example", "content-type": "text/plain;charset=UTF-8", "sec-fetch-mode": "no-cors" }));
    assert.strictEqual(res.statusCode, 403);
    assert.strictEqual(events.length, 0);
    assert.strictEqual(updateSessionCalls.length, 0);
  });

  it("refuses a browser POST /permission", async () => {
    const { handler } = startServer();
    const res = await callHandler(handler, makeReq("POST", "/permission", "{}", { "sec-fetch-site": "cross-site" }));
    assert.strictEqual(res.statusCode, 403);
  });

  it("still accepts an agent hook POST /state (no Origin)", async () => {
    const events = [];
    const { handler, updateSessionCalls } = startServer({ onStateEvent: (d) => events.push(d) });
    const res = await callHandler(handler, makeReq("POST", "/state", body));
    assert.strictEqual(res.statusCode, 200);
    assert.strictEqual(events.length, 1);
    assert.strictEqual(updateSessionCalls.length, 1);
  });
});
