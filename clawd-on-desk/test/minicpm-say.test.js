"use strict";

// Verbatim bubble text (`say`): shown as sent, without model narration.

const { describe, it } = require("node:test");
const assert = require("node:assert");

const { SAY_FEATURE, SAY_CHAR_LIMIT, sayText } = require("../src/minicpm-say");

describe("sayText", () => {
  it("returns the text as sent", () => {
    assert.strictEqual(sayText({ say: "Validation finished: 4 presets captured" }), "Validation finished: 4 presets captured");
  });

  it("is empty without a usable say field", () => {
    assert.strictEqual(sayText(null), "");
    assert.strictEqual(sayText({}), "");
    assert.strictEqual(sayText({ say: 42 }), "");
    assert.strictEqual(sayText({ say: "   \n\t " }), "");
    assert.strictEqual(sayText({ assistant_last_output: "narrate me" }), "");
  });

  it("flattens whitespace and drops control characters", () => {
    assert.strictEqual(sayText({ say: "  Slot 3:\n\tatlas   done\u0007 " }), "Slot 3: atlas done");
    assert.strictEqual(sayText({ say: "a\u009Bb\u007Fc" }), "abc");
  });

  it("caps long text with an ellipsis, counting code points", () => {
    const long = "x".repeat(SAY_CHAR_LIMIT + 20);
    const out = sayText({ say: long });
    assert.strictEqual(Array.from(out).length, SAY_CHAR_LIMIT);
    assert.ok(out.endsWith("…"));
    const exact = "y".repeat(SAY_CHAR_LIMIT);
    assert.strictEqual(sayText({ say: exact }), exact);
    const emoji = "\u{1F43E}".repeat(SAY_CHAR_LIMIT); // surrogate pairs
    assert.strictEqual(sayText({ say: emoji }), emoji);
  });

  it("honours a custom limit", () => {
    assert.strictEqual(sayText({ say: "abcdef" }, 4), "abc…");
  });

  it("advertises the feature name clients look for", () => {
    assert.strictEqual(SAY_FEATURE, "say");
  });
});
