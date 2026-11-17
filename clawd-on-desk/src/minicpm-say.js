"use strict";

// Verbatim bubble text ("say") for integrations that are not coding agents.
//
// A POST /state body may carry `say`: a short message the pet shows as-is
// in its bubble, without asking the model to narrate it. Narration is
// built for coding agents ("The AI suggested ..."), so a message such as
// "Validation finished: 4 presets captured" comes back rewritten and often
// wrong. With `say` the model is not involved at all, so it also works
// while the sidecar or model is down.
//
// GET /state advertises the capability as `features: ["say"]`, so a client
// can tell this build from one that would narrate the text instead.

const SAY_FEATURE = "say";
const SAY_CHAR_LIMIT = 140;

// Control characters (C0 except tab/newline, DEL, C1) never reach the bubble.
const CONTROL_RE = /[\u0000-\u0008\u000B-\u001F\u007F-\u009F]/g;

function sayText(data, limit = SAY_CHAR_LIMIT) {
  if (!data || typeof data !== "object" || typeof data.say !== "string") return "";
  let text = data.say.replace(CONTROL_RE, "");
  // One paragraph: the bubble is small, and a newline is easy to abuse.
  text = text.replace(/\s+/g, " ").trim();
  if (!text) return "";
  const chars = Array.from(text); // count code points, not UTF-16 units
  if (chars.length > limit) text = chars.slice(0, limit - 1).join("").trimEnd() + "…";
  return text;
}

module.exports = { SAY_FEATURE, SAY_CHAR_LIMIT, sayText };
