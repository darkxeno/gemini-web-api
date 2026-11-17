/* Gemini Cookie Provider -- MV3 service worker.
 *
 * The passive half of GEMINI_BACKEND=api: the server talks to Gemini over HTTP
 * (StreamGenerate) itself, and this extension only supplies the browser's Google
 * session. No content scripts and no open tabs: it reads the .google.com cookie
 * jar with chrome.cookies and POSTs it to http://localhost:<port>/v1/cookies
 *
 *   - when the worker starts (browser start, extension install/reload);
 *   - whenever __Secure-1PSID or __Secure-1PSIDTS changes;
 *   - every minute while the server runs, when the jar changed since its last
 *     push or the server restarted (its /health says no push yet): a rotation
 *     made while the server was down reaches it within a minute of its start;
 *   - every 8 minutes, right after asking Google to rotate __Secure-1PSIDTS
 *     (accounts.google.com/RotateCookies, through Chrome's own network stack and
 *     cookie jar, as an open Google page would). Chrome owns the session, so the
 *     server never rotates it: two rotators would log each other out.
 *
 * The whole path=/ .google.com jar goes, not only the two 1PSID cookies: another
 * signed-in account (u/N) is authenticated by SID/SAPISID and friends.
 *
 * Port: 8100, or chrome.storage.local {port: N}. The manifest's host permission
 * covers localhost:8100 only; another port needs it added there too.
 */

const DEFAULT_PORT = 8100;
const SESSION_COOKIES = ["__Secure-1PSID", "__Secure-1PSIDTS"];
const ROTATE_URL = "https://accounts.google.com/RotateCookies";
const ROTATE_EVERY_MIN = 8;
const LOG = (...a) => console.log("[GCP]", ...a);

// The last body the server accepted, so unchanged jars aren't re-sent. In
// storage.session: the worker idles out every ~30s and would forget it.
let pushTimer = null;
let lastRotation = null; // {status, at}

async function getLastSent() {
  try {
    return (await chrome.storage.session.get(["lastSent"])).lastSent || "";
  } catch (e) {
    return "";
  }
}

async function setLastSent(body) {
  try {
    await chrome.storage.session.set({ lastSent: body });
  } catch (e) {}
}

async function serverPort() {
  try {
    const { port } = await chrome.storage.local.get(["port"]);
    return port || DEFAULT_PORT;
  } catch (e) {
    return DEFAULT_PORT;
  }
}

async function googleJar() {
  const all = await chrome.cookies.getAll({ domain: "google.com" });
  const jar = {};
  for (const c of all) {
    if ((c.domain === ".google.com" || c.domain === "google.com") && c.path === "/") {
      jar[c.name] = c.value;
    }
  }
  return jar;
}

async function push(reason, { force = false } = {}) {
  const cookies = await googleJar();
  if (!cookies["__Secure-1PSID"]) {
    LOG(`${reason}: no __Secure-1PSID -- sign in to Google in this profile`);
    return;
  }
  const body = JSON.stringify({
    secure_1psid: cookies["__Secure-1PSID"],
    secure_1psidts: cookies["__Secure-1PSIDTS"] || null,
    cookies,
    rotation: lastRotation,
  });
  if (!force && body === (await getLastSent())) return;
  const url = `http://localhost:${await serverPort()}/v1/cookies`;
  try {
    const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body });
    if (r.ok) await setLastSent(body);
    LOG(`${reason}: pushed ${Object.keys(cookies).length} cookies -> HTTP ${r.status}`);
  } catch (e) {
    LOG(`${reason}: server not reachable at ${url} (${e.message}); the next change or alarm retries`);
  }
}

// Several cookies change together on a rotation or sign-in: push once they settle.
function pushSoon(reason) {
  clearTimeout(pushTimer);
  pushTimer = setTimeout(() => push(reason), 2000);
}

async function rotate() {
  try {
    // The same request gemini_webapi makes; the response's Set-Cookie lands in
    // Chrome's jar (credentials: include, host permission on google.com).
    const r = await fetch(ROTATE_URL, {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: '[000,"-0000000000000000000"]',
    });
    lastRotation = { status: r.status, at: Date.now() };
    LOG(`RotateCookies -> HTTP ${r.status}`);
  } catch (e) {
    lastRotation = { status: 0, at: Date.now(), error: String(e.message || e) };
    LOG(`RotateCookies failed: ${e.message}`);
  }
}

chrome.cookies.onChanged.addListener(({ cookie, removed }) => {
  if (removed || !SESSION_COOKIES.includes(cookie.name)) return;
  if (!cookie.domain.endsWith("google.com")) return;
  pushSoon(`${cookie.name} changed`);
});

// Feed a running server what it lacks: a changed jar, or everything if it restarted.
async function sync() {
  let health = null;
  try {
    const r = await fetch(`http://localhost:${await serverPort()}/health`);
    if (r.ok) health = await r.json();
  } catch (e) {
    return; // not running: nothing to feed
  }
  if (!health) return;
  await push("sync", { force: !health.cookies_pushed_since_start });
}

chrome.alarms.onAlarm.addListener(async (alarm) => {
  if (alarm.name === "gcp-sync") {
    await sync();
  } else if (alarm.name === "gcp-rotate") {
    await rotate();
    await push("periodic", { force: true });
  }
});

// Created only when missing: re-creating an alarm restarts its period, and the
// worker wakes every minute -- the 8-minute rotation would never come due.
async function ensureAlarm(name, periodInMinutes) {
  if (!(await chrome.alarms.get(name))) chrome.alarms.create(name, { periodInMinutes });
}

function start(reason) {
  ensureAlarm("gcp-rotate", ROTATE_EVERY_MIN);
  ensureAlarm("gcp-sync", 1);
  push(reason, { force: true });
}

// Every worker start (also after it idled out): the alarms must exist.
ensureAlarm("gcp-rotate", ROTATE_EVERY_MIN);
ensureAlarm("gcp-sync", 1);

chrome.runtime.onInstalled.addListener(() => start("installed/reloaded"));
chrome.runtime.onStartup.addListener(() => start("browser-startup"));
