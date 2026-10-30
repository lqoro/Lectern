// Renders promo/ad.html frame by frame against a running Lectern (redesigned UI) and pipes it to ffmpeg.
//   node render.js out.mp4                      full 20 s, 30 fps, silent
//   node render.js --stills 3,6,9.7 outdir      PNGs at those times (for checking)
// Env: LECTERN_URL (default http://127.0.0.1:8477), ASSETS (folder with albert-sans-latin.woff2, icon.png, demo.json)
const { chromium } = require("playwright");
const { spawn } = require("child_process");
const fs = require("fs");
const path = require("path");

const URL_ = process.env.LECTERN_URL || "http://127.0.0.1:8477";
const ASSETS = process.env.ASSETS || path.join(__dirname, "assets");
const FPS = 30, DURATION = 20;
const TYPES = { html: "text/html", woff2: "font/woff2", png: "image/png", json: "application/json" };

(async () => {
  const args = process.argv.slice(2);
  const stills = args[0] === "--stills" ? args[1].split(",").map(Number) : null;
  const out = stills ? args[2] : args[0] || "ad-silent.mp4";

  const browser = await chromium.launch();
  const ctx = await browser.newContext({ viewport: { width: 1080, height: 1920 }, deviceScaleFactor: 1, colorScheme: "dark" });
  const page = await ctx.newPage();
  page.on("pageerror", (e) => console.error("page error:", e.message));
  await page.route("**/__ad/*", (route) => {
    const name = new URL(route.request().url()).pathname.split("/").pop();
    const file = name === "ad.html" ? path.join(__dirname, "ad.html") : path.join(ASSETS, name);
    if (!fs.existsSync(file)) return route.fulfill({ status: 404, body: "missing " + name });
    route.fulfill({ body: fs.readFileSync(file), contentType: TYPES[name.split(".").pop()] || "application/octet-stream" });
  });
  await page.goto(URL_ + "/__ad/ad.html");
  await page.evaluate(() => document.fonts.ready);
  await page.evaluate(() => window.initAd());

  const frame = async (t) => {
    const cur = await page.evaluate((t) => window.layout(t), t);
    if (cur) await page.mouse.move(cur.x, cur.y);           // real hover over the iframe
    await page.evaluate((t) => window.appState(t), t);
    await page.evaluate((t) => window.layout(t), t);        // layout again: sizes may depend on the app state
  };

  if (stills) {
    fs.mkdirSync(out, { recursive: true });
    for (const t of stills) {
      for (const dt of [-0.05, 0]) await frame(Math.max(0, t + dt));   // let hover settle
      await page.waitForTimeout(80);
      await page.screenshot({ path: path.join(out, `t${String(t).replace(".", "_")}.png`) });
      console.log("still", t);
    }
    await browser.close();
    return;
  }

  const ff = spawn("ffmpeg", ["-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", String(FPS), "-c:v", "mjpeg", "-i", "-",
    "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], { stdio: ["pipe", "inherit", "inherit"] });
  const total = FPS * DURATION;
  for (let f = 0; f < total; f++) {
    await frame(f / FPS);
    const buf = await page.screenshot({ type: "jpeg", quality: 96 });
    if (!ff.stdin.write(buf)) await new Promise((r) => ff.stdin.once("drain", r));
    if (f % 60 === 0) console.log(`frame ${f}/${total}`);
  }
  ff.stdin.end();
  await new Promise((r) => ff.on("close", r));
  await browser.close();
  console.log("wrote", out);
})();
